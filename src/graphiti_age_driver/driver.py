"""The graphiti-core driver for Apache AGE (homelab#332, Stage B).

One AGE graph holds one graphiti group, as one FalkorDB graph does today. One
PostgreSQL database holds all the group graphs, and one connection pool serves
the base driver and all its clones.

The driver reports ``GraphProvider.FALKORDB``. graphiti-core's provider list is
closed, and this value selects what AGE needs: graphiti clones the driver for a
single group, keeps one graph for each group, and stores timestamps as strings.
graphiti's FalkorDB Cypher must never run, so the driver sets both of graphiti's
interface objects and implements every method (``tests/test_surface.py``).

What still arrives at :meth:`AGEDriver.execute_query` is the generic query path.
It runs portable Cypher only:

* 🚨 A FalkorDB-only function (``db.idx.fulltext``, ``vecf32``,
  ``vec.cosineDistance``) is refused with :class:`UnsupportedQueryError`. Such a
  query comes from a FalkorDB branch of graphiti that leaked past the
  interfaces, and it must fail loudly.
* Every other text is compared with :data:`GENERIC_QUERIES`, the reviewed list
  of graphiti-core 0.31.2 and of the MCP server. In strict mode an unknown text
  is refused; otherwise it runs, with one warning for each distinct text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, LiteralString, cast

import psycopg
from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.query_executor import Transaction
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings
from psycopg import errors
from psycopg.conninfo import make_conninfo
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from . import agtype
from .cypher import compose, return_columns, validate_graph_name
from .schema import (
    DEFAULT_EMBEDDING_DIM,
    EDGE_LABELS,
    VERTEX_LABELS,
    index_names,
    index_statements,
)

logger = logging.getLogger(__name__)

# The graph of graphiti's default group. With FALKORDB, graphiti names the
# default group "_", which is not a valid AGE graph name (3 characters at
# least). FalkorDB keeps that group in the graph `default_db`; so does this
# driver, unless the constructor names another graph.
DEFAULT_GRAPH = 'default_db'

# The pool. graphiti runs the writes of one episode concurrently, and the shared
# PostgreSQL app also serves paperclip and cognee, so the pool stays small.
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 8
_POOL_OPEN_TIMEOUT = 15.0

# A write that loses a race to a concurrent writer runs again this many times.
WRITE_ATTEMPTS = 5

_RETRIED_ERRORS = (errors.UniqueViolation, errors.DeadlockDetected, errors.SerializationFailure)
# 🚨 AGE 1.8.0 (cypher_set.c): a SET on a vertex or an edge that a concurrent
# transaction updated waits for that commit, and then FAILS with XX000
# "Entity failed to be updated: 3". A native UPDATE re-checks the row instead.
# Measured in Stage A (sirantd/cognee-age-adapter).
_AGE_UPDATE_CONFLICT = 'Entity failed to be updated'

# Functions that exist on FalkorDB (and Neo4j) only. A query that holds one comes
# from a graphiti provider branch that must never reach AGE.
_FOREIGN_FUNCTIONS = re.compile(
    r'db\.idx\.|db\.index\.|vecf32\s*\(|vec\.cosineDistance|vector\.similarity|'
    r'db\.indexes\s*\(|\bCALL\s+db\.',
    re.IGNORECASE,
)


def _normalize(query: str) -> str:
    return ' '.join(query.split())


# The reviewed generic queries: graphiti-core 0.31.2 and the MCP server at the
# homelab ref 16bf5143 send these texts outside the two interfaces. Each one is
# portable Cypher. ``tests/test_generic_path.py`` runs each one on AGE and checks
# that graphiti-core still holds the text.
GENERIC_QUERIES: frozenset[str] = frozenset(
    _normalize(query)
    for query in (
        # Graphiti._get_or_create_saga (add_episode with a saga). The pattern map
        # is served by the GIN index on the saga properties.
        """
            MATCH (s:Saga {name: $name, group_id: $group_id})
            RETURN s.uuid AS uuid, s.name AS name, s.group_id AS group_id, s.created_at AS created_at,
                   s.first_episode_uuid AS first_episode_uuid, s.last_episode_uuid AS last_episode_uuid
            """,
        # Graphiti.remove_episode (the MCP delete_episode tool): one count for
        # each entity that the episode mentions. 📌 The pattern map on Entity is
        # a scan of the label: no index serves `properties @> {...}` on a label
        # that holds embeddings. A group graph is small and TTL-bounded.
        'MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $uuid}) RETURN count(*) AS episode_count',
        # The MCP get_status tool, through session().run().
        'MATCH (n) RETURN count(n) as count',
    )
)


class UnsupportedQueryError(ValueError):
    """A query that AGE must not run: a FalkorDB-only function, or (strict) an
    unreviewed text on the generic query path."""


def _lost_a_race(exc: Exception) -> bool:
    if isinstance(exc, _RETRIED_ERRORS):
        return True
    return isinstance(exc, errors.InternalError_) and str(exc).startswith(_AGE_UPDATE_CONFLICT)


async def _wait_before_retry(attempt: int, exc: Exception) -> None:
    logger.info('Write lost a race (%s), attempt %d of %d', type(exc).__name__, attempt, WRITE_ATTEMPTS)
    await asyncio.sleep(0.02 * attempt + random.uniform(0, 0.02))


def scrub_nul(value: Any) -> Any:
    """Remove NUL from every string. PostgreSQL text cannot hold it, and one NUL
    fails the whole statement (``UntranslatableCharacter``), as graphiti's
    FalkorDB driver strips it too."""
    if isinstance(value, str):
        return value.replace('\x00', '')
    if isinstance(value, dict):
        return {key: scrub_nul(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_nul(item) for item in value]
    return value


def prepare_params(params: dict[str, Any]) -> str:
    """The agtype parameter map of one statement, as JSON text.

    Datetimes become UTC ISO strings, as graphiti's FalkorDB driver writes them.
    """
    return json.dumps(scrub_nul(convert_datetimes_to_strings(params)), default=str)


SqlArgs = Sequence[Any] | Mapping[str, Any]
StatementHook = Callable[[str, SqlArgs, str], None]


class _Store:
    """The connection pool and the graph registry, shared by a driver and its clones."""

    def __init__(
        self,
        conninfo: str,
        default_graph: str,
        embedding_dim: int,
        embedding_dims_strict: bool,
        strict: bool,
        extra_generic_queries: Iterable[str],
    ):
        self.conninfo = conninfo
        self.default_graph = default_graph
        self.embedding_dim = embedding_dim
        self.embedding_dims_strict = embedding_dims_strict
        self.strict = strict
        self.generic_queries = GENERIC_QUERIES | {_normalize(q) for q in extra_generic_queries}
        self.pool = AsyncConnectionPool(
            conninfo,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            open=False,
            kwargs={'autocommit': True},
            # The shared PostgreSQL app restarts for its own maintenance. A
            # connection from before a restart is replaced, not handed out dead.
            check=AsyncConnectionPool.check_connection,
            name='graphiti-age',
        )
        self.pool_lock = asyncio.Lock()
        self.pool_open = False
        self.ready_graphs: set[str] = set()
        self.graph_locks: dict[str, asyncio.Lock] = {}
        self.warned_queries: set[str] = set()
        # Called with (sql, args, kind) for each statement; kind is "own" or
        # "generic". The plan tests read the statements through it.
        self.statement_hooks: list[StatementHook] = []

    async def open(self) -> None:
        if self.pool_open:
            return
        async with self.pool_lock:
            if self.pool_open:
                return
            try:
                await self.pool.open(wait=True, timeout=_POOL_OPEN_TIMEOUT)
            except PoolTimeout:
                await self.pool.close()
                # The pool says only "initialization incomplete". One direct
                # connection raises the real cause: a wrong host, password or
                # database, or a role without the AGE preload.
                async with await psycopg.AsyncConnection.connect(self.conninfo):
                    pass
                raise
            self.pool_open = True

    async def close(self) -> None:
        await self.pool.close()
        self.pool_open = False
        self.ready_graphs.clear()

    def graph_lock(self, graph: str) -> asyncio.Lock:
        return self.graph_locks.setdefault(graph, asyncio.Lock())


class AGEDriverSession(GraphDriverSession):
    """A session of graphiti's generic query path.

    ``run`` sends graphiti's own Cypher through the generic path and returns
    None, as the FalkorDB session does: the MCP ``get_status`` tool reads the
    result only when it is truthy. ``execute_write`` runs its function in one
    PostgreSQL transaction, and again if the transaction loses a race.
    """

    provider = GraphProvider.FALKORDB

    def __init__(self, driver: AGEDriver):
        self.driver = driver
        self.connection: psycopg.AsyncConnection | None = None

    async def __aenter__(self) -> AGEDriverSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        # The connection belongs to execute_write, which releases it.
        return None

    async def run(self, query: str | list, **kwargs: Any) -> None:
        if isinstance(query, list):
            for cypher_query, params in query:
                await self.driver._generic(str(cypher_query), dict(params), session=self)
        else:
            await self.driver._generic(query, kwargs, session=self)
        return None

    async def execute_write(self, func, *args, **kwargs):
        await self.driver._ready()
        store = self.driver._store
        attempt = 0
        while True:
            attempt += 1
            try:
                async with store.pool.connection() as conn:
                    async with conn.transaction():
                        self.connection = conn
                        try:
                            return await func(self, *args, **kwargs)
                        finally:
                            self.connection = None
            except psycopg.Error as exc:
                if attempt >= WRITE_ATTEMPTS or not _lost_a_race(exc):
                    raise
                await _wait_before_retry(attempt, exc)


class _DriverTransaction(Transaction):
    def __init__(self, session: AGEDriverSession):
        self._session = session

    async def run(self, query: str, **kwargs: Any) -> Any:
        return await self._session.run(query, **kwargs)


class AGEDriver(GraphDriver):
    """A graphiti-core driver backed by Apache AGE graphs in one PostgreSQL database."""

    provider = GraphProvider.FALKORDB
    default_group_id: str = '_'
    # Unused by the driver's own search, which builds its tsquery itself. Kept
    # equal to the FalkorDB value for any graphiti code that reads it.
    fulltext_syntax: str = '@'
    aoss_client: None = None

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        user: str | None = None,
        password: str | None = None,
        dbname: str = 'graphiti',
        database: str = DEFAULT_GRAPH,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        embedding_dims_strict: bool = True,
        strict: bool = False,
        extra_generic_queries: Iterable[str] = (),
        _store: _Store | None = None,
    ):
        """Prepare a pool to PostgreSQL. No connection is made here.

        ``dbname`` is the PostgreSQL database, ``database`` the graph of the
        default group (graphiti's name for the graph is ``database``).
        ``embedding_dim`` fixes the dimension of the HNSW indexes: 1024 for
        bge-m3 in the homelab. With ``embedding_dims_strict`` (the default) a
        write of an embedding of another dimension raises ValueError: such a
        row would be unindexed and invisible to a search at ``embedding_dim``.
        graphiti's own test suite mixes dimensions and turns it off.
        ``strict`` refuses an unreviewed query on the generic path.

        🚨 The driver never sends ``LOAD 'age'`` and never sets the search path.
        The graph role's own settings preload AGE and set the search path. A
        LOAD from a non-superuser fails with ``access to library "age" is not
        allowed``.
        """
        super().__init__()
        self._database = validate_graph_name(database)
        if _store is None:
            settings: dict[str, Any] = {
                'host': host,
                'port': int(port),
                'dbname': dbname,
                'application_name': 'graphiti-age-driver',
            }
            if user:
                settings['user'] = user
            if password:
                settings['password'] = password
            _store = _Store(
                make_conninfo('', **settings),
                self._database,
                int(embedding_dim),
                embedding_dims_strict,
                strict,
                extra_generic_queries,
            )
            self._owns_store = True
        else:
            self._owns_store = False
        self._store = _store

        from .operations import AGEGraphOperations
        from .search import AGESearchInterface

        self.graph_operations_interface = AGEGraphOperations()
        self.search_interface = AGESearchInterface()

    # ------------------------------------------------------------------
    # Graphs
    # ------------------------------------------------------------------

    @property
    def graph(self) -> str:
        """The AGE graph of this driver: the group graph or the default graph."""
        return self._database

    @property
    def embedding_dim(self) -> int:
        return self._store.embedding_dim

    def check_embedding(self, embedding: list[float] | None, what: str) -> None:
        """Refuse an embedding of another dimension (``embedding_dims_strict``)."""
        store = self._store
        if embedding is None or not store.embedding_dims_strict:
            return
        if len(embedding) != store.embedding_dim:
            raise ValueError(
                f'{what} has {len(embedding)} dimensions; the graph indexes '
                f'{store.embedding_dim}. A row of another dimension is invisible to '
                'every similarity search at the indexed dimension.'
            )

    def clone(self, database: str) -> AGEDriver:
        """A driver bound to the graph of one group, on the same pool.

        The graph, its labels and its indexes are created once, on the first
        statement for the group (:meth:`_ready`). The MCP server caches clones
        by group and needs no change.
        """
        if database == self._database:
            return self
        if not database or database == self.default_group_id:
            database = self._store.default_graph
            if database == self._database:
                return self
        return AGEDriver(database=database, _store=self._store)

    def with_database(self, database: str) -> AGEDriver:
        return self.clone(database)

    async def _ready(self) -> None:
        """Make sure that the pool is open and this graph has its labels and indexes."""
        store = self._store
        await store.open()
        graph = self._database
        if graph in store.ready_graphs:
            return
        async with store.graph_lock(graph):
            if graph in store.ready_graphs:
                return
            await self._create_graph()
            store.ready_graphs.add(graph)

    async def _create_graph(self) -> None:
        """Create the graph, each label and each index, if absent.

        🚨 One label in each statement, and each statement in its own
        transaction: 3,000 labels in one transaction failed with ``out of shared
        memory`` (homelab#332). The graph role has no privilege on AGE's catalog
        tables, so it creates blind and accepts ``already exists``.
        """
        graph = self._database
        await self._create_if_absent('SELECT ag_catalog.create_graph(%s)', (graph,))
        for label in VERTEX_LABELS:
            await self._create_if_absent('SELECT ag_catalog.create_vlabel(%s, %s)', (graph, label))
        for label in EDGE_LABELS:
            await self._create_if_absent('SELECT ag_catalog.create_elabel(%s, %s)', (graph, label))
        for statement in index_statements(graph, self._store.embedding_dim):
            await self._create_if_absent(statement, ())

    async def _create_if_absent(self, sql: str, args: Sequence[Any]) -> None:
        """Measured on AGE 1.8.0: a repeated create_graph, create_vlabel or
        create_elabel fails with SQLSTATE 3F000 and ``... already exists``. Two
        processes that create one index at the same moment can collide in the
        catalog with a unique violation; the index then exists."""
        try:
            await self._sql(sql, args)
        except errors.InvalidSchemaName as exc:
            if 'already exists' not in str(exc):
                raise
        except (errors.UniqueViolation, errors.DuplicateTable, errors.DuplicateObject):
            pass

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        """Create the graph and its indexes now. With ``delete_existing``, drop
        the driver's indexes first and create them again."""
        if delete_existing:
            await self.delete_all_indexes()
        self._store.ready_graphs.discard(self._database)
        await self._ready()

    async def delete_all_indexes(self) -> None:
        """Drop every index that the driver made on this graph. AGE's own
        indexes (id, start_id, end_id) stay."""
        await self._store.open()
        for name in index_names():
            await self._sql(f'DROP INDEX IF EXISTS "{self._database}"."{name}"')
        self._store.ready_graphs.discard(self._database)

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    async def _execute(
        self, cursor: psycopg.AsyncCursor, sql: str, args: SqlArgs, kind: str = 'own'
    ) -> None:
        """The one place that sends SQL. The plan tests read what passes here.

        The SQL is composed, not literal: the graph name is validated, the
        Cypher text sits inside its dollar quotes, and every value is a bound
        parameter. The cast says so to the type checker.
        """
        for hook in self._store.statement_hooks:
            hook(sql, args, kind)
        await cursor.execute(cast(LiteralString, sql), args)

    async def _sql(
        self, sql: str, args: SqlArgs = (), *, tx: AGEDriverSession | None = None
    ) -> list[tuple[Any, ...]]:
        """Run one SQL statement; its rows."""
        if tx is not None and tx.connection is not None:
            async with tx.connection.cursor() as cursor:
                await self._execute(cursor, sql, args)
                return await cursor.fetchall() if cursor.description else []
        await self._store.open()
        async with self._store.pool.connection() as conn:
            async with conn.cursor() as cursor:
                await self._execute(cursor, sql, args)
                return await cursor.fetchall() if cursor.description else []

    async def _cypher_on(
        self,
        cursor: psycopg.AsyncCursor,
        query: str,
        params: dict[str, Any] | None,
        kind: str = 'own',
    ) -> tuple[list[dict[str, Any]], list[str]]:
        columns = return_columns(query)
        await self._execute(
            cursor,
            compose(self._database, query, columns),
            (prepare_params(params or {}),),
            kind,
        )
        if not columns:
            return [], []
        rows = [
            {name: agtype.parse(value) for name, value in zip(columns, row, strict=True)}
            for row in await cursor.fetchall()
        ]
        return rows, columns

    async def _statements(
        self,
        statements: list[tuple[str, dict[str, Any] | None]],
        *,
        tx: AGEDriverSession | None = None,
        retry: bool = False,
    ) -> list[list[dict[str, Any]]]:
        """Run Cypher statements in one transaction; the rows of each.

        In a session's ``execute_write`` they run on its transaction, and that
        transaction runs again if it loses a race. Otherwise, with ``retry``,
        this transaction runs again (idempotent writes only).
        """
        await self._ready()
        if tx is not None and tx.connection is not None:
            async with tx.connection.cursor() as cursor:
                return [(await self._cypher_on(cursor, q, p))[0] for q, p in statements]
        attempt = 0
        while True:
            attempt += 1
            try:
                async with self._store.pool.connection() as conn:
                    async with conn.transaction():
                        async with conn.cursor() as cursor:
                            return [
                                (await self._cypher_on(cursor, q, p))[0] for q, p in statements
                            ]
            except psycopg.Error as exc:
                if not retry or attempt >= WRITE_ATTEMPTS or not _lost_a_race(exc):
                    raise
                await _wait_before_retry(attempt, exc)

    async def _read(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return (await self._statements([(query, params)]))[0]

    async def _write(
        self,
        statements: list[tuple[str, dict[str, Any] | None]],
        tx: AGEDriverSession | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Run idempotent writes in one transaction, again if they lose a race."""
        return await self._statements(statements, tx=tx, retry=True)

    async def _rows(
        self, sql: str, args: SqlArgs = (), settings: Mapping[str, str] | None = None
    ) -> list[tuple[Any, ...]]:
        """Run one SQL read of the driver's own (search, traversal) on this graph.

        ``settings`` are set for this statement only (``set_config(..., true)``
        in one transaction).
        """
        await self._ready()
        if not settings:
            return await self._sql(sql, args)
        async with self._store.pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cursor:
                    for name, value in settings.items():
                        await cursor.execute('SELECT set_config(%s, %s, true)', (name, value))
                    await self._execute(cursor, sql, args)
                    return await cursor.fetchall()

    # ------------------------------------------------------------------
    # graphiti's generic query path
    # ------------------------------------------------------------------

    def _check_generic(self, query: str) -> None:
        if _FOREIGN_FUNCTIONS.search(query):
            raise UnsupportedQueryError(
                'This query uses a FalkorDB or Neo4j function, which AGE does not have. '
                'It comes from a graphiti code path that bypassed the driver interfaces:\n'
                f'{query}'
            )
        text = _normalize(query)
        if text in self._store.generic_queries:
            return
        if self._store.strict:
            raise UnsupportedQueryError(f'Unreviewed query on the generic path:\n{query}')
        if text not in self._store.warned_queries:
            self._store.warned_queries.add(text)
            logger.warning('Unreviewed graphiti query on the generic AGE path: %s', text)

    async def _generic(
        self, query: str, params: dict[str, Any], session: AGEDriverSession | None = None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        self._check_generic(query)
        params = {k: v for k, v in params.items() if k not in ('routing_', 'database_')}
        await self._ready()
        if session is not None and session.connection is not None:
            async with session.connection.cursor() as cursor:
                return await self._cypher_on(cursor, query, params, 'generic')
        async with self._store.pool.connection() as conn:
            async with conn.cursor() as cursor:
                return await self._cypher_on(cursor, query, params, 'generic')

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        """Run graphiti's own Cypher; ``(records, columns, None)`` as FalkorDB returns it.

        A returned vertex or edge becomes its property map. The query must name
        each returned item: AGE needs the column list before the query runs.
        """
        records, columns = await self._generic(cypher_query_, kwargs)
        return records, columns, None

    def session(self, database: str | None = None) -> AGEDriverSession:
        driver = self.clone(database) if database else self
        return AGEDriverSession(driver)

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Transaction]:
        """A real transaction for graphiti's namespace API (generic path).

        It cannot run the caller's block again, so it has no race retry, unlike
        ``execute_write``.
        """
        await self._ready()
        session = AGEDriverSession(self)
        async with self._store.pool.connection() as conn:
            async with conn.transaction():
                session.connection = conn
                try:
                    yield _DriverTransaction(session)
                finally:
                    session.connection = None

    async def close(self) -> None:
        """Close the pool. A clone shares the pool of its base driver and does
        not close it: graphiti makes clones for single calls and drops them."""
        if self._owns_store:
            await self._store.close()

    async def health_check(self) -> None:
        await self._rows('SELECT 1')
