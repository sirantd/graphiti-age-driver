"""graphiti's search interface on AGE: SQL on the label tables.

Full-text, vector and breadth-first search read the label tables in SQL, not in
Cypher, because the indexes that serve them are PostgreSQL indexes on
expressions of the properties:

* Full text: a GIN index on ``to_tsvector('english', ...)`` of the fields that
  FalkorDB indexes as FULLTEXT. The query words are graphiti's FalkorDB words
  (the same separators and stop words), joined with OR as on FalkorDB.
  ``ts_rank`` replaces the RediSearch score, so the order of hits changes.
* Vectors: an HNSW index on the embedding property cast to ``vector(N)``. The
  score is graphiti's FalkorDB score ``(2 - cosine distance) / 2``.
* Breadth-first search: a recursive CTE over the ``start_id`` index of the
  RELATES_TO and MENTIONS tables. AGE's variable-length paths cannot name two
  edge labels (``[:A|B*1..3]`` is a syntax error), and they load the graph into
  a cache in each backend.

Each method keeps the rows of graphiti-core 0.31.2's FalkorDB fallback for it:
the same filters, the filter before the limit, the same score threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from graphiti_core.driver.falkordb import STOPWORDS
from graphiti_core.driver.falkordb.fulltext import MAX_QUERY_LENGTH, sanitize_falkor_fulltext_query
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.helpers import validate_group_ids, validate_node_labels
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters
from graphiti_core.search.search_utils import rrf
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings

from . import agtype, records
from .driver import AGEDriver
from .operations import _age
from .schema import (
    COMMUNITY,
    ENTITY,
    EPISODIC,
    LABELS_PROPERTY,
    MENTIONS,
    RELATES_TO,
    TEXT_SEARCH_CONFIG,
    prop,
    prop_text,
    table,
    tsvector_expr,
    vector_dims_expr,
    vector_expr,
)

_STOPWORDS = frozenset(STOPWORDS)

# pgvector: keep scanning the HNSW graph until the LIMIT is met after the
# filters, in exact distance order. Without it an HNSW scan returns at most
# hnsw.ef_search rows before the filters run.
_VECTOR_SETTINGS = {'hnsw.iterative_scan': 'strict_order'}


def fulltext_tsquery(query: str, group_ids: list[str] | None = None) -> str:
    """The ``to_tsquery`` text of graphiti's FalkorDB query: its words, OR-ed.

    '' means no search, as graphiti's FalkorDB builder returns '': no word is
    left after the stop words, or the query is too long.
    """
    validate_group_ids(group_ids)
    words = [
        word
        for word in sanitize_falkor_fulltext_query(query).split()
        if word.lower() not in _STOPWORDS
    ]
    if not words:
        return ''
    if len(' | '.join(words).split(' ')) + len(group_ids or []) >= MAX_QUERY_LENGTH:
        return ''
    # The sanitizer removed every quote and every tsquery operator. Quoted, each
    # word goes through the text-search parser as a phrase of lexemes.
    return ' | '.join(f"'{word}'" for word in words)


def _agtype_text(value: Any) -> str:
    return json.dumps(value)


@dataclass
class SqlFilter:
    """WHERE conditions with named parameters, for one search statement."""

    conditions: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def param(self, value: Any) -> str:
        name = f'p{len(self.params)}'
        self.params[name] = value
        return f'%({name})s'

    def any_of(self, expression: str, values: list[Any]) -> None:
        """``expression`` equals one of ``values`` (a btree index serves it)."""
        placeholder = self.param([_agtype_text(value) for value in values])
        self.conditions.append(f'{expression} = ANY({placeholder}::ag_catalog.agtype[])')

    def entity_labels(self, alias: str, labels: list[str]) -> None:
        """``n:A|B``: the entity has one of the labels. Every entity has
        ``Entity``; the extra labels are in a list property (GIN index)."""
        validate_node_labels(labels)
        if ENTITY in labels:
            return
        placeholder = self.param(list(labels))
        self.conditions.append(f'{prop(alias, LABELS_PROPERTY)} ?| {placeholder}::text[]')

    def dates(self, expression: str, groups: list[list[DateFilter]]) -> None:
        """graphiti's date filter: an OR of AND groups."""
        ors = []
        for group in groups:
            ands = []
            for date_filter in group:
                operator = date_filter.comparison_operator
                if operator in (ComparisonOperator.is_null, ComparisonOperator.is_not_null):
                    ands.append(f'({expression} {operator.value})')
                else:
                    value = self.param(_agtype_text(convert_datetimes_to_strings(date_filter.date)))
                    ands.append(f'({expression} {operator.value} {value}::ag_catalog.agtype)')
            if ands:
                ors.append('(' + ' AND '.join(ands) + ')')
        if ors:
            self.conditions.append('(' + ' OR '.join(ors) + ')')

    def sql(self) -> str:
        return ''.join(f' AND {condition}' for condition in self.conditions)


def node_filter(search_filter: SearchFilters, alias: str = 'n') -> SqlFilter:
    sql_filter = SqlFilter()
    if search_filter.node_labels is not None:
        sql_filter.entity_labels(alias, search_filter.node_labels)
    return sql_filter


def edge_filter(search_filter: SearchFilters) -> SqlFilter:
    """graphiti's edge filters on ``e`` (the edge), ``n`` (source), ``m`` (target).

    📌 graphiti numbers the date parameters by their place in an AND group
    only, so two OR groups overwrite each other's values. Here each value has
    its own parameter. ``property_filters`` is not applied, as on FalkorDB.
    """
    sql_filter = SqlFilter()
    if search_filter.edge_types is not None:
        sql_filter.any_of(prop('e', 'name'), search_filter.edge_types)
    if search_filter.edge_uuids is not None:
        sql_filter.any_of(prop('e', 'uuid'), search_filter.edge_uuids)
    if search_filter.edge_source_node_uuids is not None:
        sql_filter.any_of(prop('n', 'uuid'), search_filter.edge_source_node_uuids)
    if search_filter.edge_target_node_uuids is not None:
        sql_filter.any_of(prop('m', 'uuid'), search_filter.edge_target_node_uuids)
    if search_filter.node_labels is not None:
        sql_filter.entity_labels('n', search_filter.node_labels)
        sql_filter.entity_labels('m', search_filter.node_labels)
    for name in ('valid_at', 'invalid_at', 'created_at', 'expired_at'):
        groups = getattr(search_filter, name)
        if groups is not None:
            sql_filter.dates(prop('e', name), groups)
    return sql_filter


def _props(value: Any) -> dict[str, Any]:
    return agtype.parse(value) if isinstance(value, str) else value


def _without(props: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        props.pop(key, None)
    return props


def _max_distance(min_score: float) -> float:
    """``(2 - d) / 2 > min_score`` (graphiti's FalkorDB score) as ``d < bound``."""
    return 2.0 - 2.0 * min_score


def _vector_text(vector: list[float]) -> str:
    return '[' + ','.join(repr(float(x)) for x in vector) + ']'


def _distance(alias: str, name: str, search_vector: list[float]) -> tuple[str, str]:
    """The dimension condition and the cosine distance to ``%(vector)s``.

    The dimension is the query's own, as a literal: the HNSW index is partial on
    its dimension, and the planner proves the predicate only from a constant.
    The condition comes first, and it is the cheaper qual, so PostgreSQL tests
    it before it casts a stored vector of another dimension.
    """
    dim = len(search_vector)
    return (
        f'{vector_dims_expr(alias, name)} = {dim}',
        f'({vector_expr(alias, name, dim)} <=> %(vector)s::vector)',
    )


_EDGE_ENDS = 'JOIN {entity} AS n ON n.id = e.start_id JOIN {entity} AS m ON m.id = e.end_id'


def _edge_rows_to_edges(rows: list[tuple[Any, ...]]) -> list[Any]:
    return [
        records.entity_edge(_without(_props(row[0]), 'fact_embedding'), row[1], row[2])
        for row in rows
    ]


class AGESearchInterface(SearchInterface):
    """Every method of graphiti-core 0.31.2's search interface."""

    # --------------------------
    # Full text
    # --------------------------

    async def edge_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        age = _age(driver)
        tsquery = fulltext_tsquery(query, group_ids)
        if not tsquery:
            return []
        sql_filter = edge_filter(search_filter)
        if group_ids is not None:
            sql_filter.any_of(prop('e', 'group_id'), group_ids)
        g = age.graph
        document = tsvector_expr('e', RELATES_TO)
        sql = (
            f'SELECT e.properties, {prop_text("n", "uuid")}, {prop_text("m", "uuid")}, '
            f'ts_rank({document}, q) AS score '
            f'FROM {table(g, RELATES_TO)} AS e '
            f'{_EDGE_ENDS.format(entity=table(g, ENTITY))}, '
            f'to_tsquery(%(config)s::regconfig, %(tsquery)s) AS q '
            f'WHERE {document} @@ q{sql_filter.sql()} '
            'ORDER BY score DESC LIMIT %(limit)s'
        )
        params = {
            **sql_filter.params,
            'config': TEXT_SEARCH_CONFIG,
            'tsquery': tsquery,
            'limit': limit,
        }
        return _edge_rows_to_edges(await age._rows(sql, params))

    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        age = _age(driver)
        tsquery = fulltext_tsquery(query, group_ids)
        if not tsquery:
            return []
        sql_filter = node_filter(search_filter)
        if group_ids is not None:
            sql_filter.any_of(prop('n', 'group_id'), group_ids)
        rows = await self._fulltext(age, ENTITY, tsquery, sql_filter, limit)
        return [records.entity_node(_without(_props(row[0]), 'name_embedding')) for row in rows]

    async def episode_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        age = _age(driver)
        tsquery = fulltext_tsquery(query, group_ids)
        if not tsquery:
            return []
        sql_filter = SqlFilter()
        if group_ids is not None:
            sql_filter.any_of(prop('n', 'group_id'), group_ids)
        rows = await self._fulltext(age, EPISODIC, tsquery, sql_filter, limit)
        return [records.episodic_node(_props(row[0])) for row in rows]

    async def community_fulltext_search(
        self,
        driver: Any,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        age = _age(driver)
        tsquery = fulltext_tsquery(query, group_ids)
        if not tsquery:
            return []
        sql_filter = SqlFilter()
        if group_ids is not None:
            sql_filter.any_of(prop('n', 'group_id'), group_ids)
        rows = await self._fulltext(age, COMMUNITY, tsquery, sql_filter, limit)
        return [records.community_node(_props(row[0])) for row in rows]

    @staticmethod
    async def _fulltext(
        age: AGEDriver, label: str, tsquery: str, sql_filter: SqlFilter, limit: int
    ) -> list[tuple[Any, ...]]:
        document = tsvector_expr('n', label)
        sql = (
            f'SELECT n.properties, ts_rank({document}, q) AS score '
            f'FROM {table(age.graph, label)} AS n, '
            'to_tsquery(%(config)s::regconfig, %(tsquery)s) AS q '
            f'WHERE {document} @@ q{sql_filter.sql()} '
            'ORDER BY score DESC LIMIT %(limit)s'
        )
        params = {
            **sql_filter.params,
            'config': TEXT_SEARCH_CONFIG,
            'tsquery': tsquery,
            'limit': limit,
        }
        return await age._rows(sql, params)

    # --------------------------
    # Vectors
    # --------------------------

    async def edge_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        """📌 graphiti applies the source and target arguments only together
        with ``group_ids`` (an indentation of its code). Here each applies when
        it is given."""
        age = _age(driver)
        sql_filter = edge_filter(search_filter)
        if group_ids is not None:
            sql_filter.any_of(prop('e', 'group_id'), group_ids)
        if source_node_uuid is not None:
            sql_filter.any_of(prop('n', 'uuid'), [source_node_uuid])
        if target_node_uuid is not None:
            sql_filter.any_of(prop('m', 'uuid'), [target_node_uuid])
        g = age.graph
        dims, distance = _distance('e', 'fact_embedding', search_vector)
        sql = (
            f'SELECT e.properties, {prop_text("n", "uuid")}, {prop_text("m", "uuid")} '
            f'FROM {table(g, RELATES_TO)} AS e '
            f'{_EDGE_ENDS.format(entity=table(g, ENTITY))} '
            f'WHERE {dims} AND {distance} < %(max_distance)s{sql_filter.sql()} '
            f'ORDER BY {distance} LIMIT %(limit)s'
        )
        params = {
            **sql_filter.params,
            'vector': _vector_text(search_vector),
            'max_distance': _max_distance(min_score),
            'limit': limit,
        }
        return _edge_rows_to_edges(await age._rows(sql, params, _VECTOR_SETTINGS))

    async def node_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[Any]:
        age = _age(driver)
        sql_filter = node_filter(search_filter)
        if group_ids is not None:
            sql_filter.any_of(prop('n', 'group_id'), group_ids)
        rows = await self._similar(age, ENTITY, search_vector, sql_filter, limit, min_score)
        return [records.entity_node(_without(_props(row[0]), 'name_embedding')) for row in rows]

    async def community_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.6,
    ) -> list[Any]:
        age = _age(driver)
        sql_filter = SqlFilter()
        if group_ids is not None:
            sql_filter.any_of(prop('n', 'group_id'), group_ids)
        rows = await self._similar(age, COMMUNITY, search_vector, sql_filter, limit, min_score)
        return [records.community_node(_props(row[0])) for row in rows]

    @staticmethod
    async def _similar(
        age: AGEDriver,
        label: str,
        search_vector: list[float],
        sql_filter: SqlFilter,
        limit: int,
        min_score: float,
    ) -> list[tuple[Any, ...]]:
        dims, distance = _distance('n', 'name_embedding', search_vector)
        sql = (
            f'SELECT n.properties FROM {table(age.graph, label)} AS n '
            f'WHERE {dims} AND {distance} < %(max_distance)s{sql_filter.sql()} '
            f'ORDER BY {distance} LIMIT %(limit)s'
        )
        params = {
            **sql_filter.params,
            'vector': _vector_text(search_vector),
            'max_distance': _max_distance(min_score),
            'limit': limit,
        }
        return await age._rows(sql, params, _VECTOR_SETTINGS)

    async def get_embeddings_for_communities(
        self,
        driver: Any,
        communities: list[Any],
    ) -> dict[str, list[float]]:
        from .operations import _load_embeddings

        uuids = [community.uuid for community in communities]
        return await _load_embeddings(_age(driver), COMMUNITY, 'name_embedding', uuids)

    # --------------------------
    # Breadth-first search
    # --------------------------

    @staticmethod
    def _walk(age: AGEDriver, max_depth_param: str) -> str:
        """The recursive CTE ``walk(origin_group, node_id, depth)``.

        The origins are the entities and episodes with one of the uuids (only
        these two labels start a RELATES_TO or a MENTIONS edge). Each step
        follows one directed RELATES_TO or MENTIONS edge through ``start_id``.
        """
        g = age.graph
        origin_uuid = '= ANY(%(origins)s::ag_catalog.agtype[])'
        return (
            'WITH RECURSIVE origin(id, group_id) AS ('
            f'SELECT o.id, {prop("o", "group_id")} FROM {table(g, ENTITY)} AS o '
            f'WHERE {prop("o", "uuid")} {origin_uuid} '
            'UNION ALL '
            f'SELECT o.id, {prop("o", "group_id")} FROM {table(g, EPISODIC)} AS o '
            f'WHERE {prop("o", "uuid")} {origin_uuid}'
            '), walk(origin_group, node_id, depth) AS ('
            'SELECT group_id, id, 0 FROM origin '
            'UNION '
            'SELECT w.origin_group, step.end_id, w.depth + 1 FROM walk AS w '
            f'JOIN (SELECT start_id, end_id FROM {table(g, RELATES_TO)} '
            f'UNION ALL SELECT start_id, end_id FROM {table(g, MENTIONS)}) AS step '
            'ON step.start_id = w.node_id '
            f'WHERE w.depth < %({max_depth_param})s'
            ') '
        )

    async def node_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        search_filter: SearchFilters,
        bfs_max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """The entities within ``bfs_max_depth`` directed RELATES_TO / MENTIONS
        hops of an origin, in the origin's group. Each entity once (graphiti's
        FalkorDB query returns one row for each path)."""
        if not bfs_origin_node_uuids or bfs_max_depth < 1:
            return []
        age = _age(driver)
        sql_filter = node_filter(search_filter)
        if group_ids is not None:
            sql_filter.any_of(prop('n', 'group_id'), group_ids)
            sql_filter.any_of('w.origin_group', group_ids)
        sql = (
            self._walk(age, 'max_depth') + f'SELECT DISTINCT ON (n.id) n.properties FROM walk AS w '
            f'JOIN {table(age.graph, ENTITY)} AS n ON n.id = w.node_id '
            f'WHERE w.depth >= 1 AND {prop("n", "group_id")} = w.origin_group{sql_filter.sql()} '
            'LIMIT %(limit)s'
        )
        params = {
            **sql_filter.params,
            'origins': [_agtype_text(uuid) for uuid in bfs_origin_node_uuids],
            'max_depth': bfs_max_depth,
            'limit': limit,
        }
        rows = await age._rows(sql, params)
        return [records.entity_node(_without(_props(row[0]), 'name_embedding')) for row in rows]

    async def edge_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        bfs_max_depth: int,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """The RELATES_TO edges on the directed RELATES_TO / MENTIONS paths of
        1 to ``bfs_max_depth`` hops from an origin: each edge that leaves a node
        within ``bfs_max_depth - 1`` hops."""
        if not bfs_origin_node_uuids:
            return []
        age = _age(driver)
        sql_filter = edge_filter(search_filter)
        if group_ids is not None:
            sql_filter.any_of(prop('e', 'group_id'), group_ids)
        g = age.graph
        sql = (
            self._walk(age, 'walk_depth')
            + f'SELECT DISTINCT ON (e.id) e.properties, {prop_text("n", "uuid")}, '
            f'{prop_text("m", "uuid")} FROM walk AS w '
            f'JOIN {table(g, RELATES_TO)} AS e ON e.start_id = w.node_id '
            f'{_EDGE_ENDS.format(entity=table(g, ENTITY))} '
            f'WHERE TRUE{sql_filter.sql()} '
            'LIMIT %(limit)s'
        )
        params = {
            **sql_filter.params,
            'origins': [_agtype_text(uuid) for uuid in bfs_origin_node_uuids],
            'walk_depth': bfs_max_depth - 1,
            'limit': limit,
        }
        return _edge_rows_to_edges(await age._rows(sql, params))

    # --------------------------
    # Rerankers
    # --------------------------

    async def node_distance_reranker(
        self,
        driver: Any,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        """graphiti's FalkorDB reranker: 1 for a direct RELATES_TO neighbour of
        the centre (either direction), 0 for the others, 10 for the centre."""
        age = _age(driver)
        filtered = [uuid for uuid in node_uuids if uuid != center_node_uuid]
        scores: dict[str, float] = {center_node_uuid: 0.0}
        neighbours: set[str] = set()
        for pattern in ('(center)-[:{r}]->(n:{l})', '(n:{l})-[:{r}]->(center)'):
            rows = await age._read(
                f"""
                MATCH (center:{ENTITY}) WHERE center.uuid = $center_uuid
                MATCH {pattern.format(r=RELATES_TO, l=ENTITY)}
                RETURN DISTINCT n.uuid AS uuid
                """,
                {'center_uuid': center_node_uuid},
            )
            neighbours.update(row['uuid'] for row in rows)
        for uuid in filtered:
            scores[uuid] = 1 if uuid in neighbours else float('inf')
        filtered.sort(key=lambda uuid: scores[uuid])
        if center_node_uuid in node_uuids:
            scores[center_node_uuid] = 0.1
            filtered = [center_node_uuid] + filtered
        kept = [uuid for uuid in filtered if 1 / scores[uuid] >= min_score]
        return kept, [1 / scores[uuid] for uuid in kept]

    async def episode_mentions_reranker(
        self,
        driver: Any,
        node_uuids: list[list[str]],
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        """graphiti's reranker: RRF order, then the number of episodes that
        mention each node, most first."""
        age = _age(driver)
        sorted_uuids, _ = rrf(node_uuids)
        scores: dict[str, float] = {}
        if sorted_uuids:
            rows = await age._read(
                f"""
                UNWIND $uuids AS wanted
                MATCH (n:{ENTITY}) WHERE n.uuid = wanted
                MATCH (episode:{EPISODIC})-[:{MENTIONS}]->(n)
                RETURN n.uuid AS uuid, count(*) AS score
                """,
                {'uuids': sorted_uuids},
            )
            scores = {row['uuid']: row['score'] for row in rows}
        for uuid in sorted_uuids:
            scores.setdefault(uuid, 0)
        sorted_uuids.sort(key=lambda uuid: scores[uuid], reverse=True)
        kept = [uuid for uuid in sorted_uuids if scores[uuid] >= min_score]
        return kept, [scores[uuid] for uuid in kept]

    # --------------------------
    # Filter builders
    # --------------------------

    def build_node_search_filters(self, search_filters: SearchFilters) -> Any:
        """The SQL conditions on the entity ``n`` (graphiti core does not call this)."""
        return node_filter(search_filters)

    def build_edge_search_filters(self, search_filters: SearchFilters) -> Any:
        """The SQL conditions on the edge ``e`` and its ends ``n`` and ``m``."""
        return edge_filter(search_filters)
