"""Shared fixtures: a driver on throwaway AGE graphs.

The connection settings come from the environment, with the defaults of the
local container in the README:

* ``AGE_HOST`` and ``AGE_PORT`` name the PostgreSQL server.
* ``AGE_USER`` and ``AGE_PASSWORD`` name the graph role. 🚨 It is never a
  superuser: ``suite/test_shadow.py`` fails the suite if it is one.
* ``AGE_DATABASE`` names the graph database, where the extensions exist.
* ``AGE_REQUIRED`` is ``1`` in CI. 🚨 A skipped gate is a green gate. Without a
  server the fixtures skip, which is right on a laptop and a false pass in CI.
  With this flag set, an unreachable server fails the run instead.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, NoReturn

import psycopg
import pytest

from graphiti_age_driver import AGEDriver

HOST = os.getenv('AGE_HOST', '127.0.0.1')
PORT = int(os.getenv('AGE_PORT', '5432'))
USER = os.getenv('AGE_USER', 'graphiti')
PASSWORD = os.getenv('AGE_PASSWORD', 'graphiti')
DATABASE = os.getenv('AGE_DATABASE', 'graphiti')
REQUIRE_SERVER = os.getenv('AGE_REQUIRED', '').strip().lower() in {'1', 'true', 'yes'}

# The production dimension (bge-m3), so the plan tests see the production indexes.
EMBEDDING_DIM = 1024


def connect() -> psycopg.Connection:
    """A plain connection as the graph role, for test setup and assertions."""
    return psycopg.connect(
        host=HOST, port=PORT, user=USER, password=PASSWORD, dbname=DATABASE, autocommit=True
    )


def unreachable(exc: Exception) -> NoReturn:
    message = f'AGE not reachable at {HOST}:{PORT}/{DATABASE} as {USER}: {exc}'
    if REQUIRE_SERVER:
        pytest.fail(message)
    pytest.skip(message)


def drop_graph(name: str) -> None:
    with connect() as conn:
        try:
            conn.execute('SELECT ag_catalog.drop_graph(%s, true)', (name,))
        except psycopg.errors.InvalidSchemaName:
            pass  # the graph was never created


def new_driver(graph: str, **settings: Any) -> AGEDriver:
    options: dict[str, Any] = {'embedding_dim': EMBEDDING_DIM, 'strict': True, **settings}
    return AGEDriver(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        dbname=DATABASE,
        database=graph,
        **options,
    )


def unique_name(prefix: str) -> str:
    return f'{prefix}_{uuid.uuid4().hex[:12]}'


@asynccontextmanager
async def throwaway_driver(prefix: str = 'test', **settings: Any) -> AsyncGenerator[AGEDriver]:
    """A driver whose default graph is new; every graph it made is dropped after."""
    driver = new_driver(unique_name(prefix), **settings)
    try:
        await driver.build_indices_and_constraints()
    except (psycopg.OperationalError, OSError) as exc:
        await driver.close()
        unreachable(exc)
    try:
        yield driver
    finally:
        graphs = set(driver._store.ready_graphs) | {driver.graph}
        await driver.close()
        for graph in graphs:
            drop_graph(graph)


@pytest.fixture
async def driver():
    async with throwaway_driver() as instance:
        yield instance


def axis(index: int, dim: int = EMBEDDING_DIM, weight: float = 1.0) -> list[float]:
    """A test embedding along one axis. Two different axes are orthogonal:
    graphiti's score ``(2 - cosine distance) / 2`` is then 0.5."""
    values = [0.0] * dim
    values[index] = weight
    return values
