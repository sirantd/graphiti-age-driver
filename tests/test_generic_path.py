"""graphiti's generic query path (seam 2).

What graphiti sends to ``execute_query`` or ``session().run`` outside the two
interfaces runs as it is. Two rules guard that path:

* A FalkorDB or Neo4j function is refused before any connection: such a query
  comes from a provider branch of graphiti that leaked past the interfaces.
* Every other text is compared with the reviewed list. In strict mode an
  unknown text is refused, so the graphiti suite fails on any query that nobody
  reviewed.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import SagaNode

from graphiti_age_driver import AGEDriver, UnsupportedQueryError
from graphiti_age_driver.driver import REVIEWED_QUERIES

# Port 1 refuses every connection, so a query that got past the checks would
# raise a connection error, not UnsupportedQueryError.
OFFLINE = {'host': '127.0.0.1', 'port': 1, 'user': 'x', 'password': 'x', 'dbname': 'x'}


@pytest.mark.parametrize(
    'query',
    [
        "CALL db.idx.fulltext.queryNodes('Entity', $query) YIELD node RETURN node",
        'MATCH (n:Entity) SET n.name_embedding = vecf32($e) RETURN n.uuid AS uuid',
        'MATCH (n) RETURN (2 - vec.cosineDistance(n.e, vecf32($v))) / 2 AS score',
        'CALL db.index.fulltext.queryNodes("node_name_and_summary", $q) YIELD node RETURN node',
        'CALL db.indexes()',
    ],
)
async def test_a_falkordb_function_is_refused_before_any_connection(query):
    driver = AGEDriver(**OFFLINE)

    with pytest.raises(UnsupportedQueryError, match='bypassed the driver interfaces'):
        await driver.execute_query(query, query='x')


async def test_strict_mode_refuses_an_unreviewed_query():
    driver = AGEDriver(**OFFLINE, strict=True)

    with pytest.raises(UnsupportedQueryError, match='Unreviewed'):
        await driver.execute_query('MATCH (n:Entity {uuid: $uuid}) RETURN n.name AS name', uuid='x')


@pytest.mark.parametrize(
    'name', [name for name in REVIEWED_QUERIES if name.startswith('graphiti_core.')]
)
def test_graphiti_core_still_sends_each_reviewed_query(name):
    """A graphiti bump that changes a reviewed text fails here, in CI, before the
    strict suite or production meets the new text."""
    module_name, class_name, function_name = name.rsplit('.', 2)
    function = getattr(getattr(importlib.import_module(module_name), class_name), function_name)

    assert ' '.join(REVIEWED_QUERIES[name].split()) in ' '.join(inspect.getsource(function).split())


@pytest.mark.integration
async def test_an_unreviewed_query_runs_with_one_warning_when_not_strict(caplog, driver):
    lenient = AGEDriver(database=driver.graph, _store=driver._store)
    driver._store.strict = False
    query = 'MATCH (n:Saga) RETURN count(n) AS sagas'

    with caplog.at_level(logging.WARNING, logger='graphiti_age_driver.driver'):
        first, _, _ = await lenient.execute_query(query)
        await lenient.execute_query(query)

    assert first == [{'sagas': 0}]
    assert [r.message for r in caplog.records].count(
        'Unreviewed graphiti query on the generic AGE path: ' + query
    ) == 1


@pytest.mark.integration
async def test_the_saga_lookup_finds_the_saga_and_does_not_create_a_second(driver):
    """Graphiti._get_or_create_saga, the add_episode path for a saga."""
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=Mock(spec=LLMClient),
        embedder=Mock(spec=EmbedderClient),
        cross_encoder=Mock(spec=CrossEncoderClient),
    )
    created_at = datetime(2026, 11, 9, 1, 0, tzinfo=timezone.utc)

    first = await graphiti._get_or_create_saga('weekly sync', 'homelab', created_at)
    again = await graphiti._get_or_create_saga('weekly sync', 'homelab', created_at)
    other_group = await graphiti._get_or_create_saga('weekly sync', 'paperclip', created_at)

    assert again.uuid == first.uuid
    assert again.created_at == created_at
    assert other_group.uuid != first.uuid
    assert len(await SagaNode.get_by_group_ids(driver, ['homelab'])) == 1


@pytest.mark.integration
async def test_get_status_runs_through_a_session_and_returns_none(driver):
    """The MCP get_status tool reads the result only when it is truthy."""
    async with driver.session() as session:
        result = await session.run('MATCH (n) RETURN count(n) as count')

    assert result is None
