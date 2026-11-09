"""One AGE graph for each graphiti group (homelab#332, stories 32 and 37).

graphiti clones the driver for a single group, and the MCP server caches one
clone for each group. The clone reads and writes only its group's graph, which
it creates with its labels and indexes on first use, once.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from conftest import axis, connect, unique_name
from graphiti_age_driver import AGEDriver
from graphiti_age_driver.schema import index_names

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 9, 2, 0, tzinfo=timezone.utc)


def _graphiti(driver: AGEDriver) -> Graphiti:
    return Graphiti(
        graph_driver=driver,
        llm_client=Mock(spec=LLMClient),
        embedder=Mock(spec=EmbedderClient),
        cross_encoder=Mock(spec=CrossEncoderClient),
    )


def _episode(name: str, group: str, valid_at: datetime, content: str = 'c') -> EpisodicNode:
    return EpisodicNode(
        name=name,
        group_id=group,
        created_at=NOW,
        valid_at=valid_at,
        source=EpisodeType.message,
        source_description='conversation message',
        content=content,
        entity_edges=[],
    )


def _graph_indexes(graph: str) -> set[str]:
    with connect() as conn:
        rows = conn.execute('SELECT indexname FROM pg_indexes WHERE schemaname = %s', (graph,))
        return {row[0] for row in rows}


def test_the_default_group_maps_to_the_default_graph(driver):
    """With FALKORDB, graphiti names the default group "_", which AGE refuses
    as a graph name (3 characters at least)."""
    assert driver.clone('_') is driver
    assert driver.clone(driver.graph) is driver
    assert driver.clone('homelab').graph == 'homelab'
    assert driver.clone('homelab').clone('_').graph == driver.graph


def test_a_group_that_cannot_be_a_graph_name_is_refused(driver):
    with pytest.raises(ValueError, match='graph name'):
        driver.clone('a-b')


async def test_a_group_graph_gets_every_index_once_on_first_use(driver):
    group = unique_name('grp')
    created: list[str] = []
    driver._store.statement_hooks.append(
        lambda sql, args, kind: created.append(sql) if 'create_graph' in sql else None
    )

    first, second = driver.clone(group), driver.clone(group)
    await _episode('one', group, NOW).save(first)
    await _episode('two', group, NOW).save(second)

    assert len(created) == 1
    assert set(index_names()) <= _graph_indexes(group)


async def test_a_failed_graph_creation_fails_the_call_and_is_tried_again(driver):
    """A clone whose graph cannot be created must not run unindexed: the first
    statement fails, and the next call tries again."""
    clone = driver.clone('public')  # a schema that exists and is no graph

    for _ in range(2):
        with pytest.raises(Exception, match='public'):
            await _episode('one', 'public', NOW).save(clone)
    assert 'public' not in driver._store.ready_graphs


async def test_a_group_clone_reads_and_writes_only_its_graph(driver):
    group = unique_name('grp')
    clone = driver.clone(group)
    node = EntityNode(name='Alice', group_id=group, labels=['Person'], created_at=NOW)
    node.name_embedding = axis(0)
    await node.save(clone)

    assert [n.uuid for n in await EntityNode.get_by_group_ids(clone, [group])] == [node.uuid]
    assert await EntityNode.get_by_group_ids(driver, [group]) == []


async def test_retrieve_episodes_reads_the_group_graph(driver):
    """graphiti's test_graphiti_retrieve_episodes, with its episodes in the
    group's graph, where add_episode writes them."""
    group = unique_name('grp')
    clone = driver.clone(group)
    episodes = [
        _episode(f'test_episode_{n}', group, NOW - timedelta(days=2 * n), f'Test message {n}')
        for n in (1, 2, 3)
    ]
    for episode in episodes:
        await episode.save(clone)

    found = await _graphiti(driver).retrieve_episodes(
        NOW - timedelta(days=3), last_n=5, group_ids=[group], source=EpisodeType.message
    )

    assert [episode.name for episode in found] == ['test_episode_3', 'test_episode_2']


async def test_clear_data_clears_only_the_drivers_graph(driver):
    """As on FalkorDB: the MCP clear_graph tool passes the group's clone."""
    group = unique_name('grp')
    clone = driver.clone(group)
    await _episode('in group graph', group, NOW).save(clone)
    await _episode('in default graph', group, NOW).save(driver)

    await clear_data(driver, [group])

    assert await EpisodicNode.get_by_group_ids(driver, [group]) == []
    assert [e.name for e in await EpisodicNode.get_by_group_ids(clone, [group])] == ['in group graph']


async def test_an_embedding_of_another_dimension_is_refused(driver):
    """The HNSW index is partial on 1024 dimensions: a 384-dimension row would be
    stored, unindexed and invisible to every search."""
    node = EntityNode(name='Alice', group_id='homelab', created_at=NOW)
    node.name_embedding = [0.1] * 384

    with pytest.raises(ValueError, match='384 dimensions'):
        await node.save(driver)
