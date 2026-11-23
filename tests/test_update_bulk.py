"""graphiti-core 0.32.0: the update-only bulk writes and the group check of a UUID.

Concurrent ingestion writes a stored node or edge through ``node_update_bulk``
and ``edge_update_bulk``. These change what exists and create nothing, so a
node or an edge that another writer deleted stays deleted.

From 0.32.0 a caller that gives a group gets ``NodeNotFoundError`` or
``EdgeNotFoundError`` for a UUID of another group (getzep/graphiti#2178). The
MCP tools ``delete_episode`` and ``get_entity_edge`` send that group.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from conftest import axis, unique_name
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

from graphiti_age_driver.operations import AGEGraphOperations

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 24, 2, 0, tzinfo=UTC)


def _entity(name: str, group: str, index: int) -> EntityNode:
    return EntityNode(
        name=name,
        group_id=group,
        labels=['Entity'],
        name_embedding=axis(index),
        summary=f'{name} summary',
        created_at=NOW,
    )


def _edge(source: EntityNode, target: EntityNode, fact: str) -> EntityEdge:
    return EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='KNOWS',
        fact=fact,
        fact_embedding=axis(7),
        group_id=source.group_id,
        created_at=NOW,
        episodes=[],
    )


def _graphiti(driver) -> Graphiti:
    return Graphiti(
        graph_driver=driver,
        llm_client=Mock(spec=LLMClient),
        embedder=Mock(spec=EmbedderClient),
        cross_encoder=Mock(spec=CrossEncoderClient),
    )


async def test_node_update_bulk_changes_a_stored_node_and_creates_no_absent_node(driver):
    ops = AGEGraphOperations()
    stored = _entity('alice', driver.graph, 0)
    await stored.save(driver)
    absent = _entity('bob', driver.graph, 1)

    stored.summary = 'changed'
    await ops.node_update_bulk(None, driver, None, [stored, absent])

    assert (await EntityNode.get_by_uuid(driver, stored.uuid)).summary == 'changed'
    with pytest.raises(NodeNotFoundError):
        await EntityNode.get_by_uuid(driver, absent.uuid)


async def test_edge_update_bulk_changes_a_stored_edge_and_creates_no_absent_edge(driver):
    ops = AGEGraphOperations()
    alice, bob = _entity('alice', driver.graph, 0), _entity('bob', driver.graph, 1)
    await alice.save(driver)
    await bob.save(driver)
    stored = _edge(alice, bob, 'alice knows bob')
    await stored.save(driver)
    absent = _edge(bob, alice, 'bob knows alice')

    stored.fact = 'alice knows bob well'
    await ops.edge_update_bulk(None, driver, None, [stored.model_dump(), absent.model_dump()])

    assert (await EntityEdge.get_by_uuid(driver, stored.uuid)).fact == 'alice knows bob well'
    with pytest.raises(EdgeNotFoundError):
        await EntityEdge.get_by_uuid(driver, absent.uuid)


async def test_an_edge_uuid_of_another_group_is_not_found(driver):
    group, other = unique_name('grp'), unique_name('oth')
    clone = driver.clone(group)
    alice, bob = _entity('alice', group, 0), _entity('bob', group, 1)
    await alice.save(clone)
    await bob.save(clone)
    edge = _edge(alice, bob, 'alice knows bob')
    await edge.save(clone)

    assert (await EntityEdge.get_by_uuid(clone, edge.uuid, group_id=group)).uuid == edge.uuid
    with pytest.raises(EdgeNotFoundError):
        await EntityEdge.get_by_uuid(driver.clone(other), edge.uuid, group_id=other)


async def test_remove_episode_with_another_group_changes_nothing(driver):
    group, other = unique_name('grp'), unique_name('oth')
    clone = driver.clone(group)
    episode = EpisodicNode(
        name='e1',
        group_id=group,
        created_at=NOW,
        valid_at=NOW,
        source=EpisodeType.message,
        source_description='conversation message',
        content='c',
        entity_edges=[],
    )
    await episode.save(clone)
    graphiti = _graphiti(driver)

    with pytest.raises(NodeNotFoundError):
        await graphiti.remove_episode(episode.uuid, group_id=other)

    assert (await EpisodicNode.get_by_uuid(clone, episode.uuid)).uuid == episode.uuid
