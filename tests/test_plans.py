"""The plan gate (seam 4): every statement the driver sends reads its label tables
through an index.

A test of "the index exists" is not enough, because the plan is what silently
changes. So this file runs a workload that calls EVERY method of both
interfaces on a seeded graph, records every statement at the one place the
driver sends SQL (``_execute``), and asks PostgreSQL for the plan of exactly
that statement with its real parameters.

The rules for a plan, as in sirantd/cognee-age-adapter:

* ``enable_seqscan``, ``enable_hashjoin`` and ``enable_mergejoin`` are off. On
  small test data PostgreSQL can choose a sequential scan even when the index is
  correct; with the three off, it takes an index whenever one serves the query.
* ⚠ So a sequential scan is not the only sign of a missing index: the planner
  can read a whole table through an unrelated index instead. The check is
  therefore (a) no ``Disabled: true`` in the plan, and (b) every scan of a label
  table carries an ``Index Cond`` or a ``Recheck Cond``, or is an index scan for
  an ``ORDER BY`` (the HNSW ``Order By``).

``WHOLE_GRAPH`` is the closed list of calls whose job is to read or delete a
whole label. They are in the workload, so the census sees them, but their
plans are not checked. graphiti's own Cypher on the generic path is not
checked here either: ``driver.REVIEWED_QUERIES`` names it, with its plan.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, LiteralString, cast
from uuid import uuid4

import pytest
from conftest import EMBEDDING_DIM, axis, connect
from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode, SagaNode
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters
from psycopg import sql

from graphiti_age_driver import AGEDriver
from graphiti_age_driver.schema import EDGE_LABELS, VERTEX_LABELS

pytestmark = pytest.mark.integration

LABEL_TABLES = (*VERTEX_LABELS, *EDGE_LABELS)
# AGE's default-label tables. An unlabeled pattern such as the HAS_MEMBER target
# (m) also reads them, and the planner may read an empty one through its whole
# index. Every statement of the driver creates a labeled row, so both stay
# empty. The test asserts that instead of their plans.
DEFAULT_LABEL_TABLES = ('_ag_label_vertex', '_ag_label_edge')

# Calls whose job is a read or a delete of a whole label. Asserted by name.
WHOLE_GRAPH = frozenset(
    {
        'clear_data[all groups]',  # MATCH (n:L) DETACH DELETE n
        'remove_communities[all groups]',  # MATCH (c:Community) DETACH DELETE c
        'get_community_clusters[all groups]',  # the distinct group ids of all entities
    }
)

GROUP = 'homelab'
NOW = datetime(2026, 11, 9, 3, 0, tzinfo=UTC)
# Rows of filler on every label, so that a full read costs more than an index
# lookup. Measured in Stage A: on 4 edges the planner read the whole edge table
# through an unrelated index although the start_id index served the query.
FILLER = 1000

Call = Callable[[Any, AGEDriver, dict[str, Any]], Awaitable[Any]]

ENTITY_FILTER = SearchFilters(node_labels=['Person'])
EDGE_FILTER = SearchFilters(
    edge_types=['KNOWS'],
    node_labels=['Person'],
    valid_at=[[DateFilter(date=NOW, comparison_operator=ComparisonOperator.less_than_equal)]],
    invalid_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]],
)


def _operations() -> dict[str, Call]:
    """``{name: call}`` for every method of the graph-operations interface."""
    g = GROUP

    def new_entity() -> EntityNode:
        node = EntityNode(name='new', group_id=g, labels=['Person'], created_at=NOW)
        node.name_embedding = axis(9)
        return node

    absent = str(uuid4())
    return {
        'node_save': lambda o, d, s: o.node_save(new_entity(), d),
        'node_delete': lambda o, d, s: o.node_delete(
            EntityNode(uuid=absent, name='x', group_id=g), d
        ),
        'node_save_bulk': lambda o, d, s: o.node_save_bulk(
            None, d, None, [new_entity(), new_entity()]
        ),
        'node_delete_by_group_id': lambda o, d, s: o.node_delete_by_group_id(None, d, 'absent'),
        'node_delete_by_uuids': lambda o, d, s: o.node_delete_by_uuids(None, d, [absent]),
        'node_get_by_uuid': lambda o, d, s: o.node_get_by_uuid(None, d, s['alice'].uuid),
        'node_get_by_uuids': lambda o, d, s: o.node_get_by_uuids(
            None, d, [s['alice'].uuid, s['bob'].uuid], g
        ),
        'node_get_by_group_ids': lambda o, d, s: o.node_get_by_group_ids(
            None, d, [g], 2, s['bob'].uuid
        ),
        'node_get_by_names': lambda o, d, s: o.node_get_by_names(None, d, ['Alice', 'Bob'], [g]),
        'node_get_by_attribute_values': lambda o, d, s: o.node_get_by_attribute_values(
            None, d, g, {'age': 30}, ['Person'], 5
        ),
        'node_load_embeddings': lambda o, d, s: o.node_load_embeddings(s['alice'], d),
        'node_load_embeddings_bulk': lambda o, d, s: o.node_load_embeddings_bulk(
            d, [s['alice'], s['bob']]
        ),
        'episodic_node_save': lambda o, d, s: o.episodic_node_save(_episode('new', NOW), d),
        'episodic_node_delete': lambda o, d, s: o.episodic_node_delete(
            _episode('x', NOW, uuid=absent), d
        ),
        'episodic_node_save_bulk': lambda o, d, s: o.episodic_node_save_bulk(
            None, d, None, [_episode('b1', NOW), _episode('b2', NOW)]
        ),
        'episodic_edge_save_bulk': lambda o, d, s: o.episodic_edge_save_bulk(
            None,
            d,
            None,
            [_mentions(s['episode_2'], s['carol']).model_dump()],
        ),
        'episodic_node_delete_by_group_id': lambda o, d, s: o.episodic_node_delete_by_group_id(
            None, d, 'absent'
        ),
        'episodic_node_delete_by_uuids': lambda o, d, s: o.episodic_node_delete_by_uuids(
            None, d, [absent]
        ),
        'episodic_node_get_by_uuid': lambda o, d, s: o.episodic_node_get_by_uuid(
            None, d, s['episode_1'].uuid
        ),
        'episodic_node_get_by_uuids': lambda o, d, s: o.episodic_node_get_by_uuids(
            None, d, [s['episode_1'].uuid, s['episode_2'].uuid]
        ),
        'episodic_node_get_by_group_ids': lambda o, d, s: o.episodic_node_get_by_group_ids(
            None, d, [g], 5, None
        ),
        'retrieve_episodes': _all(
            lambda o, d, s: o.retrieve_episodes(d, NOW, 3, [g], EpisodeType.message),
            lambda o, d, s: o.retrieve_episodes(d, NOW, 3, [g], None, 'saga'),
        ),
        'community_node_save': lambda o, d, s: o.community_node_save(_community('new'), d),
        'community_node_delete': lambda o, d, s: o.community_node_delete(
            _community('x', uuid=absent), d
        ),
        'community_node_save_bulk': lambda o, d, s: o.community_node_save_bulk(
            None, d, None, [_community('b1'), _community('b2')]
        ),
        'community_node_delete_by_group_id': lambda o, d, s: o.community_node_delete_by_group_id(
            None, d, 'absent'
        ),
        'community_node_delete_by_uuids': lambda o, d, s: o.community_node_delete_by_uuids(
            None, d, [absent]
        ),
        'community_node_get_by_uuid': lambda o, d, s: o.community_node_get_by_uuid(
            None, d, s['community'].uuid
        ),
        'community_node_get_by_uuids': lambda o, d, s: o.community_node_get_by_uuids(
            None, d, [s['community'].uuid]
        ),
        'community_node_get_by_group_ids': lambda o, d, s: o.community_node_get_by_group_ids(
            None, d, [g], 5, None
        ),
        'saga_node_save': lambda o, d, s: o.saga_node_save(_saga('new'), d),
        'saga_node_delete': lambda o, d, s: o.saga_node_delete(_saga('x', uuid=absent), d),
        'saga_node_save_bulk': lambda o, d, s: o.saga_node_save_bulk(
            None, d, None, [_saga('b1'), _saga('b2')]
        ),
        'saga_node_delete_by_group_id': lambda o, d, s: o.saga_node_delete_by_group_id(
            None, d, 'absent'
        ),
        'saga_node_delete_by_uuids': lambda o, d, s: o.saga_node_delete_by_uuids(None, d, [absent]),
        'saga_node_get_by_uuid': lambda o, d, s: o.saga_node_get_by_uuid(None, d, s['saga'].uuid),
        'saga_node_get_by_uuids': lambda o, d, s: o.saga_node_get_by_uuids(
            None, d, [s['saga'].uuid]
        ),
        'saga_node_get_by_group_ids': lambda o, d, s: o.saga_node_get_by_group_ids(
            None, d, [g], 5, None
        ),
        'saga_get_previous_episode_uuid': lambda o, d, s: o.saga_get_previous_episode_uuid(
            d, s['saga'].uuid, s['episode_2'].uuid
        ),
        'saga_get_episode_contents': _all(
            lambda o, d, s: o.saga_get_episode_contents(d, s['saga'].uuid),
            lambda o, d, s: o.saga_get_episode_contents(d, s['saga'].uuid, NOW - timedelta(days=9)),
        ),
        'edge_save': lambda o, d, s: o.edge_save(_relates(s['alice'], s['carol']), d),
        'edge_delete': lambda o, d, s: o.edge_delete(_relates(s['alice'], s['bob'], absent), d),
        # 🚨 Not bob -> alice: with alice -> bob that is a pair with two edges, and
        # graphiti's label_propagation then swaps their communities forever
        # (get_community_clusters, also on FalkorDB).
        'edge_save_bulk': lambda o, d, s: o.edge_save_bulk(
            None, d, None, [_relates(s['carol'], s['carol']).model_dump()]
        ),
        'edge_delete_by_uuids': lambda o, d, s: o.edge_delete_by_uuids(None, d, [absent]),
        'edge_get_by_uuid': lambda o, d, s: o.edge_get_by_uuid(None, d, s['knows'].uuid),
        'edge_get_by_uuids': lambda o, d, s: o.edge_get_by_uuids(None, d, [s['knows'].uuid]),
        'edge_get_by_group_ids': lambda o, d, s: o.edge_get_by_group_ids(None, d, [g], 5, None),
        'edge_load_embeddings': lambda o, d, s: o.edge_load_embeddings(s['knows'], d),
        'edge_load_embeddings_bulk': lambda o, d, s: o.edge_load_embeddings_bulk(d, [s['knows']]),
        'episodic_edge_save': lambda o, d, s: o.episodic_edge_save(
            _mentions(s['episode_2'], s['alice']), d
        ),
        'episodic_edge_delete': lambda o, d, s: o.episodic_edge_delete(
            _mentions(s['episode_1'], s['alice'], absent), d
        ),
        'episodic_edge_delete_by_uuids': lambda o, d, s: o.episodic_edge_delete_by_uuids(
            None, d, [absent]
        ),
        'episodic_edge_get_by_uuid': lambda o, d, s: o.episodic_edge_get_by_uuid(
            None, d, s['mention'].uuid
        ),
        'episodic_edge_get_by_uuids': lambda o, d, s: o.episodic_edge_get_by_uuids(
            None, d, [s['mention'].uuid]
        ),
        'episodic_edge_get_by_group_ids': lambda o, d, s: o.episodic_edge_get_by_group_ids(
            None, d, [g], 5, None
        ),
        'community_edge_save': lambda o, d, s: o.community_edge_save(
            _member(s['community'], s['bob']), d
        ),
        'community_edge_delete': lambda o, d, s: o.community_edge_delete(
            _member(s['community'], s['bob'], absent), d
        ),
        'community_edge_delete_by_uuids': lambda o, d, s: o.community_edge_delete_by_uuids(
            None, d, [absent]
        ),
        'community_edge_get_by_uuid': lambda o, d, s: o.community_edge_get_by_uuid(
            None, d, s['member'].uuid
        ),
        'community_edge_get_by_uuids': lambda o, d, s: o.community_edge_get_by_uuids(
            None, d, [s['member'].uuid]
        ),
        'community_edge_get_by_group_ids': lambda o, d, s: o.community_edge_get_by_group_ids(
            None, d, [g], 5, None
        ),
        'has_episode_edge_save': lambda o, d, s: o.has_episode_edge_save(
            _has_episode(s['saga'], s['episode_2']), d
        ),
        'has_episode_edge_delete': lambda o, d, s: o.has_episode_edge_delete(
            _has_episode(s['saga'], s['episode_1'], absent), d
        ),
        'has_episode_edge_save_bulk': lambda o, d, s: o.has_episode_edge_save_bulk(
            None, d, None, [_has_episode(s['saga'], s['episode_2'])]
        ),
        'has_episode_edge_delete_by_uuids': lambda o, d, s: o.has_episode_edge_delete_by_uuids(
            None, d, [absent]
        ),
        'has_episode_edge_get_by_uuid': lambda o, d, s: o.has_episode_edge_get_by_uuid(
            None, d, s['has_episode'].uuid
        ),
        'has_episode_edge_get_by_uuids': lambda o, d, s: o.has_episode_edge_get_by_uuids(
            None, d, [s['has_episode'].uuid]
        ),
        'has_episode_edge_get_by_group_ids': lambda o, d, s: o.has_episode_edge_get_by_group_ids(
            None, d, [g], 5, None
        ),
        'next_episode_edge_save': lambda o, d, s: o.next_episode_edge_save(
            _next(s['episode_2'], s['episode_1']), d
        ),
        'next_episode_edge_delete': lambda o, d, s: o.next_episode_edge_delete(
            _next(s['episode_1'], s['episode_2'], absent), d
        ),
        'next_episode_edge_save_bulk': lambda o, d, s: o.next_episode_edge_save_bulk(
            None, d, None, [_next(s['episode_2'], s['episode_1'])]
        ),
        'next_episode_edge_delete_by_uuids': lambda o, d, s: o.next_episode_edge_delete_by_uuids(
            None, d, [absent]
        ),
        'next_episode_edge_get_by_uuid': lambda o, d, s: o.next_episode_edge_get_by_uuid(
            None, d, s['next'].uuid
        ),
        'next_episode_edge_get_by_uuids': lambda o, d, s: o.next_episode_edge_get_by_uuids(
            None, d, [s['next'].uuid]
        ),
        'next_episode_edge_get_by_group_ids': lambda o, d, s: o.next_episode_edge_get_by_group_ids(
            None, d, [g], 5, None
        ),
        'get_mentioned_nodes': lambda o, d, s: o.get_mentioned_nodes(d, [s['episode_1']]),
        'get_communities_by_nodes': lambda o, d, s: o.get_communities_by_nodes(d, [s['alice']]),
        'clear_data': lambda o, d, s: o.clear_data(d, ['absent']),
        'get_community_clusters': lambda o, d, s: o.get_community_clusters(d, [g]),
        'remove_communities': lambda o, d, s: o.remove_communities(d, ['absent']),
        'determine_entity_community': _all(
            lambda o, d, s: o.determine_entity_community(d, s['alice']),
            lambda o, d, s: o.determine_entity_community(d, s['carol']),
        ),
        'episodic_node_get_by_entity_node_uuid': lambda o, d, s: (
            o.episodic_node_get_by_entity_node_uuid(None, d, s['alice'].uuid)
        ),
        'community_node_load_name_embedding': lambda o, d, s: o.community_node_load_name_embedding(
            s['community'], d
        ),
        'edge_get_between_nodes': lambda o, d, s: o.edge_get_between_nodes(
            None, d, s['alice'].uuid, s['bob'].uuid
        ),
        'edge_get_by_node_uuid': lambda o, d, s: o.edge_get_by_node_uuid(None, d, s['bob'].uuid),
        # Last, as they delete whole labels. The cluster call after the deletes:
        # on the filler it would read each of 2,000 clusters with its own query.
        'clear_data[all groups]': lambda o, d, s: o.clear_data(d, None),
        'remove_communities[all groups]': lambda o, d, s: o.remove_communities(d, None),
        'get_community_clusters[all groups]': lambda o, d, s: o.get_community_clusters(d, None),
    }


def _searches() -> dict[str, Call]:
    """``{name: call}`` for every method of the search interface."""
    g = GROUP
    return {
        'edge_fulltext_search': _all(
            lambda o, d, s: o.edge_fulltext_search(d, 'Alice knows', SearchFilters(), [g], 5),
            lambda o, d, s: o.edge_fulltext_search(d, 'Alice knows', EDGE_FILTER, [g], 5),
        ),
        'edge_similarity_search': _all(
            lambda o, d, s: o.edge_similarity_search(
                d, axis(5), None, None, SearchFilters(), [g], 5, 0.6
            ),
            lambda o, d, s: o.edge_similarity_search(
                d, axis(5), s['alice'].uuid, s['bob'].uuid, EDGE_FILTER, [g], 5, 0.6
            ),
        ),
        'node_fulltext_search': _all(
            lambda o, d, s: o.node_fulltext_search(d, 'Alice', SearchFilters(), [g], 5),
            lambda o, d, s: o.node_fulltext_search(d, 'Alice', ENTITY_FILTER, [g], 5),
        ),
        'node_similarity_search': _all(
            lambda o, d, s: o.node_similarity_search(d, axis(0), SearchFilters(), [g], 5, 0.6),
            lambda o, d, s: o.node_similarity_search(d, axis(0), ENTITY_FILTER, [g], 5, 0.6),
        ),
        'episode_fulltext_search': lambda o, d, s: o.episode_fulltext_search(
            d, 'message', SearchFilters(), [g], 5
        ),
        'edge_bfs_search': lambda o, d, s: o.edge_bfs_search(
            d, [s['alice'].uuid, s['episode_1'].uuid], 3, EDGE_FILTER, [g], 5
        ),
        'node_bfs_search': lambda o, d, s: o.node_bfs_search(
            d, [s['alice'].uuid, s['episode_1'].uuid], ENTITY_FILTER, 3, [g], 5
        ),
        'community_fulltext_search': lambda o, d, s: o.community_fulltext_search(
            d, 'friends', [g], 5
        ),
        'community_similarity_search': lambda o, d, s: o.community_similarity_search(
            d, axis(7), [g], 5, 0.6
        ),
        'get_embeddings_for_communities': lambda o, d, s: o.get_embeddings_for_communities(
            d, [s['community']]
        ),
        'node_distance_reranker': lambda o, d, s: o.node_distance_reranker(
            d, [s['bob'].uuid, s['carol'].uuid], s['alice'].uuid
        ),
        'episode_mentions_reranker': lambda o, d, s: o.episode_mentions_reranker(
            d, [[s['alice'].uuid, s['bob'].uuid]]
        ),
        'build_node_search_filters': lambda o, d, s: _sync(
            o.build_node_search_filters(ENTITY_FILTER)
        ),
        'build_edge_search_filters': lambda o, d, s: _sync(
            o.build_edge_search_filters(EDGE_FILTER)
        ),
    }


# Methods that send no statement: they build filters for a caller.
NO_STATEMENT = frozenset({'build_node_search_filters', 'build_edge_search_filters'})


async def _sync(value: Any) -> Any:
    return value


def _all(*calls: Call) -> Call:
    async def call(o, d, s):
        for each in calls:
            await each(o, d, s)

    return call


def _episode(name: str, valid_at: datetime, uuid: str | None = None) -> EpisodicNode:
    fields: dict[str, Any] = {'uuid': uuid} if uuid else {}
    return EpisodicNode(
        name=name,
        group_id=GROUP,
        created_at=valid_at,
        valid_at=valid_at,
        source=EpisodeType.message,
        source_description='chat',
        content=f'{name} message',
        **fields,
    )


def _community(name: str, uuid: str | None = None) -> CommunityNode:
    fields: dict[str, Any] = {'uuid': uuid} if uuid else {}
    return CommunityNode(
        name=name,
        group_id=GROUP,
        created_at=NOW,
        summary='friends',
        name_embedding=axis(7),
        **fields,
    )


def _saga(name: str, uuid: str | None = None) -> SagaNode:
    fields: dict[str, Any] = {'uuid': uuid} if uuid else {}
    return SagaNode(name=name, group_id=GROUP, created_at=NOW, **fields)


def _relates(source: EntityNode, target: EntityNode, uuid: str | None = None) -> EntityEdge:
    fields: dict[str, Any] = {'uuid': uuid} if uuid else {}
    return EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='KNOWS',
        fact=f'{source.name} knows {target.name}',
        fact_embedding=axis(5),
        group_id=GROUP,
        created_at=NOW,
        valid_at=NOW - timedelta(days=1),
        episodes=[],
        **fields,
    )


def _mentions(episode: EpisodicNode, node: EntityNode, uuid: str | None = None) -> EpisodicEdge:
    fields: dict[str, Any] = {'uuid': uuid} if uuid else {}
    return EpisodicEdge(
        source_node_uuid=episode.uuid,
        target_node_uuid=node.uuid,
        group_id=GROUP,
        created_at=NOW,
        **fields,
    )


def _member(community: CommunityNode, node: EntityNode, uuid: str | None = None) -> CommunityEdge:
    fields: dict[str, Any] = {'uuid': uuid} if uuid else {}
    return CommunityEdge(
        source_node_uuid=community.uuid,
        target_node_uuid=node.uuid,
        group_id=GROUP,
        created_at=NOW,
        **fields,
    )


def _has_episode(saga: SagaNode, episode: EpisodicNode, uuid: str | None = None) -> HasEpisodeEdge:
    fields: dict[str, Any] = {'uuid': uuid} if uuid else {}
    return HasEpisodeEdge(
        source_node_uuid=saga.uuid,
        target_node_uuid=episode.uuid,
        group_id=GROUP,
        created_at=NOW,
        **fields,
    )


def _next(source: EpisodicNode, target: EpisodicNode, uuid: str | None = None) -> NextEpisodeEdge:
    fields: dict[str, Any] = {'uuid': uuid} if uuid else {}
    return NextEpisodeEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        group_id=GROUP,
        created_at=NOW,
        **fields,
    )


async def _seed(driver: AGEDriver) -> dict[str, Any]:
    """A small group graph through the interfaces, and filler on every label."""
    people = {}
    for index, name in enumerate(('Alice', 'Bob', 'Carol')):
        node = EntityNode(
            name=name,
            group_id=GROUP,
            labels=['Person'],
            created_at=NOW,
            summary=f'{name} likes graphs',
            attributes={'age': 30 + index},
        )
        node.name_embedding = axis(index)
        await node.save(driver)
        people[name.lower()] = node
    episode_1 = _episode('one', NOW - timedelta(days=2))
    episode_2 = _episode('two', NOW - timedelta(days=1))
    community = _community('friends')
    saga = _saga('saga')
    for node in (episode_1, episode_2, community, saga):
        await node.save(driver)
    knows = _relates(people['alice'], people['bob'])
    likes = _relates(people['bob'], people['carol'])
    mention = _mentions(episode_1, people['alice'])
    member = _member(community, people['alice'])
    has_episode = _has_episode(saga, episode_1)
    following = _next(episode_1, episode_2)
    for edge in (knows, likes, mention, member, has_episode, following):
        await edge.save(driver)

    # Each filler edge is created with its two ends, in one statement for each
    # edge label: a MATCH of the ends by a computed uuid reads the label once
    # for each row (no index serves an expression of the row).
    vector = f'[x IN range(1, {EMBEDDING_DIM}) | CASE WHEN x = 1 + i % 50 THEN 1.0 ELSE 0.0 END]'
    at = "'2020-01-01T00:00:00+00:00'"
    common = f", name: 'f' + toString(i), created_at: {at}"
    vertex = {
        'Entity': f"{common}, summary: '', labels: ['Filler'], name_embedding: {vector}",
        'Episodic': (
            f"{common}, content: 'filler', source: 'message', source_description: 'filler', "
            f'valid_at: {at}, entity_edges: []'
        ),
        'Community': f"{common}, summary: '', name_embedding: {vector}",
        'Saga': f"{common}, summary: ''",
    }
    edge_extra = {'RELATES_TO': f", name: 'FILLS', fact: 'filler', fact_embedding: {vector}"}
    for label, source, target in (
        ('RELATES_TO', 'Entity', 'Entity'),
        ('MENTIONS', 'Episodic', 'Entity'),
        ('HAS_MEMBER', 'Community', 'Entity'),
        ('HAS_EPISODE', 'Saga', 'Episodic'),
        ('NEXT_EPISODE', 'Episodic', 'Episodic'),
    ):
        end_ = "{uuid: '%s-%s-' + toString(i), group_id: $group%s}"
        edge = f"{{uuid: '{label}-' + toString(i), group_id: $group{edge_extra.get(label, '')}}}"
        await driver._write(
            [
                (
                    f"""
                    UNWIND range(1, $count) AS i
                    CREATE (:{source} {end_ % (label, 's', vertex[source])})
                           -[:{label} {edge}]->
                           (:{target} {end_ % (label, 't', vertex[target])})
                    """,
                    {'count': FILLER, 'group': 'filler'},
                )
            ]
        )
    with connect() as conn:
        for label in (*VERTEX_LABELS, *EDGE_LABELS):
            conn.execute(sql.SQL('ANALYZE {}').format(sql.Identifier(driver.graph, label)))
    return {
        **people,
        'episode_1': episode_1,
        'episode_2': episode_2,
        'community': community,
        'saga': saga,
        'knows': knows,
        'mention': mention,
        'member': member,
        'has_episode': has_episode,
        'next': following,
    }


def _interface_methods(interface: type) -> set[str]:
    return {
        name
        for name, member in vars(interface).items()
        if inspect.isfunction(member) and not name.startswith('_')
    }


def _plan(statement: str, args: Any) -> str:
    with connect() as conn:
        for knob in ('enable_seqscan', 'enable_hashjoin', 'enable_mergejoin'):
            conn.execute(f'SET {knob} = off')
        explain = cast(LiteralString, 'EXPLAIN ' + statement)
        return '\n'.join(row[0] for row in conn.execute(explain, args).fetchall())


_SCAN = re.compile(
    r'(?P<kind>Seq Scan|Index Scan|Index Only Scan|Bitmap Heap Scan)'
    r'(?: Backward)?(?: using \S+)? on (?:\S+\.)?"?(?P<table>[A-Za-z_]+)"?'
)


def _unindexed_label_scans(plan: str) -> list[str]:
    """Scans of a label table that no index condition drives."""
    lines = plan.splitlines()
    problems = []
    for index, line in enumerate(lines):
        match = _SCAN.search(line)
        if not match or match.group('table') not in LABEL_TABLES:
            continue
        details = []
        for follower in lines[index + 1 :]:
            if '->' in follower or _SCAN.search(follower):
                break
            details.append(follower)
        driven = any(
            marker in detail
            for detail in details
            for marker in ('Index Cond:', 'Recheck Cond:', 'Order By:')
        )
        if match.group('kind') == 'Seq Scan' or not driven:
            problems.append(line.strip())
    return problems


def _default_label_rows(graph: str) -> int:
    counts = sql.SQL(' + ').join(
        sql.SQL('(SELECT count(*) FROM ONLY {})').format(sql.Identifier(graph, table))
        for table in DEFAULT_LABEL_TABLES
    )
    with connect() as conn:
        rows = conn.execute(sql.SQL('SELECT {}').format(counts)).fetchall()
    return rows[0][0]


def test_the_workload_calls_every_interface_method():
    """🚨 A new interface method fails here until the workload calls it."""
    calls = {name.split('[')[0] for name in _operations()}
    assert calls == _interface_methods(GraphOperationsInterface)
    assert set(_searches()) == _interface_methods(SearchInterface)
    assert WHOLE_GRAPH <= set(_operations())


async def test_every_statement_reads_its_labels_through_an_index(driver):
    seed = await _seed(driver)
    recorded: list[tuple[str, str, Any]] = []
    current = ['']
    driver._store.statement_hooks.append(
        lambda statement, args, kind: (
            recorded.append((current[0], statement, args)) if kind == 'own' else None
        )
    )

    workload = [
        *((name, call, driver.graph_operations_interface) for name, call in _operations().items()),
        *((name, call, driver.search_interface) for name, call in _searches().items()),
    ]
    # Searches first: the operations end with whole-graph deletes.
    workload.sort(key=lambda entry: entry[2] is driver.graph_operations_interface)
    silent = []
    for name, call, interface in workload:
        current[0] = name
        before = len(recorded)
        await call(interface, driver, seed)
        if len(recorded) == before and name not in NO_STATEMENT:
            silent.append(name)
    assert silent == [], f'sent no statement: {silent}'
    assert _default_label_rows(driver.graph) == 0

    problems = []
    for name, statement, args in recorded:
        if name in WHOLE_GRAPH:
            continue
        plan = _plan(statement, args)
        if 'Disabled: true' in plan or _unindexed_label_scans(plan):
            problems.append(f'[{name}]\n{statement}\n{plan}')
    assert not problems, '\n\n'.join(problems)
