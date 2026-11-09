"""graphiti objects from the property maps that the driver reads.

Every read of the driver returns the property map of a vertex or an edge (and,
for an edge, the uuids of its two ends). These functions shape that map into
the record that graphiti's own parsers take, so that graphiti-core builds the
objects as it does for FalkorDB: the same field rules, the same ``attributes``
handling, the same date parsing.
"""

from __future__ import annotations

from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
    get_community_edge_from_record,
    get_entity_edge_from_record,
    get_episodic_edge_from_record,
    get_has_episode_edge_from_record,
    get_next_episode_edge_from_record,
)
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodicNode,
    SagaNode,
    get_community_node_from_record,
    get_entity_node_from_record,
    get_episodic_node_from_record,
    get_saga_node_from_record,
)

from .schema import ENTITY, LABELS_PROPERTY

# graphiti parses records with the FalkorDB rules: the driver reports FALKORDB.
_PROVIDER = GraphProvider.FALKORDB


def entity_labels(props: dict[str, Any]) -> list[str]:
    """``Entity`` and the extra labels, as ``labels(n)`` returns them on FalkorDB."""
    extra = props.get(LABELS_PROPERTY) or []
    return [ENTITY, *(label for label in extra if label != ENTITY)]


def entity_node(props: dict[str, Any]) -> EntityNode:
    record = {
        'uuid': props.get('uuid'),
        'name': props.get('name'),
        'group_id': props.get('group_id'),
        'created_at': props.get('created_at'),
        'summary': props.get('summary'),
        'labels': entity_labels(props),
        # graphiti pops the known fields; what stays is the attributes.
        'attributes': dict(props),
    }
    return get_entity_node_from_record(record, _PROVIDER)


def episodic_node(props: dict[str, Any]) -> EpisodicNode:
    return get_episodic_node_from_record(
        {
            'uuid': props.get('uuid'),
            'name': props.get('name'),
            'group_id': props.get('group_id'),
            'created_at': props.get('created_at'),
            'source': props.get('source'),
            'source_description': props.get('source_description'),
            'content': props.get('content'),
            'valid_at': props.get('valid_at'),
            'entity_edges': props.get('entity_edges') or [],
        }
    )


def community_node(props: dict[str, Any]) -> CommunityNode:
    return get_community_node_from_record(
        {
            'uuid': props.get('uuid'),
            'name': props.get('name'),
            'group_id': props.get('group_id'),
            'created_at': props.get('created_at'),
            'name_embedding': props.get('name_embedding'),
            'summary': props.get('summary'),
        }
    )


def saga_node(props: dict[str, Any]) -> SagaNode:
    return get_saga_node_from_record(dict(props))


def entity_edge(props: dict[str, Any], source: str, target: str) -> EntityEdge:
    record = {
        'uuid': props.get('uuid'),
        'source_node_uuid': source,
        'target_node_uuid': target,
        'group_id': props.get('group_id'),
        'created_at': props.get('created_at'),
        'name': props.get('name'),
        'fact': props.get('fact'),
        'episodes': props.get('episodes') or [],
        'expired_at': props.get('expired_at'),
        'valid_at': props.get('valid_at'),
        'invalid_at': props.get('invalid_at'),
        'reference_time': props.get('reference_time'),
        'attributes': dict(props),
    }
    return get_entity_edge_from_record(record, _PROVIDER)


def _plain_edge_record(props: dict[str, Any], source: str, target: str) -> dict[str, Any]:
    return {
        'uuid': props.get('uuid'),
        'group_id': props.get('group_id'),
        'source_node_uuid': source,
        'target_node_uuid': target,
        'created_at': props.get('created_at'),
    }


def episodic_edge(props: dict[str, Any], source: str, target: str) -> EpisodicEdge:
    return get_episodic_edge_from_record(_plain_edge_record(props, source, target))


def community_edge(props: dict[str, Any], source: str, target: str) -> CommunityEdge:
    return get_community_edge_from_record(_plain_edge_record(props, source, target))


def has_episode_edge(props: dict[str, Any], source: str, target: str) -> HasEpisodeEdge:
    return get_has_episode_edge_from_record(_plain_edge_record(props, source, target))


def next_episode_edge(props: dict[str, Any], source: str, target: str) -> NextEpisodeEdge:
    return get_next_episode_edge_from_record(_plain_edge_record(props, source, target))
