"""graphiti's graph-operations interface on AGE.

Each method keeps the semantics of graphiti-core 0.31.2's FalkorDB fallback for
it (``nodes.py``, ``edges.py``, ``search_utils.py``, ``community_operations.py``,
``graph_data_operations.py``): the same rows, the same order, the same errors.
The queries are new, for the rules of AGE 1.8.0 (homelab#332, Stage A):

* A lookup by a list of values is ``UNWIND $values AS v ... WHERE n.p = v``.
  ``WHERE n.p IN $values`` compiles to ``agtype_in_operator``, which no index
  serves. ``IN`` stays only as a filter after an index-backed step.
* A lookup uses ``WHERE``, never a pattern map: ``{uuid: $x}`` compiles to
  ``properties @> {...}``, which no btree index serves.
* A vertex write uses no ``MERGE``, for the same reason: it creates the absent
  uuids, then updates every vertex through the unique ``uuid`` index, in one
  transaction.
* ``[e:A|B]`` is a syntax error on AGE. A write or a delete over several edge
  labels is one statement for each label.
* An undirected pattern plans as an ``OR`` join over all vertices. Each
  undirected read here is two directed reads.
* A null in ``SET n = map`` is STORED as a null (measured on 2026-11-09). So
  null values never leave Python: an absent key is what FalkorDB stores for a
  null.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EdgeNotFoundError, GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode, SagaNode
from graphiti_core.utils.maintenance.community_operations import Neighbor, label_propagation

from . import records
from .driver import AGEDriver, AGEDriverSession
from .schema import (
    COMMUNITY,
    ENTITY,
    EPISODIC,
    HAS_EPISODE,
    HAS_MEMBER,
    LABELS_PROPERTY,
    MENTIONS,
    NEXT_EPISODE,
    RELATES_TO,
    SAGA,
    VERTEX_LABELS,
)

# Node.delete and its two bulk forms act on these three labels for every node
# class, as on FalkorDB. A saga has its own delete.
_NODE_LABELS = (ENTITY, EPISODIC, COMMUNITY)
# Edge.delete and Edge.delete_by_uuids act on these three labels.
_EDGE_DELETE_LABELS = (MENTIONS, RELATES_TO, HAS_MEMBER)

# The ends of each edge label: (source label, target labels).
_ENDS = {
    RELATES_TO: (ENTITY, (ENTITY,)),
    MENTIONS: (EPISODIC, (ENTITY,)),
    # A community holds entities and, in graphiti's model, other communities.
    HAS_MEMBER: (COMMUNITY, (ENTITY, COMMUNITY)),
    HAS_EPISODE: (SAGA, (EPISODIC,)),
    NEXT_EPISODE: (EPISODIC, (EPISODIC,)),
}

_EDGE_RECORD = {
    RELATES_TO: records.entity_edge,
    MENTIONS: records.episodic_edge,
    HAS_MEMBER: records.community_edge,
    HAS_EPISODE: records.has_episode_edge,
    NEXT_EPISODE: records.next_episode_edge,
}


def _age(driver: Any) -> AGEDriver:
    if not isinstance(driver, AGEDriver):
        raise TypeError(f'the AGE graph operations need an AGEDriver, not {type(driver).__name__}')
    return driver


def _session(transaction: Any) -> AGEDriverSession | None:
    return transaction if isinstance(transaction, AGEDriverSession) else None


def _without_nulls(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _unique(items: Iterable[Any], key) -> list[Any]:
    seen: dict[Any, Any] = {}
    for item in items:
        seen.setdefault(key(item), item)
    return list(seen.values())


# ----------------------------------------------------------------------
# Property maps, as graphiti's FalkorDB queries write them
# ----------------------------------------------------------------------


def _entity_props(data: dict[str, Any]) -> dict[str, Any]:
    """An entity as ``SET n = $entity_data`` stores it, with the extra labels in
    the ``labels`` property instead of as labels."""
    labels = [label for label in data.get('labels') or [] if label != ENTITY]
    props = {key: value for key, value in data.items() if key != 'labels'}
    props[LABELS_PROPERTY] = list(dict.fromkeys(labels))
    return _without_nulls(props)


def _entity_data(node: EntityNode | dict[str, Any]) -> dict[str, Any]:
    if isinstance(node, dict):
        return node
    data: dict[str, Any] = {
        'uuid': node.uuid,
        'name': node.name,
        'name_embedding': node.name_embedding,
        'group_id': node.group_id,
        'summary': node.summary,
        'created_at': node.created_at,
    }
    for key, value in (node.attributes or {}).items():
        data.setdefault(key, value)
    data['labels'] = list(node.labels)
    return data


_EPISODE_FIELDS = (
    'uuid',
    'name',
    'group_id',
    'source_description',
    'source',
    'content',
    'entity_edges',
    'created_at',
    'valid_at',
)


def _episode_props(node: EpisodicNode | dict[str, Any]) -> dict[str, Any]:
    data = node if isinstance(node, dict) else node.model_dump()
    props = {field: data.get(field) for field in _EPISODE_FIELDS}
    source = props['source']
    props['source'] = getattr(source, 'value', source)
    return _without_nulls(props)


def _community_props(node: CommunityNode | dict[str, Any]) -> dict[str, Any]:
    data = node if isinstance(node, dict) else node.model_dump()
    fields = ('uuid', 'name', 'group_id', 'summary', 'created_at', 'name_embedding')
    return _without_nulls({field: data.get(field) for field in fields})


def _saga_props(node: SagaNode | dict[str, Any]) -> dict[str, Any]:
    data = node if isinstance(node, dict) else node.model_dump()
    fields = (
        'uuid',
        'name',
        'group_id',
        'created_at',
        'summary',
        'first_episode_uuid',
        'last_episode_uuid',
        'last_summarized_at',
        'last_summarized_episode_valid_at',
    )
    return _without_nulls({field: data.get(field) for field in fields})


_ENTITY_EDGE_FIELDS = (
    'uuid',
    'name',
    'group_id',
    'fact',
    'fact_embedding',
    'episodes',
    'created_at',
    'expired_at',
    'valid_at',
    'invalid_at',
    'reference_time',
)


def _entity_edge_row(edge: EntityEdge | dict[str, Any]) -> dict[str, Any]:
    data = edge if isinstance(edge, dict) else edge.model_dump()
    props = {field: data.get(field) for field in _ENTITY_EDGE_FIELDS}
    attributes = data.get('attributes') if not isinstance(edge, dict) else None
    extra = (
        attributes
        if attributes is not None
        else {
            key: value
            for key, value in data.items()
            if key not in props
            and key not in ('source_node_uuid', 'target_node_uuid', 'attributes')
        }
    )
    for key, value in (extra or {}).items():
        props.setdefault(key, value)
    return {
        'uuid': data['uuid'],
        'source': data['source_node_uuid'],
        'target': data['target_node_uuid'],
        'props': _without_nulls(props),
    }


def _plain_edge_row(edge: Any) -> dict[str, Any]:
    data = edge if isinstance(edge, dict) else edge.model_dump()
    return {
        'uuid': data['uuid'],
        'source': data['source_node_uuid'],
        'target': data['target_node_uuid'],
        'props': _without_nulls(
            {
                'uuid': data['uuid'],
                'group_id': data.get('group_id'),
                'created_at': data.get('created_at'),
            }
        ),
    }


# ----------------------------------------------------------------------
# Vertex and edge primitives
# ----------------------------------------------------------------------


def _sorted_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by uuid (the last row wins, as two SETs in sequence would)
    and sort. A repeated uuid in one create fails the statement, and two writers
    that take the index locks in different orders can deadlock."""
    by_uuid: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_uuid[row['uuid']] = row
    return [by_uuid[uuid] for uuid in sorted(by_uuid)]


async def upsert_vertices(
    driver: AGEDriver,
    label: str,
    props_list: list[dict[str, Any]],
    tx: AGEDriverSession | None = None,
) -> None:
    """Create the absent uuids, then replace the properties of each vertex.

    ``SET n = props`` replaces the map, as graphiti's ``SET n = $data`` does on
    FalkorDB. For an entity the extra labels accumulate, as FalkorDB labels do:
    ``SET n:Label`` never removes a label.
    """
    for props in props_list:
        driver.check_embedding(props.get('name_embedding'), f'{label} {props["uuid"]}')
    rows = _sorted_rows({'uuid': props['uuid'], 'props': props} for props in props_list)
    if not rows:
        return
    create = f"""
        UNWIND $uuids AS wanted
        OPTIONAL MATCH (n:{label}) WHERE n.uuid = wanted
        WITH wanted, n WHERE n IS NULL
        CREATE (:{label} {{uuid: wanted}})
    """
    if label == ENTITY:
        update = f"""
            UNWIND $rows AS row
            MATCH (n:{ENTITY}) WHERE n.uuid = row.uuid
            WITH n, row.props AS props,
                 [kept IN coalesce(n.{LABELS_PROPERTY}, [])
                  WHERE NOT kept IN row.props.{LABELS_PROPERTY}] AS kept_labels
            SET n = props
            SET n.{LABELS_PROPERTY} = props.{LABELS_PROPERTY} + kept_labels
        """
    else:
        update = f"""
            UNWIND $rows AS row
            MATCH (n:{label}) WHERE n.uuid = row.uuid
            WITH n, row.props AS props
            SET n = props
        """
    await driver._write(
        [(create, {'uuids': [row['uuid'] for row in rows]}), (update, {'rows': rows})], tx=tx
    )


async def upsert_edges(
    driver: AGEDriver,
    label: str,
    rows: list[dict[str, Any]],
    tx: AGEDriverSession | None = None,
) -> None:
    """Create each absent edge between its two ends, then replace its properties.

    As graphiti's ``MERGE (s)-[e:L {uuid}]->(t)`` on FalkorDB: an edge whose end
    is absent is not written. The edge uuid is unique for each label, so a uuid
    that exists between two other ends fails visibly; graphiti gives such an
    edge a new uuid itself (``add_triplet``).
    """
    for row in rows:
        driver.check_embedding(row['props'].get('fact_embedding'), f'{label} {row["uuid"]}')
    rows = _sorted_rows(rows)
    if not rows:
        return
    source_label, target_labels = _ENDS[label]
    statements: list[tuple[str, dict[str, Any] | None]] = []
    for target_label in target_labels:
        statements.append(
            (
                f"""
                UNWIND $rows AS row
                MATCH (s:{source_label}) WHERE s.uuid = row.source
                MATCH (t:{target_label}) WHERE t.uuid = row.target
                OPTIONAL MATCH (s)-[e:{label}]->(t) WHERE e.uuid = row.uuid
                WITH s, t, row, e WHERE e IS NULL
                CREATE (s)-[:{label} {{uuid: row.uuid}}]->(t)
                """,
                {'rows': rows},
            )
        )
        statements.append(
            (
                f"""
                UNWIND $rows AS row
                MATCH (s:{source_label})-[e:{label}]->(t:{target_label})
                WHERE e.uuid = row.uuid AND s.uuid = row.source AND t.uuid = row.target
                WITH e, row.props AS props
                SET e = props
                """,
                {'rows': rows},
            )
        )
    await driver._write(statements, tx=tx)


def _group_match(alias: str, group_ids: list[str]) -> tuple[str, dict[str, Any]]:
    """``MATCH {pattern} WHERE <alias>.group_id ...`` for one or more groups.

    graphiti pages one group at a time. For one group the condition is a plain
    equality, so the (group_id, uuid) index serves the condition and the uuid
    order together; more groups need an ``UNWIND``.
    """
    unique = list(dict.fromkeys(group_ids))
    if len(unique) == 1:
        return f'MATCH {{pattern}} WHERE {alias}.group_id = $group_id', {'group_id': unique[0]}
    return (
        f'UNWIND $group_ids AS wanted MATCH {{pattern}} WHERE {alias}.group_id = wanted',
        {'group_ids': unique},
    )


async def vertices_by_uuids(
    driver: AGEDriver, label: str, uuids: list[str]
) -> list[dict[str, Any]]:
    if not uuids:
        return []
    rows = await driver._read(
        f"""
        UNWIND $uuids AS wanted
        MATCH (n:{label}) WHERE n.uuid = wanted
        RETURN properties(n) AS props
        """,
        {'uuids': list(dict.fromkeys(uuids))},
    )
    return [row['props'] for row in rows]


async def vertices_by_group_ids(
    driver: AGEDriver,
    label: str,
    group_ids: list[str],
    limit: int | None,
    uuid_cursor: str | None,
) -> list[dict[str, Any]]:
    """Return the vertices of the groups, newest uuid first, as graphiti pages them."""
    if not group_ids:
        return []
    cursor = 'AND n.uuid < $uuid' if uuid_cursor else ''
    limit_clause = 'LIMIT $limit' if limit is not None else ''
    match, params = _group_match('n', group_ids)
    rows = await driver._read(
        f"""
        {match.format(pattern=f'(n:{label})')} {cursor}
        RETURN properties(n) AS props
        ORDER BY n.uuid DESC
        {limit_clause}
        """,
        {**params, 'uuid': uuid_cursor, 'limit': limit},
    )
    return [row['props'] for row in rows]


async def delete_vertices(
    driver: AGEDriver,
    labels: Iterable[str],
    uuids: list[str],
    group_id: str | None = None,
) -> None:
    if not uuids:
        return
    group_filter = 'AND n.group_id = $group_id' if group_id is not None else ''
    await driver._write(
        [
            (
                f"""
                UNWIND $uuids AS wanted
                MATCH (n:{label}) WHERE n.uuid = wanted {group_filter}
                DETACH DELETE n
                """,
                {'uuids': sorted(set(uuids)), 'group_id': group_id},
            )
            for label in labels
        ]
    )


async def delete_group_vertices(driver: AGEDriver, labels: Iterable[str], group_id: str) -> None:
    await driver._write(
        [
            (
                f"""
                MATCH (n:{label}) WHERE n.group_id = $group_id
                DETACH DELETE n
                """,
                {'group_id': group_id},
            )
            for label in labels
        ]
    )


def _edge_match(label: str, target_label: str | None = None) -> str:
    source_label, target_labels = _ENDS[label]
    target = target_label or (target_labels[0] if len(target_labels) == 1 else '')
    return f'(n:{source_label})-[e:{label}]->(m{":" + target if target else ""})'


async def edges_by_uuids(driver: AGEDriver, label: str, uuids: list[str]) -> list[Any]:
    if not uuids:
        return []
    rows = await driver._read(
        f"""
        UNWIND $uuids AS wanted
        MATCH {_edge_match(label)} WHERE e.uuid = wanted
        RETURN properties(e) AS props, n.uuid AS source, m.uuid AS target
        """,
        {'uuids': list(dict.fromkeys(uuids))},
    )
    return [_EDGE_RECORD[label](row['props'], row['source'], row['target']) for row in rows]


async def edges_by_group_ids(
    driver: AGEDriver,
    label: str,
    group_ids: list[str],
    limit: int | None,
    uuid_cursor: str | None,
) -> list[Any]:
    if not group_ids:
        return []
    cursor = 'AND e.uuid < $uuid' if uuid_cursor else ''
    limit_clause = 'LIMIT $limit' if limit is not None else ''
    match, params = _group_match('e', group_ids)
    rows = await driver._read(
        f"""
        {match.format(pattern=_edge_match(label))} {cursor}
        RETURN properties(e) AS props, n.uuid AS source, m.uuid AS target
        ORDER BY e.uuid DESC
        {limit_clause}
        """,
        {**params, 'uuid': uuid_cursor, 'limit': limit},
    )
    return [_EDGE_RECORD[label](row['props'], row['source'], row['target']) for row in rows]


async def delete_edges(
    driver: AGEDriver, labels: Iterable[str], uuids: list[str], group_id: str | None = None
) -> None:
    if not uuids:
        return
    group_filter = 'AND e.group_id = $group_id' if group_id is not None else ''
    await driver._write(
        [
            (
                f"""
                UNWIND $uuids AS wanted
                MATCH ()-[e:{label}]->() WHERE e.uuid = wanted {group_filter}
                DELETE e
                """,
                {'uuids': sorted(set(uuids)), 'group_id': group_id},
            )
            for label in labels
        ]
    )


async def _one_edge(driver: AGEDriver, label: str, uuid: str) -> Any:
    edges = await edges_by_uuids(driver, label, [uuid])
    if not edges:
        raise EdgeNotFoundError(uuid)
    return edges[0]


async def _load_embeddings(
    driver: AGEDriver, label: str, prop_name: str, uuids: list[str]
) -> dict[str, list[float]]:
    if not uuids:
        return {}
    if label == RELATES_TO:
        match = f'MATCH ()-[x:{RELATES_TO}]->() WHERE x.uuid = wanted'
    else:
        match = f'MATCH (x:{label}) WHERE x.uuid = wanted'
    rows = await driver._read(
        f"""
        UNWIND $uuids AS wanted
        {match}
        RETURN x.uuid AS uuid, x.{prop_name} AS embedding
        """,
        {'uuids': list(dict.fromkeys(uuids))},
    )
    return {row['uuid']: row['embedding'] for row in rows if row['embedding'] is not None}


class AGEGraphOperations(GraphOperationsInterface):
    """Every method of graphiti-core 0.31.2's graph-operations interface."""

    # --------------------------
    # Entity nodes
    # --------------------------

    async def node_save(self, node: Any, driver: Any) -> None:
        await upsert_vertices(_age(driver), ENTITY, [_entity_props(_entity_data(node))])

    async def node_delete(self, node: Any, driver: Any) -> None:
        await delete_vertices(_age(driver), _NODE_LABELS, [node.uuid])

    async def node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        props = [_entity_props(_entity_data(node)) for node in nodes]
        await upsert_vertices(_age(driver), ENTITY, props, tx=_session(transaction))

    async def node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await delete_group_vertices(_age(driver), _NODE_LABELS, group_id)

    async def node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await delete_vertices(_age(driver), _NODE_LABELS, uuids, group_id)

    async def node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        props = await vertices_by_uuids(_age(driver), ENTITY, [uuid])
        if not props:
            raise NodeNotFoundError(uuid)
        return records.entity_node(props[0])

    async def node_get_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> list[Any]:
        nodes = [
            records.entity_node(p) for p in await vertices_by_uuids(_age(driver), ENTITY, uuids)
        ]
        if group_id is not None:
            nodes = [node for node in nodes if node.group_id == group_id]
        return nodes

    async def node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        props = await vertices_by_group_ids(_age(driver), ENTITY, group_ids, limit, uuid_cursor)
        return [records.entity_node(p) for p in props]

    async def node_get_by_names(
        self,
        _cls: Any,
        driver: Any,
        names: list[str],
        group_ids: list[str],
    ) -> list[Any]:
        unique_names = sorted({name for name in names if name})
        unique_group_ids = sorted({group_id for group_id in group_ids if group_id})
        if not unique_names or not unique_group_ids:
            return []
        rows = await _age(driver)._read(
            f"""
            UNWIND $names AS wanted
            MATCH (n:{ENTITY}) WHERE n.name = wanted AND n.group_id IN $group_ids
            RETURN properties(n) AS props
            """,
            {'names': unique_names, 'group_ids': unique_group_ids},
        )
        return [records.entity_node(row['props']) for row in rows]

    async def node_get_by_attribute_values(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        attributes: dict[str, Any],
        node_labels: list[str] | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """The entities of a group whose attributes equal all of ``attributes``.

        graphiti calls this only for entity types with identity properties.
        ``graphiti_entity_types`` is empty in the homelab, so it does not run
        there today. The group lookup is the index-backed step; the attribute
        and label tests filter after it.
        """
        if not attributes:
            return []
        rows = await _age(driver)._read(
            f"""
            MATCH (n:{ENTITY}) WHERE n.group_id = $group_id
            WITH n, properties(n) AS props
            WHERE all(key IN keys($attributes) WHERE props[key] = $attributes[key])
            RETURN props
            LIMIT $limit
            """,
            {'group_id': group_id, 'attributes': attributes, 'limit': limit},
        )
        nodes = [records.entity_node(row['props']) for row in rows]
        if node_labels:
            wanted = set(node_labels)
            nodes = [node for node in nodes if wanted & set(node.labels)]
        return nodes

    async def node_load_embeddings(self, node: Any, driver: Any) -> None:
        embeddings = await _load_embeddings(_age(driver), ENTITY, 'name_embedding', [node.uuid])
        if node.uuid not in embeddings:
            raise NodeNotFoundError(node.uuid)
        node.name_embedding = embeddings[node.uuid]

    async def node_load_embeddings_bulk(
        self,
        driver: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        uuids = [node.uuid for node in nodes]
        return await _load_embeddings(_age(driver), ENTITY, 'name_embedding', uuids)

    # --------------------------
    # Episodic nodes
    # --------------------------

    async def episodic_node_save(self, node: Any, driver: Any) -> None:
        await upsert_vertices(_age(driver), EPISODIC, [_episode_props(node)])

    async def episodic_node_delete(self, node: Any, driver: Any) -> None:
        await delete_vertices(_age(driver), (EPISODIC,), [node.uuid])

    async def episodic_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        props = [_episode_props(node) for node in nodes]
        await upsert_vertices(_age(driver), EPISODIC, props, tx=_session(transaction))

    async def episodic_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        episodic_edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        rows = [_plain_edge_row(edge) for edge in episodic_edges]
        await upsert_edges(_age(driver), MENTIONS, rows, tx=_session(transaction))

    async def episodic_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await delete_group_vertices(_age(driver), (EPISODIC,), group_id)

    async def episodic_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await delete_vertices(_age(driver), (EPISODIC,), uuids, group_id)

    async def episodic_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        props = await vertices_by_uuids(_age(driver), EPISODIC, [uuid])
        if not props:
            raise NodeNotFoundError(uuid)
        return records.episodic_node(props[0])

    async def episodic_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return [
            records.episodic_node(p) for p in await vertices_by_uuids(_age(driver), EPISODIC, uuids)
        ]

    async def episodic_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        props = await vertices_by_group_ids(_age(driver), EPISODIC, group_ids, limit, uuid_cursor)
        return [records.episodic_node(p) for p in props]

    async def retrieve_episodes(
        self,
        driver: Any,
        reference_time: datetime,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: Any | None = None,
        saga: str | None = None,
    ) -> list[Any]:
        """Return the ``last_n`` newest episodes at ``reference_time``, oldest first."""
        age = _age(driver)
        source_filter = 'AND e.source = $source' if source is not None else ''
        params: dict[str, Any] = {
            'reference_time': reference_time,
            'source': source.name if source is not None else None,
            'num_episodes': last_n,
        }
        if saga is not None:
            group_id = group_ids[0] if group_ids else None
            if group_id is None:
                # FalkorDB matches {group_id: null}, which matches no saga.
                return []
            query = f"""
                MATCH (s:{SAGA}) WHERE s.name = $saga_name AND s.group_id = $group_id
                MATCH (s)-[:{HAS_EPISODE}]->(e:{EPISODIC})
                WHERE e.valid_at <= $reference_time {source_filter}
                RETURN properties(e) AS props
                ORDER BY e.valid_at DESC, e.created_at DESC
                LIMIT $num_episodes
            """
            params.update(saga_name=saga, group_id=group_id)
        else:
            group_filter = 'AND e.group_id IN $group_ids' if group_ids else ''
            query = f"""
                MATCH (e:{EPISODIC})
                WHERE e.valid_at <= $reference_time {group_filter} {source_filter}
                RETURN properties(e) AS props
                ORDER BY e.valid_at DESC, e.created_at DESC
                LIMIT $num_episodes
            """
            params['group_ids'] = group_ids
        rows = await age._read(query, params)
        return list(reversed([records.episodic_node(row['props']) for row in rows]))

    # --------------------------
    # Community nodes
    # --------------------------

    async def community_node_save(self, node: Any, driver: Any) -> None:
        await upsert_vertices(_age(driver), COMMUNITY, [_community_props(node)])

    async def community_node_delete(self, node: Any, driver: Any) -> None:
        await delete_vertices(_age(driver), (COMMUNITY,), [node.uuid])

    async def community_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        props = [_community_props(node) for node in nodes]
        await upsert_vertices(_age(driver), COMMUNITY, props, tx=_session(transaction))

    async def community_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await delete_group_vertices(_age(driver), (COMMUNITY,), group_id)

    async def community_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await delete_vertices(_age(driver), (COMMUNITY,), uuids, group_id)

    async def community_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        props = await vertices_by_uuids(_age(driver), COMMUNITY, [uuid])
        if not props:
            raise NodeNotFoundError(uuid)
        return records.community_node(props[0])

    async def community_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return [
            records.community_node(p)
            for p in await vertices_by_uuids(_age(driver), COMMUNITY, uuids)
        ]

    async def community_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        props = await vertices_by_group_ids(_age(driver), COMMUNITY, group_ids, limit, uuid_cursor)
        return [records.community_node(p) for p in props]

    # --------------------------
    # Saga nodes
    # --------------------------

    async def saga_node_save(self, node: Any, driver: Any) -> None:
        await upsert_vertices(_age(driver), SAGA, [_saga_props(node)])

    async def saga_node_delete(self, node: Any, driver: Any) -> None:
        await delete_vertices(_age(driver), (SAGA,), [node.uuid])

    async def saga_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        props = [_saga_props(node) for node in nodes]
        await upsert_vertices(_age(driver), SAGA, props, tx=_session(transaction))

    async def saga_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await delete_group_vertices(_age(driver), (SAGA,), group_id)

    async def saga_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        await delete_vertices(_age(driver), (SAGA,), uuids, group_id)

    async def saga_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        props = await vertices_by_uuids(_age(driver), SAGA, [uuid])
        if not props:
            raise NodeNotFoundError(uuid)
        return records.saga_node(props[0])

    async def saga_node_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[Any]:
        return [records.saga_node(p) for p in await vertices_by_uuids(_age(driver), SAGA, uuids)]

    async def saga_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        props = await vertices_by_group_ids(_age(driver), SAGA, group_ids, limit, uuid_cursor)
        return [records.saga_node(p) for p in props]

    async def saga_get_previous_episode_uuid(
        self,
        driver: Any,
        saga_uuid: str,
        current_episode_uuid: str,
    ) -> str | None:
        rows = await _age(driver)._read(
            f"""
            MATCH (s:{SAGA}) WHERE s.uuid = $saga_uuid
            MATCH (s)-[:{HAS_EPISODE}]->(e:{EPISODIC})
            WHERE e.uuid <> $current_episode_uuid
            RETURN e.uuid AS uuid
            ORDER BY e.valid_at DESC, e.created_at DESC
            LIMIT 1
            """,
            {'saga_uuid': saga_uuid, 'current_episode_uuid': current_episode_uuid},
        )
        return rows[0]['uuid'] if rows else None

    async def saga_get_episode_contents(
        self,
        driver: Any,
        saga_uuid: str,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[tuple[str, Any, Any]]:
        """(content, valid_at, created_at) of the saga's episodes after ``since``,
        oldest first, as graphiti's own fallback reads them."""
        since_filter = 'AND e.created_at > $since' if since is not None else ''
        rows = await _age(driver)._read(
            f"""
            MATCH (s:{SAGA}) WHERE s.uuid = $saga_uuid
            MATCH (s)-[:{HAS_EPISODE}]->(e:{EPISODIC})
            WHERE e.created_at IS NOT NULL {since_filter}
            RETURN e.content AS content, e.valid_at AS valid_at, e.created_at AS created_at
            ORDER BY e.created_at ASC, e.valid_at ASC
            LIMIT $limit
            """,
            {'saga_uuid': saga_uuid, 'since': since, 'limit': limit},
        )
        contents = []
        for row in rows:
            created_at = parse_db_date(row.get('created_at'))
            if row.get('content') and created_at is not None:
                contents.append((row['content'], parse_db_date(row.get('valid_at')), created_at))
        return contents

    # --------------------------
    # Entity edges
    # --------------------------

    async def edge_save(self, edge: Any, driver: Any) -> None:
        await upsert_edges(_age(driver), RELATES_TO, [_entity_edge_row(edge)])

    async def edge_delete(self, edge: Any, driver: Any) -> None:
        await delete_edges(_age(driver), _EDGE_DELETE_LABELS, [edge.uuid])

    async def edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        rows = [_entity_edge_row(edge) for edge in edges]
        await upsert_edges(_age(driver), RELATES_TO, rows, tx=_session(transaction))

    async def edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await delete_edges(_age(driver), _EDGE_DELETE_LABELS, uuids, group_id)

    async def edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await _one_edge(_age(driver), RELATES_TO, uuid)

    async def edge_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[Any]:
        return await edges_by_uuids(_age(driver), RELATES_TO, uuids)

    async def edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        edges = await edges_by_group_ids(_age(driver), RELATES_TO, group_ids, limit, uuid_cursor)
        if not edges:
            raise GroupsEdgesNotFoundError(group_ids)
        return edges

    async def edge_load_embeddings(self, edge: Any, driver: Any) -> None:
        embeddings = await _load_embeddings(_age(driver), RELATES_TO, 'fact_embedding', [edge.uuid])
        if edge.uuid not in embeddings:
            raise EdgeNotFoundError(edge.uuid)
        edge.fact_embedding = embeddings[edge.uuid]

    async def edge_load_embeddings_bulk(
        self,
        driver: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        uuids = [edge.uuid for edge in edges]
        return await _load_embeddings(_age(driver), RELATES_TO, 'fact_embedding', uuids)

    # --------------------------
    # Episodic edges (MENTIONS)
    # --------------------------

    async def episodic_edge_save(self, edge: Any, driver: Any) -> None:
        await upsert_edges(_age(driver), MENTIONS, [_plain_edge_row(edge)])

    async def episodic_edge_delete(self, edge: Any, driver: Any) -> None:
        await delete_edges(_age(driver), (MENTIONS,), [edge.uuid])

    async def episodic_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await delete_edges(_age(driver), (MENTIONS,), uuids, group_id)

    async def episodic_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await _one_edge(_age(driver), MENTIONS, uuid)

    async def episodic_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        edges = await edges_by_uuids(_age(driver), MENTIONS, uuids)
        if not edges:
            raise EdgeNotFoundError(uuids[0] if uuids else '')
        return edges

    async def episodic_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        edges = await edges_by_group_ids(_age(driver), MENTIONS, group_ids, limit, uuid_cursor)
        if not edges:
            raise GroupsEdgesNotFoundError(group_ids)
        return edges

    # --------------------------
    # Community edges (HAS_MEMBER)
    # --------------------------

    async def community_edge_save(self, edge: Any, driver: Any) -> None:
        await upsert_edges(_age(driver), HAS_MEMBER, [_plain_edge_row(edge)])

    async def community_edge_delete(self, edge: Any, driver: Any) -> None:
        await delete_edges(_age(driver), (HAS_MEMBER,), [edge.uuid])

    async def community_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await delete_edges(_age(driver), (HAS_MEMBER,), uuids, group_id)

    async def community_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await _one_edge(_age(driver), HAS_MEMBER, uuid)

    async def community_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await edges_by_uuids(_age(driver), HAS_MEMBER, uuids)

    async def community_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await edges_by_group_ids(_age(driver), HAS_MEMBER, group_ids, limit, uuid_cursor)

    # --------------------------
    # Saga edges (HAS_EPISODE)
    # --------------------------

    async def has_episode_edge_save(self, edge: Any, driver: Any) -> None:
        await upsert_edges(_age(driver), HAS_EPISODE, [_plain_edge_row(edge)])

    async def has_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        await delete_edges(_age(driver), (HAS_EPISODE,), [edge.uuid])

    async def has_episode_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        rows = [_plain_edge_row(edge) for edge in edges]
        await upsert_edges(_age(driver), HAS_EPISODE, rows, tx=_session(transaction))

    async def has_episode_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await delete_edges(_age(driver), (HAS_EPISODE,), uuids, group_id)

    async def has_episode_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await _one_edge(_age(driver), HAS_EPISODE, uuid)

    async def has_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await edges_by_uuids(_age(driver), HAS_EPISODE, uuids)

    async def has_episode_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await edges_by_group_ids(_age(driver), HAS_EPISODE, group_ids, limit, uuid_cursor)

    # --------------------------
    # Episode chain edges (NEXT_EPISODE)
    # --------------------------

    async def next_episode_edge_save(self, edge: Any, driver: Any) -> None:
        await upsert_edges(_age(driver), NEXT_EPISODE, [_plain_edge_row(edge)])

    async def next_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        await delete_edges(_age(driver), (NEXT_EPISODE,), [edge.uuid])

    async def next_episode_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        rows = [_plain_edge_row(edge) for edge in edges]
        await upsert_edges(_age(driver), NEXT_EPISODE, rows, tx=_session(transaction))

    async def next_episode_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        await delete_edges(_age(driver), (NEXT_EPISODE,), uuids, group_id)

    async def next_episode_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> Any:
        return await _one_edge(_age(driver), NEXT_EPISODE, uuid)

    async def next_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[Any]:
        return await edges_by_uuids(_age(driver), NEXT_EPISODE, uuids)

    async def next_episode_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[Any]:
        return await edges_by_group_ids(_age(driver), NEXT_EPISODE, group_ids, limit, uuid_cursor)

    # --------------------------
    # Search and graph maintenance helpers
    # --------------------------

    async def get_mentioned_nodes(self, driver: Any, episodes: list[Any]) -> list[Any]:
        if not episodes:
            return []
        rows = await _age(driver)._read(
            f"""
            UNWIND $uuids AS wanted
            MATCH (episode:{EPISODIC}) WHERE episode.uuid = wanted
            MATCH (episode)-[:{MENTIONS}]->(n:{ENTITY})
            RETURN properties(n) AS props
            """,
            {'uuids': list(dict.fromkeys(episode.uuid for episode in episodes))},
        )
        unique = _unique((row['props'] for row in rows), key=lambda props: props['uuid'])
        return [records.entity_node(props) for props in unique]

    async def get_communities_by_nodes(self, driver: Any, nodes: list[Any]) -> list[Any]:
        if not nodes:
            return []
        rows = await _age(driver)._read(
            f"""
            UNWIND $uuids AS wanted
            MATCH (m:{ENTITY}) WHERE m.uuid = wanted
            MATCH (c:{COMMUNITY})-[:{HAS_MEMBER}]->(m)
            RETURN properties(c) AS props
            """,
            {'uuids': list(dict.fromkeys(node.uuid for node in nodes))},
        )
        unique = _unique((row['props'] for row in rows), key=lambda props: props['uuid'])
        return [records.community_node(props) for props in unique]

    async def clear_data(self, driver: Any, group_ids: list[str] | None = None) -> None:
        """Delete the data of the groups (or all data) in THIS driver's graph.

        As on FalkorDB, the scope is the graph of the driver: a group's data
        lives in the group's graph, so the MCP ``clear_graph`` tool passes the
        group's clone.
        """
        age = _age(driver)
        if group_ids is None:
            await age._write(
                [(f'MATCH (n:{label}) DETACH DELETE n', None) for label in VERTEX_LABELS]
            )
            return
        await age._write(
            [
                (
                    f"""
                    UNWIND $group_ids AS wanted
                    MATCH (n:{label}) WHERE n.group_id = wanted
                    DETACH DELETE n
                    """,
                    {'group_ids': sorted(set(group_ids))},
                )
                for label in VERTEX_LABELS
            ]
        )

    async def get_community_clusters(
        self, driver: Any, group_ids: list[str] | None
    ) -> list[list[Any]]:
        """Label propagation over the RELATES_TO projection of each group.

        One read for each group gives the edge count between each pair of
        entities in either direction, which graphiti's fallback reads with one
        query for each entity.
        """
        age = _age(driver)
        if group_ids is None:
            rows = await age._read(
                f'MATCH (n:{ENTITY}) WHERE n.group_id IS NOT NULL '
                'RETURN collect(DISTINCT n.group_id) AS group_ids'
            )
            group_ids = rows[0]['group_ids'] if rows else []
        clusters: list[list[Any]] = []
        for group_id in group_ids or []:
            nodes = await self.node_get_by_group_ids(None, age, [group_id])
            pairs = await age._read(
                f"""
                MATCH (n:{ENTITY}) WHERE n.group_id = $group_id
                MATCH (n)-[e:{RELATES_TO}]->(m:{ENTITY})
                WHERE m.group_id = $group_id
                RETURN n.uuid AS source, m.uuid AS target, count(e) AS count
                """,
                {'group_id': group_id},
            )
            counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for pair in pairs:
                counts[pair['source']][pair['target']] += pair['count']
                if pair['source'] != pair['target']:
                    counts[pair['target']][pair['source']] += pair['count']
            projection = {
                node.uuid: [
                    Neighbor(node_uuid=neighbor, edge_count=count)
                    for neighbor, count in counts[node.uuid].items()
                ]
                for node in nodes
            }
            for cluster in label_propagation(projection):
                clusters.append(await self.node_get_by_uuids(None, age, cluster))
        return clusters

    async def remove_communities(self, driver: Any, group_ids: list[str] | None = None) -> None:
        age = _age(driver)
        if group_ids:
            await age._write(
                [
                    (
                        f"""
                        UNWIND $group_ids AS wanted
                        MATCH (c:{COMMUNITY}) WHERE c.group_id = wanted
                        DETACH DELETE c
                        """,
                        {'group_ids': sorted(set(group_ids))},
                    )
                ]
            )
        else:
            await age._write([(f'MATCH (c:{COMMUNITY}) DETACH DELETE c', None)])

    async def determine_entity_community(self, driver: Any, entity: Any) -> tuple[Any | None, bool]:
        """The community of the entity, or the community that most of its
        RELATES_TO neighbours belong to (then ``True``: the entity must join)."""
        age = _age(driver)
        rows = await age._read(
            f"""
            MATCH (n:{ENTITY}) WHERE n.uuid = $entity_uuid
            MATCH (c:{COMMUNITY})-[:{HAS_MEMBER}]->(n)
            RETURN properties(c) AS props
            """,
            {'entity_uuid': entity.uuid},
        )
        if rows:
            return records.community_node(rows[0]['props']), False

        communities = []
        # AGE refuses a second label on a bound variable, so `m` gets its label
        # where it is first bound.
        for direction in ('(n)-[:{r}]->(m:{l})', '(m:{l})-[:{r}]->(n)'):
            pattern = direction.format(r=RELATES_TO, l=ENTITY)
            rows = await age._read(
                f"""
                MATCH (n:{ENTITY}) WHERE n.uuid = $entity_uuid
                MATCH {pattern}
                MATCH (c:{COMMUNITY})-[:{HAS_MEMBER}]->(m)
                RETURN properties(c) AS props
                """,
                {'entity_uuid': entity.uuid},
            )
            communities.extend(records.community_node(row['props']) for row in rows)

        counts: dict[str, int] = defaultdict(int)
        for community in communities:
            counts[community.uuid] += 1
        community_uuid, best = None, 0
        for uuid, count in counts.items():
            if count > best:
                community_uuid, best = uuid, count
        if best == 0:
            return None, False
        for community in communities:
            if community.uuid == community_uuid:
                return community, True
        return None, False

    async def episodic_node_get_by_entity_node_uuid(
        self,
        _cls: Any,
        driver: Any,
        entity_node_uuid: str,
    ) -> list[Any]:
        rows = await _age(driver)._read(
            f"""
            MATCH (n:{ENTITY}) WHERE n.uuid = $entity_node_uuid
            MATCH (e:{EPISODIC})-[:{MENTIONS}]->(n)
            RETURN properties(e) AS props
            """,
            {'entity_node_uuid': entity_node_uuid},
        )
        unique = _unique((row['props'] for row in rows), key=lambda props: props['uuid'])
        return [records.episodic_node(props) for props in unique]

    async def community_node_load_name_embedding(self, node: Any, driver: Any) -> None:
        embeddings = await _load_embeddings(_age(driver), COMMUNITY, 'name_embedding', [node.uuid])
        if node.uuid not in embeddings:
            raise NodeNotFoundError(node.uuid)
        node.name_embedding = embeddings[node.uuid]

    async def edge_get_between_nodes(
        self,
        _cls: Any,
        driver: Any,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[Any]:
        rows = await _age(driver)._read(
            f"""
            MATCH (n:{ENTITY}) WHERE n.uuid = $source_node_uuid
            MATCH (n)-[e:{RELATES_TO}]->(m:{ENTITY})
            WHERE m.uuid = $target_node_uuid
            RETURN properties(e) AS props, n.uuid AS source, m.uuid AS target
            """,
            {'source_node_uuid': source_node_uuid, 'target_node_uuid': target_node_uuid},
        )
        return [records.entity_edge(row['props'], row['source'], row['target']) for row in rows]

    async def edge_get_by_node_uuid(self, _cls: Any, driver: Any, node_uuid: str) -> list[Any]:
        """The RELATES_TO edges at the node, in either direction: two directed reads."""
        age = _age(driver)
        rows = []
        for pattern in ('(x)-[e:{r}]->(y:{l})', '(y:{l})-[e:{r}]->(x)'):
            rows += await age._read(
                f"""
                MATCH (x:{ENTITY}) WHERE x.uuid = $node_uuid
                MATCH {pattern.format(r=RELATES_TO, l=ENTITY)}
                RETURN properties(e) AS props, startNode(e).uuid AS source,
                       endNode(e).uuid AS target
                """,
                {'node_uuid': node_uuid},
            )
        unique = _unique(rows, key=lambda row: row['props']['uuid'])
        return [records.entity_edge(row['props'], row['source'], row['target']) for row in unique]
