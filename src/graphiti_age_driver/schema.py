"""Labels, indexes and SQL expressions of one graphiti group graph on AGE.

Each graphiti group is one AGE graph (homelab#332, story 32), as each group is
one FalkorDB graph today. A graph has the four vertex labels and the five edge
labels of graphiti-core 0.31.2, and each label is one PostgreSQL table.

🚨 One label for each vertex. AGE gives a vertex exactly one label: ``SET
n:Label`` and a second label on ``CREATE`` do not work (measured on
2026-11-08). graphiti's extra entity types (``Person``, ...) go in the list
property ``labels`` of the ``Entity`` vertex. ``graphiti_entity_types`` is empty
in the homelab today, so the list is empty in practice.

The index expressions are exactly what AGE compiles ``n.<property>`` into:
``agtype_access_operator(VARIADIC ARRAY[properties, '"<property>"'::agtype])``.
Then the planner can match the SQL of the driver and the Cypher of graphiti to
one index.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

AGTYPE = 'ag_catalog.agtype'

ENTITY = 'Entity'
EPISODIC = 'Episodic'
COMMUNITY = 'Community'
SAGA = 'Saga'
VERTEX_LABELS = (ENTITY, EPISODIC, COMMUNITY, SAGA)

RELATES_TO = 'RELATES_TO'
MENTIONS = 'MENTIONS'
HAS_MEMBER = 'HAS_MEMBER'
HAS_EPISODE = 'HAS_EPISODE'
NEXT_EPISODE = 'NEXT_EPISODE'
EDGE_LABELS = (RELATES_TO, MENTIONS, HAS_MEMBER, HAS_EPISODE, NEXT_EPISODE)

# The list property that holds the extra labels of an Entity vertex.
LABELS_PROPERTY = 'labels'

# The embedding properties, and the label that holds each one.
VECTOR_PROPERTIES = (
    (ENTITY, 'name_embedding'),
    (COMMUNITY, 'name_embedding'),
    (RELATES_TO, 'fact_embedding'),
)

# The properties of each full-text index. They are the fields that graphiti's
# FalkorDB indexes hold as FULLTEXT today, without group_id: one graph holds one
# group, and a group filter is a btree condition.
FULLTEXT_PROPERTIES = {
    ENTITY: ('name', 'summary'),
    EPISODIC: ('content', 'source', 'source_description'),
    COMMUNITY: ('name',),
    RELATES_TO: ('name', 'fact'),
}

# 📌 English, as RediSearch on FalkorDB: stemming and an English stop-word list.
# ts_rank replaces the RediSearch score, so the order of full-text hits changes
# (homelab#332 Further Notes).
TEXT_SEARCH_CONFIG = 'english'

# Btree expression indexes, as (label, properties, unique). Every label gets:
# * a UNIQUE index on uuid: graphiti finds every object by it, and a concurrent
#   create of one uuid then fails visibly;
# * an index on (group_id, uuid): graphiti pages a group newest uuid first
#   (`WHERE group_id = $g ORDER BY uuid DESC LIMIT $n`), which is then one
#   backward range scan. Measured on 2026-11-09: with a group_id index only, the
#   planner read the whole label backwards through the uuid index and filtered
#   the group.
_PER_LABEL_BTREE = ((('uuid',), True), (('group_id', 'uuid'), False))
_EXTRA_BTREE = (
    (ENTITY, ('name',), False),  # EntityNode.get_by_names
    (EPISODIC, ('valid_at',), False),  # retrieve_episodes: valid_at <= t, newest first
    (SAGA, ('name',), False),  # the saga by name: retrieve_episodes, graphiti's saga lookup
    (RELATES_TO, ('name',), False),  # the edge-type search filter
)
BTREE_INDEXES: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    tuple(
        (label, props, unique)
        for label in (*VERTEX_LABELS, *EDGE_LABELS)
        for props, unique in _PER_LABEL_BTREE
    )
    + _EXTRA_BTREE
)

# The default embedding dimension: bge-m3 in the homelab (graphiti_embedding_dimensions).
DEFAULT_EMBEDDING_DIM = 1024


def literal(value: str) -> str:
    """Return a SQL string literal of ``value``."""
    return "'" + value.replace("'", "''") + "'"


def prop(alias: str | None, name: str) -> str:
    """The SQL of ``<alias>.<name>`` as AGE compiles a Cypher property access.

    With ``alias=None`` the column has no table prefix, as an index needs it.
    """
    column = f'{alias}.properties' if alias else 'properties'
    return (
        f'ag_catalog.agtype_access_operator(VARIADIC ARRAY[{column}, '
        f'{literal(json.dumps(name))}::{AGTYPE}])'
    )


def prop_text(alias: str | None, name: str) -> str:
    """Return the text of a property: a string without its quotes, or NULL."""
    return f'({prop(alias, name)})::text'


def vector_expr(alias: str | None, name: str, dim: int) -> str:
    """The embedding property as a pgvector value of ``dim`` dimensions."""
    return f'(({prop(alias, name)})::text)::vector({int(dim)})'


def vector_dims_expr(alias: str | None, name: str) -> str:
    """The number of dimensions of the embedding property; NULL if absent."""
    return f'vector_dims((({prop(alias, name)})::text)::vector)'


def tsvector_expr(alias: str | None, label: str) -> str:
    """The full-text document of a label: its full-text properties, joined."""
    parts = " || ' ' || ".join(
        f"coalesce({prop_text(alias, name)}, '')" for name in FULLTEXT_PROPERTIES[label]
    )
    return f'to_tsvector({literal(TEXT_SEARCH_CONFIG)}::regconfig, {parts})'


def table(graph: str, label: str) -> str:
    """The label table. ``graph`` is a validated graph name."""
    return f'"{graph}"."{label}"'


def index_name(label: str, kind: str, prop_name: str = '') -> str:
    return '_'.join(part for part in (label.lower(), prop_name, kind) if part)


def index_statements(graph: str, embedding_dim: int) -> list[str]:
    """``CREATE INDEX IF NOT EXISTS`` for every index that a query relies on.

    🚨 NOTHING ELSE CREATES THESE, AND THE FAILURE IS SILENT: without them each
    lookup reads the whole label. ``tests/test_plans.py`` checks the plan of
    every statement that the driver sends.

    ⚠ An HNSW index needs a fixed dimension. Each one is PARTIAL on
    ``embedding_dim``: an embedding of another dimension is stored but not
    indexed, as FalkorDB stores any length. The driver refuses such a write
    unless it is told otherwise (``AGEDriver(embedding_dims_strict=...)``).
    """
    statements = []
    for label, props, unique in BTREE_INDEXES:
        statements.append(
            f'CREATE {"UNIQUE " if unique else ""}INDEX IF NOT EXISTS '
            f'"{index_name(label, "btree", "_".join(props))}" '
            f'ON {table(graph, label)} ({", ".join(prop(None, name) for name in props)})'
        )
    for label in FULLTEXT_PROPERTIES:
        statements.append(
            f'CREATE INDEX IF NOT EXISTS "{index_name(label, "fts")}" '
            f'ON {table(graph, label)} USING gin ({tsvector_expr(None, label)})'
        )
    for label, prop_name in VECTOR_PROPERTIES:
        statements.append(
            f'CREATE INDEX IF NOT EXISTS "{index_name(label, "hnsw", prop_name)}" '
            f'ON {table(graph, label)} '
            f'USING hnsw (({vector_expr(None, prop_name, embedding_dim)}) vector_cosine_ops) '
            f'WHERE {vector_dims_expr(None, prop_name)} = {int(embedding_dim)}'
        )
    # The extra labels of an entity, for the label filter of a search.
    statements.append(
        f'CREATE INDEX IF NOT EXISTS "{index_name(ENTITY, "gin", LABELS_PROPERTY)}" '
        f'ON {table(graph, ENTITY)} USING gin ({prop(None, LABELS_PROPERTY)})'
    )
    return statements


def index_names() -> Iterable[str]:
    """The names of every index that :func:`index_statements` creates."""
    for label, props, _ in BTREE_INDEXES:
        yield index_name(label, 'btree', '_'.join(props))
    for label in FULLTEXT_PROPERTIES:
        yield index_name(label, 'fts')
    for label, prop_name in VECTOR_PROPERTIES:
        yield index_name(label, 'hnsw', prop_name)
    yield index_name(ENTITY, 'gin', LABELS_PROPERTY)
