"""graphiti-core's own test suite, run against the AGE driver (homelab#332, seam 2).

``ci/fetch-graphiti-suite.sh`` puts graphiti's ``tests/`` directory of the
pinned release next to this file, unchanged. This conftest takes the place of
graphiti's root conftest: it shadows only the ``graph_driver`` fixture with the
AGE driver, and keeps graphiti's ``mock_embedder``.

* The suite is upstream's database job (``unit_tests.yml``,
  ``database-integration-tests``): the three mocked-LLM test files that take
  ``graph_driver`` (:data:`SUITE_FILES`), with ``-m "not integration"``.
  Upstream runs ``test_add_triplet.py`` with no driver at all, so its driver
  tests never run there, and with graphiti's mock embedder they fail on every
  store. It is not part of the suite.
* ``ci/fetch-graphiti-suite.sh`` lifts the skips for the FalkorDB server
  (``ci/lift-falkordb-skips.py``); the driver only borrows the provider value.
* Four tests are deselected by name (:data:`DESELECTED`), and
  ``tests/test_suite_pins.py`` checks that each one exists, so the list cannot
  go stale:

  * Three test four functions that no code of graphiti-core or of the MCP
    server calls: ``get_relevant_nodes``, ``get_relevant_edges``,
    ``get_edge_invalidation_candidates`` and
    ``filter_existing_duplicate_of_edges``. Each one runs FalkorDB Cypher
    directly, around the two interfaces.
  * ``test_graphiti_retrieve_episodes`` saves its episodes with the base driver,
    into the default graph, and reads them through ``Graphiti.retrieve_episodes``
    with one group id, which reads the group's own graph. It fails on FalkorDB
    for that reason, and the driver keeps FalkorDB's graph for each group on
    purpose (homelab#332, story 32). ``tests/test_group_graphs.py`` runs its
    assertions with the episodes in the group's graph.
* The driver runs in strict mode: a graphiti query on the generic path that is
  not reviewed fails the test that sends it.
"""

from __future__ import annotations

import os
import sys

# graphiti's helpers import a Neo4j and a FalkorDB driver unless these are set.
os.environ['DISABLE_NEO4J'] = '1'
os.environ['DISABLE_FALKORDB'] = '1'
# graphiti's tests import `tests.helpers_test` from the directory of this file.
sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # noqa: E402

from graphiti_age_driver import AGEDriver  # noqa: E402
from tests.helpers_test import group_id, group_id_2, mock_embedder  # noqa: E402, F401

SUITE_FILES = (
    'tests/test_edge_int.py',
    'tests/test_graphiti_mock.py',
    'tests/test_node_int.py',
)

# The tests of the four functions that only tests call.
DESELECTED_DEAD_CODE = (
    'tests/test_graphiti_mock.py::test_filter_existing_duplicate_of_edges',
    'tests/test_graphiti_mock.py::test_get_relevant_nodes',
    'tests/test_graphiti_mock.py::test_get_relevant_edges_and_invalidation_candidates',
)
# A test that assumes one graph for all groups.
DESELECTED_ONE_GRAPH = ('tests/test_graphiti_mock.py::test_graphiti_retrieve_episodes',)
DESELECTED = DESELECTED_DEAD_CODE + DESELECTED_ONE_GRAPH

# The test helpers of graphiti's suite count and print rows with their own Cypher.
HELPER_QUERIES = (
    """
        MATCH (n)
        WHERE n.uuid IN $uuids
        RETURN COUNT(n) as count
        """,
    """
        MATCH (n)-[e]->(m)
        WHERE e.uuid IN $uuids
        RETURN COUNT(e) as count
        UNION ALL
        MATCH (e:RelatesToNode_)
        WHERE e.uuid IN $uuids
        RETURN COUNT(e) as count
        """,
    """
        MATCH (n)
        RETURN n.uuid, n.name
        """,
    """
        MATCH (n)-[e]->(m)
        RETURN n.name, e.uuid, m.name
        """,
)

# graphiti's mock embedder returns 384 dimensions, so the suite indexes 384;
# production indexes 1024.
SUITE_EMBEDDING_DIM = 384


def pytest_collection_modifyitems(config, items):
    """Deselect the four tests. ``tests/test_suite_pins.py`` checks that they
    exist in the fetched suite, so a renamed test cannot slip back in."""
    by_id = {item.nodeid.split('suite/', 1)[-1]: item for item in items}
    deselected = [by_id[test] for test in DESELECTED if test in by_id]
    config.hook.pytest_deselected(items=deselected)
    items[:] = [item for item in items if item not in deselected]


def new_driver() -> AGEDriver:
    return AGEDriver(
        host=os.environ.get('AGE_HOST', '127.0.0.1'),
        port=int(os.environ.get('AGE_PORT', '5432')),
        user=os.environ.get('AGE_USER', 'graphiti'),
        password=os.environ.get('AGE_PASSWORD', 'graphiti'),
        dbname=os.environ.get('AGE_DATABASE', 'graphiti'),
        embedding_dim=SUITE_EMBEDDING_DIM,
        # The suite stores 1024-, 384- and 1-dimension embeddings side by side.
        embedding_dims_strict=False,
        strict=True,
        extra_generic_queries=HELPER_QUERIES,
    )


@pytest.fixture
async def graph_driver():
    """graphiti's fixture, with the AGE driver. As graphiti's fixture, it clears
    the two test groups first. On AGE each group also has a graph of its own
    (``add_episode`` and a search with one group id clone the driver into it),
    so the fixture clears those graphs too."""
    driver = new_driver()
    for group in (group_id, group_id_2):
        await clear_data(driver.clone(group), [group])
    await clear_data(driver, [group_id, group_id_2])
    try:
        yield driver
    finally:
        await driver.close()
