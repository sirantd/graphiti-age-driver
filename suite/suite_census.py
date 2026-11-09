"""The census of graphiti-core 0.31.2's suite that the AGE gate runs.

``suite/conftest.py`` runs and deselects by these lists, and
``tests/test_suite_pins.py`` checks them against the fetched suite. A graphiti
bump that adds, removes or renames a test changes the census and fails the
server-free job first.
"""

from __future__ import annotations

# Upstream's database job (unit_tests.yml, database-integration-tests).
SUITE_FILES = (
    'tests/test_edge_int.py',
    'tests/test_graphiti_mock.py',
    'tests/test_node_int.py',
)

# Every test function of SUITE_FILES. Each one takes graph_driver.
SUITE_TESTS = (
    'tests/test_edge_int.py::test_episodic_edge',
    'tests/test_edge_int.py::test_entity_edge',
    'tests/test_edge_int.py::test_community_edge',
    'tests/test_graphiti_mock.py::test_add_bulk',
    'tests/test_graphiti_mock.py::test_remove_episode',
    'tests/test_graphiti_mock.py::test_remove_episode_keeps_restated_fact',
    'tests/test_graphiti_mock.py::test_remove_episode_rebuilds_summary_from_remaining_episodes',
    'tests/test_graphiti_mock.py::test_remove_episode_rebuilds_summary_in_batches',
    'tests/test_graphiti_mock.py::test_remove_episode_keeps_old_summary_when_rebuild_fails',
    'tests/test_graphiti_mock.py::test_graphiti_retrieve_episodes',
    'tests/test_graphiti_mock.py::test_filter_existing_duplicate_of_edges',
    'tests/test_graphiti_mock.py::test_determine_entity_community',
    'tests/test_graphiti_mock.py::test_get_community_clusters',
    'tests/test_graphiti_mock.py::test_get_mentioned_nodes',
    'tests/test_graphiti_mock.py::test_get_communities_by_nodes',
    'tests/test_graphiti_mock.py::test_edge_fulltext_search',
    'tests/test_graphiti_mock.py::test_edge_similarity_search',
    'tests/test_graphiti_mock.py::test_edge_bfs_search',
    'tests/test_graphiti_mock.py::test_node_fulltext_search',
    'tests/test_graphiti_mock.py::test_node_similarity_search',
    'tests/test_graphiti_mock.py::test_node_bfs_search',
    'tests/test_graphiti_mock.py::test_episode_fulltext_search',
    'tests/test_graphiti_mock.py::test_community_fulltext_search',
    'tests/test_graphiti_mock.py::test_community_similarity_search',
    'tests/test_graphiti_mock.py::test_get_relevant_nodes',
    'tests/test_graphiti_mock.py::test_get_relevant_edges_and_invalidation_candidates',
    'tests/test_graphiti_mock.py::test_node_distance_reranker',
    'tests/test_graphiti_mock.py::test_episode_mentions_reranker',
    'tests/test_graphiti_mock.py::test_get_embeddings_for_edges',
    'tests/test_graphiti_mock.py::test_get_embeddings_for_nodes',
    'tests/test_graphiti_mock.py::test_get_embeddings_for_communities',
    'tests/test_node_int.py::test_entity_node',
    'tests/test_node_int.py::test_community_node',
    'tests/test_node_int.py::test_episodic_node',
)

# The tests of the four functions that only tests call (no code of graphiti-core
# or of the MCP server calls them): get_relevant_nodes, get_relevant_edges,
# get_edge_invalidation_candidates, filter_existing_duplicate_of_edges. Each one
# runs FalkorDB Cypher directly, around the two interfaces.
DESELECTED_DEAD_CODE = (
    'tests/test_graphiti_mock.py::test_filter_existing_duplicate_of_edges',
    'tests/test_graphiti_mock.py::test_get_relevant_nodes',
    'tests/test_graphiti_mock.py::test_get_relevant_edges_and_invalidation_candidates',
)

# A test that saves to the default graph and reads one group's own graph. It
# fails on FalkorDB for that reason, and the driver keeps FalkorDB's graph for
# each group on purpose (homelab#332, story 32). tests/test_group_graphs.py runs
# its assertions with the episodes in the group's graph.
DESELECTED_ONE_GRAPH = ('tests/test_graphiti_mock.py::test_graphiti_retrieve_episodes',)

DESELECTED = DESELECTED_DEAD_CODE + DESELECTED_ONE_GRAPH

# The skips for the FalkorDB server that ci/lift-falkordb-skips.py lifts.
LIFTED_SKIPS = 13

# 34 functions, one parametrized twice, minus the 4 deselected.
EXPECTED_PASSED = 31
