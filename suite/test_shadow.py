"""The shadow takes effect: graphiti's tests get the AGE driver, on the graph role.

Without this check a fixture that silently fell back to graphiti's own (with
no driver enabled, an empty parameter set) would skip every test, and the
skips would read as a green gate.
"""

from __future__ import annotations

from graphiti_age_driver import AGEDriver


async def test_graph_driver_is_the_age_driver(graph_driver):
    assert isinstance(graph_driver, AGEDriver)
    assert graph_driver.embedding_dim == 384


async def test_the_suite_runs_as_a_role_that_is_not_a_superuser(graph_driver):
    """A client that sends LOAD 'age' fails on this role (homelab#332, story 60)."""
    rows = await graph_driver._sql(
        'SELECT current_user, rolsuper FROM pg_roles WHERE rolname = current_user'
    )
    assert rows[0][1] is False
