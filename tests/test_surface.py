"""The driver surface against graphiti-core 0.31.2 — server-free.

graphiti-core asks the driver's two interface objects first, and on a
``NotImplementedError`` (or with no object) it runs its own provider Cypher.
That Cypher is written for Neo4j and FalkorDB: on AGE it fails
(``[e:A|B]`` is a syntax error) or reads every row (``{uuid: $x}``). So every
method of both interfaces must be the driver's own, with the parameter names and
the order that graphiti uses: graphiti passes most arguments by position.
"""

from __future__ import annotations

import inspect

import pytest
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.driver.search_interface.search_interface import SearchInterface

from graphiti_age_driver import AGEDriver
from graphiti_age_driver.operations import AGEGraphOperations
from graphiti_age_driver.search import AGESearchInterface


def _interface_methods(interface: type) -> list[str]:
    return [
        name
        for name, member in vars(interface).items()
        if inspect.isfunction(member) and not name.startswith('_')
    ]


@pytest.mark.parametrize(
    ('interface', 'implementation', 'count'),
    [
        # The counts are the census of graphiti-core 0.31.2. A new method in a
        # bump changes the count and fails here, before it falls back to Cypher.
        (GraphOperationsInterface, AGEGraphOperations, 85),
        (SearchInterface, AGESearchInterface, 14),
    ],
)
def test_every_interface_method_is_the_drivers_own(interface, implementation, count):
    methods = _interface_methods(interface)
    assert len(methods) == count

    inherited = [name for name in methods if name not in vars(implementation)]
    assert inherited == [], f'falls back to graphiti Cypher: {inherited}'

    for name in methods:
        declared = list(inspect.signature(getattr(interface, name)).parameters)
        implemented = list(inspect.signature(getattr(implementation, name)).parameters)
        assert implemented == declared, name


def test_the_driver_reports_falkordb_and_carries_both_interfaces():
    """graphiti-core's provider list is closed. FALKORDB selects one graph for
    each group, string timestamps and the request-scope clone (homelab#332)."""
    driver = AGEDriver(host='127.0.0.1', port=1, user='x', password='x', dbname='x')

    assert isinstance(driver, GraphDriver)
    assert driver.provider == GraphProvider.FALKORDB
    assert driver.default_group_id == '_'
    assert isinstance(driver.graph_operations_interface, AGEGraphOperations)
    assert isinstance(driver.search_interface, AGESearchInterface)


def test_construction_opens_no_connection():
    """graphiti and the MCP server construct the driver outside a running loop.
    Port 1 refuses every connection, so any connect here would raise."""
    AGEDriver(host='127.0.0.1', port=1, user='x', password='x', dbname='x')
