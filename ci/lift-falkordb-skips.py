"""Lift the skips that graphiti's suite applies to the FalkorDB server.

graphiti-core's test_graphiti_mock.py skips 13 tests with
``if graph_driver.provider == GraphProvider.FALKORDB: pytest.skip('... fail on
FalkorDB')``. The skip is about the FalkorDB server. The AGE driver only borrows
the provider value FALKORDB (graphiti's provider list is closed), so these tests
(vector search, breadth-first search, the rerankers, bulk writes, communities)
must run. Upstream runs them on Neo4j. Two of the 13 are in tests that
suite/conftest.py deselects, so 11 more tests run.

The rewrite is exact: it changes only an ``if`` line whose next line is such a
skip, and it fails unless it changes exactly EXPECTED lines. A graphiti bump
that adds or removes one fails the fetch, not the gate.

Usage: python3 ci/lift-falkordb-skips.py <test_graphiti_mock.py>
"""

from __future__ import annotations

import re
import sys

EXPECTED = 13
CONDITION = 'if graph_driver.provider == GraphProvider.FALKORDB:'
SKIP = re.compile(r"pytest\.skip\('Skipping as tests? fails? on Falkor(DB|db)'\)")
LIFTED = 'if False:  # AGE: lifted by ci/lift-falkordb-skips.py'


def main(path: str) -> None:
    lines = open(path).read().split('\n')
    lifted = 0
    for index, line in enumerate(lines[:-1]):
        if line.strip() == CONDITION and SKIP.fullmatch(lines[index + 1].strip()):
            lines[index] = line.replace(CONDITION, LIFTED)
            lifted += 1
    if lifted != EXPECTED:
        sys.exit(f'{path}: lifted {lifted} FalkorDB skips, expected {EXPECTED}')
    open(path, 'w').write('\n'.join(lines))
    print(f'{path}: lifted {lifted} FalkorDB skips')


if __name__ == '__main__':
    main(sys.argv[1])
