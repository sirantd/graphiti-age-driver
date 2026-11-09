#!/usr/bin/env bash
# Put graphiti-core's own test suite next to suite/conftest.py:
#   suite/tests/                  graphiti's tests/ directory
#   suite/graphiti-pyproject.toml graphiti's pyproject.toml, for the version check
# The PyPI wheel holds no tests, so they come from the pinned source tarball.
# One change: the skips for the FalkorDB server are lifted
# (ci/lift-falkordb-skips.py says why).
# Usage: ci/fetch-graphiti-suite.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
suite="$here/../suite"
# shellcheck source=pins.env
source "$here/pins.env"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

curl -fsSL -o "$work/graphiti.tar.gz" "$GRAPHITI_SRC_URL"
echo "$GRAPHITI_SRC_SHA256  $work/graphiti.tar.gz" | sha256sum -c -
tar -xzf "$work/graphiti.tar.gz" -C "$work"

rm -rf "$suite/tests"
cp -R "$work/graphiti-$GRAPHITI_REF/tests" "$suite/tests"
cp "$work/graphiti-$GRAPHITI_REF/pyproject.toml" "$suite/graphiti-pyproject.toml"
python3 "$here/lift-falkordb-skips.py" "$suite/tests/test_graphiti_mock.py"
grep -m1 '^version = ' "$suite/graphiti-pyproject.toml"
