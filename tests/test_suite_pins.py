"""The pins and the census of graphiti-core's suite — server-free.

The gate runs graphiti-core's own tests from the source tarball of the release
that the image installs from PyPI. These checks make sure that the two are one
release, and that the gate runs, deselects and lifts exactly what
``suite/suite_census.py`` says. Run ``ci/fetch-graphiti-suite.sh`` first; a
missing suite fails here, it does not skip.
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import pathlib
import tomllib

import pytest

SUITE = pathlib.Path(__file__).resolve().parent.parent / 'suite'


def _census():
    spec = importlib.util.spec_from_file_location('suite_census', SUITE / 'suite_census.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


census = _census()


@pytest.fixture(scope='module')
def fetched() -> pathlib.Path:
    if not (SUITE / 'tests').is_dir():
        pytest.fail('the graphiti suite is not fetched: run ci/fetch-graphiti-suite.sh')
    return SUITE


def _test_functions(path: pathlib.Path) -> list[ast.AsyncFunctionDef | ast.FunctionDef]:
    tree = ast.parse(path.read_text())
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_')
    ]


def test_the_suite_is_the_release_that_is_installed(fetched):
    """📌 0.31.2 is graphiti_core_version in the homelab role truenas-graphiti."""
    installed = importlib.metadata.version('graphiti-core')
    source = tomllib.loads((fetched / 'graphiti-pyproject.toml').read_text())['project']['version']

    assert installed == '0.31.2'
    assert source == installed


def test_the_census_names_every_test_of_the_suite_files(fetched):
    found = [
        f'{name}::{function.name}'
        for name in census.SUITE_FILES
        for function in _test_functions(fetched / name)
    ]

    assert found == list(census.SUITE_TESTS)


def test_every_suite_test_takes_the_shadowed_fixture(fetched):
    for name in census.SUITE_FILES:
        for function in _test_functions(fetched / name):
            assert 'graph_driver' in [arg.arg for arg in function.args.args], function.name


def test_each_deselected_test_is_a_suite_test():
    assert set(census.DESELECTED) <= set(census.SUITE_TESTS)
    assert len(census.DESELECTED) == len(set(census.DESELECTED)) == 4


def test_the_fetch_lifted_exactly_the_falkordb_skips(fetched):
    text = (fetched / 'tests/test_graphiti_mock.py').read_text()

    assert text.count('if False:  # AGE: lifted by ci/lift-falkordb-skips.py') == census.LIFTED_SKIPS
    assert 'if graph_driver.provider == GraphProvider.FALKORDB:' not in text
