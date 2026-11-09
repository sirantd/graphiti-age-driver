# graphiti-age-driver

An [Apache AGE](https://age.apache.org/) graph driver for
[graphiti-core](https://github.com/getzep/graphiti) 0.31.2.

It replaces graphiti's FalkorDB driver in a homelab deployment
([homelab#332](https://github.com/sirantd/homelab/issues/332)). The graphs move
into AGE, a PostgreSQL extension, in a shared PostgreSQL 18 server. The graphs
get WAL, MVCC, `statement_timeout`, `pg_dump` and disk-resident storage.

## Status

**Stage B of homelab#332: not deployed.** The CI gate runs against the
production store:

* graphiti-core's own suite must pass all 31 selected tests, with graphiti's
  driver fixture shadowed by this driver. The three suite files have 35 test
  runs (one test is parametrized twice), and four of them are deselected by
  name (see [The graphiti suite](#the-graphiti-suite)).
* Every statement that the driver emits must read its label tables through an
  index (see [The plan gate](#the-plan-gate)).

The graphiti MCP server needs one more patch for an `age` provider. That patch
is a later stage of homelab#332, in the homelab image build.

## Use

graphiti-core is a peer dependency: install the driver into an environment
that already has graphiti-core 0.31.2. Pin the driver by commit SHA.

```python
from graphiti_core import Graphiti
from graphiti_age_driver import AGEDriver

driver = AGEDriver(host='localhost', user='graphiti', password='...', dbname='graphiti')
await driver.build_indices_and_constraints()
graphiti = Graphiti(graph_driver=driver, llm_client=..., embedder=..., cross_encoder=...)
```

| argument | default | meaning |
|---|---|---|
| `host`, `port` | `localhost`, `5432` | the PostgreSQL server |
| `user`, `password` | none | the graph role |
| `dbname` | `graphiti` | the PostgreSQL database that holds every group graph |
| `database` | `default_db` | the graph of graphiti's default group |
| `embedding_dim` | `1024` | the dimension of the HNSW indexes |
| `embedding_dims_strict` | `True` | refuse a write of an embedding of another dimension |
| `strict` | `False` | refuse an unreviewed query on the generic path |
| `extra_generic_queries` | `()` | more reviewed texts for the generic path |

`build_indices_and_constraints()` creates the default graph, its labels and its
indexes, and raises an error if it cannot. A group graph is created in the same
way on its first statement. If that creation fails, the statement fails, and
the next statement tries again. So the driver never runs on an unindexed graph.

If the server is down when the driver opens its pool, the call fails, and the
next call opens a new pool. After `close()`, the next call also opens a new
pool.

### What the server must have

1. AGE 1.8.0 available to the cluster. In production the files come from a
   read-only mount (`ci/pins.env`, `ci/fetch-age.sh`) and two settings
   (`ci/init-cluster.sql`).
2. A graph database with the `age` and `vector` extensions, and a graph role
   that is **not** a superuser. The role's own settings hold the AGE preload,
   the search path and a `statement_timeout` (`ci/init-graph-db.sql`). The
   search path is load-bearing: AGE's own label DDL fails without `ag_catalog`
   in it.

🚨 The driver never sends `LOAD 'age'`. For a non-superuser it fails with
`access to library "age" is not allowed`. AGE comes from the role's
`session_preload_libraries`, for the graph database only.

## Store layout

* One AGE graph holds one graphiti group, as one FalkorDB graph does now. The
  driver reports `GraphProvider.FALKORDB`, so graphiti clones the driver for a
  single group (`clone()`), and each clone works on its group's graph. All
  clones share one connection pool.
* graphiti's default group id is `_`. AGE refuses a graph name shorter than 3
  characters, so `_` and the empty group map to the graph `default_db`.
* The vertex labels are `Entity`, `Episodic`, `Community` and `Saga`. The edge
  labels are `RELATES_TO`, `MENTIONS`, `HAS_MEMBER`, `HAS_EPISODE` and
  `NEXT_EPISODE`.
* AGE gives a vertex one label. The extra labels of an entity go in the list
  property `labels`, without `Entity`. The label filters of a search read that
  property.
* Timestamps are UTC ISO strings, as in the FalkorDB driver. A null value is an
  absent property, as on FalkorDB.

🚨 A group id must be a valid AGE graph name: 3 to 63 letters, digits, `_`,
`.` or `-`, with a letter or `_` first and no `.` or `-` last. `clone('ab-')`
raises `ValueError`. graphiti accepts more: check every live group id before a
migration.

### Indexes

Each graph gets these indexes when it is created:

| label | index |
|---|---|
| every label | btree on `uuid` (unique), btree on `(group_id, uuid)` |
| every edge label | btree on the `id` column |
| `Entity` | btree on `name`; GIN on `labels`; GIN full text on `name`, `summary`; HNSW on `name_embedding` |
| `Episodic` | btree on `valid_at`; GIN full text on `content`, `source`, `source_description` |
| `Community` | GIN full text on `name`; HNSW on `name_embedding` |
| `Saga` | btree on `name` |
| `RELATES_TO` | btree on `name`; GIN full text on `name`, `fact`; HNSW on `fact_embedding` |

AGE 1.8.0 makes the `id` index of a vertex label, and the `start_id` and
`end_id` indexes of an edge label. It makes no `id` index for an edge label.

* 🚨 The `id` index of an edge label is the driver's. AGE's `SET` and `DELETE`
  find the row to change through a btree index on `id`, else they scan the
  whole label table (`src/backend/executor/cypher_set.c`). The plan gate cannot
  see that scan, so `tests/test_group_graphs.py` checks the catalog.
* The `(group_id, uuid)` index serves graphiti's pages of a group, newest uuid
  first. It also serves every lookup by `group_id`.
* An HNSW index is partial on its dimension: `vector(N)` where
  `vector_dims(...) = N`. A row of another dimension is not in the index, and
  no search finds it. For this reason `embedding_dims_strict` refuses that
  write with `ValueError`.
* The full-text indexes use the `english` configuration. graphiti's FalkorDB
  query words (the same separators and stop words) are joined with OR.
  `ts_rank` replaces the RediSearch score, so the order of hits changes.

## Query rules

The driver implements every method of graphiti's graph-operations interface
(85) and search interface (14). graphiti uses these first, so graphiti's
FalkorDB Cypher never runs. Each method keeps the rows, the order and the
errors of graphiti's FalkorDB fallback.

Each rule is a measurement on AGE 1.8.0, and each has a test:

* A lookup by a list of values is `UNWIND $values AS v ... WHERE n.p = v`.
  `WHERE n.p IN $values` compiles to `agtype_in_operator`, which no index
  serves.
* A lookup uses `WHERE`, not a pattern map. `{uuid: $x}` compiles to
  `properties @> {...}`, which no btree index serves.
* A vertex write uses no `MERGE`. It creates the absent uuids, then sets every
  vertex through the unique `uuid` index, in one transaction.
* `[e:A|B]` is a syntax error. A statement over several edge labels is one
  statement for each label.
* An undirected pattern plans as an `OR` join over all vertices. Each
  undirected read is two directed reads.
* `SET n = map` stores a null as a null. So null values never leave Python.
* A label is created in a statement of its own.
* Full-text, vector and breadth-first search run in SQL on the label tables,
  because their indexes are PostgreSQL expression indexes. Breadth-first
  search is a recursive CTE over the `start_id` indexes of `RELATES_TO` and
  `MENTIONS`.

### Concurrent writers

The unique `uuid` indexes turn a race of two writers into a unique violation.
The driver runs the losing transaction again, up to five times. A deadlock, a
serialization failure, and AGE's `XX000 Entity failed to be updated` are run
again in the same way. The batches are sorted, so that two writers take the
index locks in the same order.

## The generic query path

A small number of graphiti queries go around the two interfaces to
`execute_query()` or `session().run()`. The driver runs them as portable
Cypher, with one result column for each item of the last `RETURN`.

* 🚨 A query with a FalkorDB or Neo4j function (`db.idx.fulltext`, `vecf32`,
  `vec.cosineDistance`, `db.index`, `CALL db.`) is refused with
  `UnsupportedQueryError`, before any connection. Such a query comes from a
  FalkorDB branch of graphiti that leaked past the interfaces.
* Every other text is compared with `GENERIC_QUERIES`: the saga lookup and the
  mention count of `remove_episode` in graphiti-core, and the node count of the
  MCP server's `get_status`. If `strict` is set, an unknown text is refused.
  Otherwise it runs, with one warning for each distinct text.
* The saga lookup and the mention count use a pattern map. AGE compiles it to
  `properties @> {...}`, which no btree index serves. So the driver runs each
  of these texts in an equivalent `WHERE` form.
* `tests/test_generic_path.py` reads each reviewed graphiti text from
  graphiti-core's source. A graphiti bump that changes a text fails CI.

## The graphiti suite

`suite/conftest.py` takes the place of graphiti's root conftest. It shadows
only the `graph_driver` fixture with this driver, in strict mode.

* The suite is upstream's database job: the mocked-LLM files
  `test_edge_int.py`, `test_graphiti_mock.py` and `test_node_int.py`.
* `ci/fetch-graphiti-suite.sh` fetches graphiti's tests from the pinned source
  tarball and lifts the 13 skips for the FalkorDB server
  (`ci/lift-falkordb-skips.py`). The driver only borrows the provider value.
* Three deselected tests test functions that no code of graphiti-core or of
  the MCP server calls. Each one runs FalkorDB Cypher directly.
* `test_graphiti_retrieve_episodes` is deselected because it saves to the
  default graph and reads a group graph. It fails on FalkorDB for the same
  reason. `tests/test_group_graphs.py` runs its assertions in the group graph.
* `suite/suite_census.py` lists every suite test and each deselection.
  `tests/test_suite_pins.py` checks the lists against the fetched suite.
* 🚨 With `AGE_REQUIRED` set, the run fails unless all 31 tests pass and none
  skips. Run the whole suite when you set it.

## The plan gate

`tests/test_plans.py` calls every interface method on a seeded graph with 1,000
filler rows on each label. It records each statement at the one place the
driver sends SQL. Then it asks PostgreSQL for the plan of exactly that
statement with its parameters. With sequential scans, hash joins and merge
joins disabled, the plan must have no disabled node. Every scan of a label
table must carry an `Index Cond`, a `Recheck Cond` or an `Order By`.

* A census fails when an interface method is missing from the workload.
* Three calls delete or read a whole graph by design, and they are listed by
  name.
* graphiti's reviewed texts on the generic path get the same check, with the
  parameters that graphiti sends. The `get_status` count reads the whole graph
  by design.
* An unlabeled pattern also reads AGE's default-label tables. Every statement
  of the driver writes a labeled row, so the test asserts that these tables
  stay empty.

## Testing

### Local

Make a virtual environment, fetch the graphiti suite, and run the server-free
tests:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e '.[test]'
ci/fetch-graphiti-suite.sh
.venv/bin/pytest -m "not integration"
```

Start the production image with AGE mounted, as CI does. On an arm64 laptop,
set `AGE_ARCH=arm64`:

```sh
AGE_ARCH=arm64 ci/fetch-age.sh .ci-age
AGE_CONTAINER=graphiti-age AGE_PORT=55435 ci/start-postgres.sh .ci-age
AGE_PORT=55435 AGE_REQUIRED=1 .venv/bin/pytest
AGE_PORT=55435 AGE_REQUIRED=1 .venv/bin/pytest -m "not integration" suite/test_shadow.py \
  suite/tests/test_edge_int.py suite/tests/test_graphiti_mock.py suite/tests/test_node_int.py
```

### CI

* The `surface` job runs the server-free tests: the interface census, the
  generic-path guard and the suite census.
* The `contract` job runs graphiti's suite and the integration tests on the
  production image digest, as uid 999 with its `PGDATA` and command. AGE is
  mounted read-only, and the tests connect as a non-superuser graph role.
  `AGE_REQUIRED=1` turns an absent server into a failure, not a skip.
* The `age-source` workflow proves that the pinned PGDG package is the ASF
  1.8.0 release (`ci/verify-age-source.sh`). It runs when a pin changes.

### Pins

The pins are in `pyproject.toml` (graphiti-core for the tests) and
`ci/pins.env` (the PostgreSQL image, the AGE package, and graphiti's source
tarball). `tests/test_suite_pins.py` fails when the graphiti-core pin and the
fetched source disagree.

⚠ PGDG moves superseded packages from `apt.postgresql.org` to
`apt-archive.postgresql.org`. If `ci/fetch-age.sh` fails with a 404, point
`AGE_DEB_URL` at the archive; the sha256 pin stays.
