-- The graphiti graph database and its role, as production has them
-- (homelab#332, Stage C). Run as the superuser, connected to the maintenance
-- database:
--   psql -v role=graphiti -v password=... -v database=graphiti -f init-graph-db.sql
--
-- Each graphiti group is one AGE graph inside this one database. The driver
-- creates a graph, its labels and its indexes on the first use of the group.
--
-- 🚨 The role is NOT a superuser. A client that sends LOAD 'age' fails on it with
-- `access to library "age" is not allowed`, so a gate that connects as this role
-- proves that no client sends it.

CREATE ROLE :"role" LOGIN PASSWORD :'password';
CREATE DATABASE :"database" OWNER :"role";

\connect :"database"

-- AGE only in the graph databases. The paperclip database never gets it.
CREATE EXTENSION age;
GRANT USAGE ON SCHEMA ag_catalog TO :"role";
-- pgvector, for the HNSW indexes on the embedding properties. The vector type
-- lands in the public schema, which the role owns as the database owner.
CREATE EXTENSION vector;

-- Per role and database, never for the whole cluster: the fix for utility
-- commands in a database without AGE under a cluster-wide preload (#2578) is on
-- AGE master only.
ALTER ROLE :"role" IN DATABASE :"database" SET session_preload_libraries = 'age';
-- Load-bearing: AGE's own label DDL resolves graphid_ops through the search path,
-- and the driver's SQL names the vector type and its operators without a schema.
ALTER ROLE :"role" IN DATABASE :"database" SET search_path = ag_catalog, "$user", public;
-- homelab#332 story 6: one long graph query stops and cannot freeze the store.
-- 5 min is the value of the cognee graph role; Stage C confirms the production
-- value for both roles.
ALTER ROLE :"role" IN DATABASE :"database" SET statement_timeout = '5min';
