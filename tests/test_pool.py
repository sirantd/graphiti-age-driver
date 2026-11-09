"""The connection pool after a failure or a close.

The shared PostgreSQL app restarts for its own maintenance. A driver that
starts while the server is down must work when the server is back, and a
closed driver must work again on its next call.
"""

from __future__ import annotations

import asyncio
import contextlib

import psycopg
import pytest
from conftest import DATABASE, HOST, PASSWORD, PORT, USER, drop_graph, unique_name, unreachable

import graphiti_age_driver.driver as driver_module
from graphiti_age_driver import AGEDriver

pytestmark = pytest.mark.integration

COUNT = 'MATCH (n) RETURN count(n) as count'  # the reviewed get_status text


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    with contextlib.suppress(ConnectionError):
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    writer.close()


class _Relay:
    """A TCP relay to the AGE server. While it is down, it closes every
    connection that it accepts, as a server that restarts does."""

    def __init__(self) -> None:
        self.up = False
        self.server: asyncio.Server | None = None

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._relay, '127.0.0.1', 0)
        return self.server.sockets[0].getsockname()[1]

    async def _relay(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if not self.up:
            writer.close()
            return
        upstream_reader, upstream_writer = await asyncio.open_connection(HOST, PORT)
        await asyncio.gather(_pipe(reader, upstream_writer), _pipe(upstream_reader, writer))

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()


async def test_a_closed_driver_works_again_on_its_next_call(driver):
    await driver.close()

    records, _, _ = await driver.execute_query(COUNT)

    assert records == [{'count': 0}]


async def test_a_driver_that_starts_while_the_server_is_down_works_when_it_is_back(
    monkeypatch,
):
    monkeypatch.setattr(driver_module, '_POOL_OPEN_TIMEOUT', 1.0)
    relay = _Relay()
    graph = unique_name('test')
    port = await relay.start()
    driver = AGEDriver(
        host='127.0.0.1', port=port, user=USER, password=PASSWORD, dbname=DATABASE, database=graph
    )
    try:
        with pytest.raises(psycopg.OperationalError):
            await driver.build_indices_and_constraints()

        relay.up = True
        try:
            await driver.build_indices_and_constraints()
        except (psycopg.OperationalError, OSError) as exc:
            unreachable(exc)
        records, _, _ = await driver.execute_query(COUNT)

        assert records == [{'count': 0}]
    finally:
        await driver.close()
        await relay.stop()
        drop_graph(graph)
