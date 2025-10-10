import asyncio
import pytest

from memfuse.client import AsyncMemFuse, MemFuse
from memfuse.utils import MemFuseHTTPError


class AsyncStubAgents:
    def __init__(self):
        self.create_calls = 0
        self.get_calls = 0
        self._behavior = "success"  # success | conflict | server_error | network_error
        self._server_error_until = 0

    def set_conflict(self):
        self._behavior = "conflict"

    def set_server_error_for_calls(self, n: int):
        self._behavior = "server_error"
        self._server_error_until = n

    def set_network_error(self):
        self._behavior = "network_error"

    async def create(self, name: str, description: str, idempotency_key: str = None):
        self.create_calls += 1
        # Simulate slight delay to expose concurrency
        await asyncio.sleep(0.05)

        if self._behavior == "conflict":
            raise MemFuseHTTPError("already exists", 409, {"message": "exists"})
        if self._behavior == "server_error":
            if self.create_calls <= self._server_error_until:
                raise MemFuseHTTPError("server error", 500, {"message": "error"})
        if self._behavior == "network_error":
            import aiohttp
            raise aiohttp.ClientError("network down")

        return {"data": {"agent": {"id": f"agent-{name}"}}}

    async def get_by_name(self, name: str):
        self.get_calls += 1
        await asyncio.sleep(0.01)
        return {"data": {"agents": [{"id": f"agent-{name}"}]}}


class SyncStubAgents:
    def __init__(self):
        self.create_calls = 0
        self.get_calls = 0
        self._behavior = "success"
        self._server_error_until = 0

    def set_conflict(self):
        self._behavior = "conflict"

    def set_server_error_for_calls(self, n: int):
        self._behavior = "server_error"
        self._server_error_until = n

    def set_network_error(self):
        self._behavior = "network_error"

    def create_sync(self, name: str, description: str, idempotency_key: str = None):
        self.create_calls += 1
        if self._behavior == "conflict":
            raise MemFuseHTTPError("already exists", 409, {"message": "exists"})
        if self._behavior == "server_error":
            if self.create_calls <= self._server_error_until:
                raise MemFuseHTTPError("server error", 500, {"message": "error"})
        if self._behavior == "network_error":
            import requests
            raise requests.exceptions.RequestException("network down")

        return {"data": {"agent": {"id": f"agent-{name}"}}}

    def get_by_name_sync(self, name: str):
        self.get_calls += 1
        return {"data": {"agents": [{"id": f"agent-{name}"}]}}


def test_async_singleflight_agent_create_only_once():
    client = AsyncMemFuse()
    stub = AsyncStubAgents()
    client.agents = stub

    async def run():
        async def task():
            return await client._get_or_create_agent("agent_default")

        results = await asyncio.gather(*(task() for _ in range(10)))
        assert all(r == "agent-agent_default" for r in results)
        # Only one create should be issued due to singleflight
        assert stub.create_calls == 1
        assert stub.get_calls == 0

    asyncio.run(run())


def test_async_post_409_then_get_by_name_returns():
    client = AsyncMemFuse()
    stub = AsyncStubAgents()
    stub.set_conflict()
    client.agents = stub

    async def run():
        agent_id = await client._get_or_create_agent("agent_default")
        assert agent_id == "agent-agent_default"
        assert stub.create_calls == 1
        assert stub.get_calls >= 1

    asyncio.run(run())


def test_async_retry_on_5xx_then_success():
    client = AsyncMemFuse()
    stub = AsyncStubAgents()
    stub.set_server_error_for_calls(2)  # first 2 attempts fail with 500
    client.agents = stub

    async def run():
        agent_id = await client._get_or_create_agent("agent_default")
        assert agent_id == "agent-agent_default"
        assert stub.create_calls == 3  # retried twice, then success
        assert stub.get_calls == 0

    asyncio.run(run())


def test_async_retry_on_network_error_then_fail_after_retries():
    client = AsyncMemFuse()
    stub = AsyncStubAgents()
    stub.set_network_error()
    client.agents = stub

    async def run():
        with pytest.raises(Exception):
            await client._get_or_create_agent("agent_default")
        assert stub.create_calls >= 3  # exhausted retries

    asyncio.run(run())


def test_sync_post_409_then_get_by_name_returns():
    client = MemFuse()
    stub = SyncStubAgents()
    stub.set_conflict()
    client.agents = stub

    agent_id = client._get_or_create_agent_sync("agent_default")
    assert agent_id == "agent-agent_default"
    assert stub.create_calls == 1
    assert stub.get_calls >= 1
