import pytest
from typing import Any, Dict, Optional

from memfuse.api.users import UsersApi
from memfuse.memory import Memory, AsyncMemory

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


class DummyClient:
    def __init__(self) -> None:
        self.last_call: Dict[str, Any] = {}
        # Attach APIs that expect a client reference
        self.users = UsersApi(self)

    async def _request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        self.last_call = {"method": method, "endpoint": endpoint, "data": data, "headers": extra_headers}
        return {"ok": True}

    def _request_sync(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        self.last_call = {"method": method, "endpoint": endpoint, "data": data, "headers": extra_headers}
        return {"ok": True}


@pytest.mark.asyncio
async def test_users_api_query_payload_without_metadata_async():
    c = DummyClient()
    await c.users.query(
        user_id="u1",
        query="what's up",
        session_id="s1",
        agent_id="a1",
        top_k=7,
    )
    assert c.last_call["endpoint"] == "/api/v1/users/u1/query"
    sent = c.last_call["data"]
    assert set(sent.keys()) == {"query", "session_id", "agent_id", "top_k"}
    assert sent["query"] == "what's up"
    assert sent["session_id"] == "s1"
    assert sent["agent_id"] == "a1"
    assert sent["top_k"] == 7


@pytest.mark.asyncio
async def test_users_api_query_payload_with_metadata_async():
    c = DummyClient()
    meta = {"task": "triage", "mode": "interactive"}
    await c.users.query(
        user_id="u2",
        query="hi",
        session_id=None,
        agent_id=None,
        top_k=3,
        metadata=meta,
    )
    sent = c.last_call["data"]
    assert set(sent.keys()) == {"query", "session_id", "agent_id", "top_k", "metadata"}
    assert sent["metadata"] == meta


def test_users_api_query_sync_deprecated_args_warn():
    c = DummyClient()
    with pytest.warns(DeprecationWarning):
        c.users.query_sync(
            user_id="u3",
            query="hello",
            session_id="s3",
            agent_id="a3",
            top_k=5,
            store_type="vector",
            include_messages=False,
            include_knowledge=False,
        )
    sent = c.last_call["data"]
    # Deprecated fields should not be present in payload
    assert "store_type" not in sent
    assert "include_messages" not in sent
    assert "include_knowledge" not in sent
    assert set(sent.keys()) == {"query", "session_id", "agent_id", "top_k"}


def test_users_api_query_sync_payload_with_metadata():
    c = DummyClient()
    meta = {"task": "qna"}
    c.users.query_sync(
        user_id="u4",
        query="hello",
        session_id=None,
        agent_id=None,
        top_k=1,
        metadata=meta,
    )
    sent = c.last_call["data"]
    assert sent["metadata"] == meta


def test_memory_query_passes_metadata_sync():
    c = DummyClient()
    # Memory uses c.users.query_sync under the hood
    mem = Memory(
        client=c,
        session_id="sess-1",
        user_id="user-1",
        agent_id="agent-1",
        user_name="Alice",
        agent_name="Bob",
    )
    meta = {"task": "triage", "mode": "batch"}
    mem.query("question?", metadata=meta)
    sent = c.last_call["data"]
    assert sent.get("metadata") == meta
