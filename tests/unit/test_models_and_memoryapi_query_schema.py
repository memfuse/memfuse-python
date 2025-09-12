import pytest
from typing import Any, Dict, Optional

from memfuse.models.requests import Message, QueryRequest
from memfuse.api.memory import MemoryApi


pytestmark = pytest.mark.unit


def test_message_model_accepts_metadata():
    msg = Message(role="user", content="hi", metadata={"task": "greeting", "mode": "casual"})
    assert msg.role == "user"
    assert msg.content == "hi"
    assert isinstance(msg.metadata, dict)
    assert msg.metadata["task"] == "greeting"


def test_query_request_fields_match_server_schema():
    req = QueryRequest(
        query="what",
        top_k=7,
        agent_id="a1",
        session_id="s1",
        metadata={"task": "triage"},
    )
    dumped = req.model_dump()
    assert set(dumped.keys()) == {"query", "top_k", "agent_id", "session_id", "metadata"}
    assert dumped["query"] == "what"
    assert dumped["top_k"] == 7
    assert dumped["agent_id"] == "a1"
    assert dumped["session_id"] == "s1"
    assert dumped["metadata"] == {"task": "triage"}


class DummyClient:
    def __init__(self) -> None:
        self.last_call: Dict[str, Any] = {}

    async def _request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        self.last_call = {"method": method, "endpoint": endpoint, "data": data, "headers": extra_headers}
        return {"ok": True}


@pytest.mark.asyncio
async def test_memory_api_query_payload_and_deprecations():
    c = DummyClient()
    api = MemoryApi(c)

    # With deprecated args to ensure they do not appear in payload
    with pytest.warns(DeprecationWarning):
        await api.query(
            session_id="s1",
            query="hello",
            top_k=5,
            agent_id="a1",
            metadata={"task": "t"},
            store_type="vector",
            include_messages=False,
            include_knowledge=False,
        )

    assert c.last_call["endpoint"].startswith("/api/v1/memory/query?session_id=s1")
    sent = c.last_call["data"]
    assert set(sent.keys()) == {"query", "top_k", "agent_id", "session_id", "metadata"}
    assert sent["metadata"] == {"task": "t"}
    # Ensure deprecated fields are not present
    assert "store_type" not in sent and "include_messages" not in sent and "include_knowledge" not in sent

