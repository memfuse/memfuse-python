import pytest
from typing import Any, Dict, Optional

from memfuse.api.messages import MessagesApi
from memfuse.memory import Memory


# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


class DummyClient:
    def __init__(self) -> None:
        self.last_call: Dict[str, Any] = {}
        self.messages = MessagesApi(self)

    async def _request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        self.last_call = {"method": method, "endpoint": endpoint, "data": data, "headers": extra_headers}
        return {"ok": True}

    def _request_sync(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        self.last_call = {"method": method, "endpoint": endpoint, "data": data, "headers": extra_headers}
        return {"ok": True}


def test_messages_api_add_accepts_metadata_sync():
    c = DummyClient()
    msgs = [
        {"role": "user", "content": "Hello", "metadata": {"task": "greeting", "mode": "casual"}},
        {"role": "assistant", "content": "Hi!"},
    ]
    c.messages.add_sync(session_id="s1", messages=msgs)
    assert c.last_call["endpoint"] == "/api/v1/sessions/s1/messages"
    assert c.last_call["data"]["messages"] == msgs


def test_memory_add_passes_metadata_sync():
    c = DummyClient()
    mem = Memory(
        client=c,
        session_id="sess-1",
        user_id="user-1",
        agent_id="agent-1",
        user_name="Alice",
        agent_name="Bob",
    )
    msgs = [
        {"role": "user", "content": "Ping", "metadata": {"task": "test"}},
        {"role": "assistant", "content": "Pong"},
    ]
    mem.add(msgs)
    assert c.last_call["endpoint"] == "/api/v1/sessions/sess-1/messages"
    assert c.last_call["data"]["messages"] == msgs

