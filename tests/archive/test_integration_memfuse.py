import os
import pytest
from memfuse import MemFuseClient
from dotenv import load_dotenv

load_dotenv()

# Read API key and base URL from environment variables
MEMFUSE_API_KEY = os.getenv("MEMFUSE_API_KEY")
MEMFUSE_BASE_URL = os.getenv("MEMFUSE_BASE_URL", "http://localhost:8000")

# Mark all tests in this file as integration
pytestmark = pytest.mark.integration

def backend_available():
    if not MEMFUSE_API_KEY:
        return False
    try:
        client = MemFuseClient(api_key=MEMFUSE_API_KEY, base_url=MEMFUSE_BASE_URL)
        return client._check_server_health()
    except Exception:
        return False

@pytest.fixture(scope="module")
def client():
    if not MEMFUSE_API_KEY:
        pytest.skip("No API key set for integration test")
    client = MemFuseClient(api_key=MEMFUSE_API_KEY, base_url=MEMFUSE_BASE_URL)
    if not client._check_server_health():
        pytest.skip("MemFuse backend not available")
    yield client
    client.close()

@pytest.fixture(scope="module")
def memory(client):
    # Use unique names to avoid collisions
    user = "integration_user"
    agent = "integration_agent"
    session = "integration_session"
    memory = client.create_memory(user=user, agent=agent, session=session)
    yield memory
    # Cleanup: delete all messages in the session
    try:
        messages = memory.get_session_messages(get_last_n_messages=100)
        if messages:
            ids = [m.get("id") for m in messages if "id" in m]
            if ids:
                memory.delete(ids)
    except Exception:
        pass

@pytest.mark.skipif(not backend_available(), reason="Backend or API key not available")
def test_create_memory(memory):
    assert memory.user == "integration_user"
    assert memory.agent == "integration_agent"
    assert memory.session == "integration_session"
    assert memory.user_id
    assert memory.agent_id
    assert memory.session_id

@pytest.mark.skipif(not backend_available(), reason="Backend or API key not available")
def test_add_and_query_message(memory):
    # Add a message
    add_result = memory.add([{"role": "user", "content": "Hello, integration test!"}])
    assert "data" in add_result

    # Query for the message
    query_result = memory.query(query="Hello")
    assert "result" in query_result or "data" in query_result

@pytest.mark.skipif(not backend_available(), reason="Backend or API key not available")
def test_get_and_delete_messages(memory):
    # Add two messages
    memory.add([
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "Second message"},
    ])
    messages = memory.get_session_messages(get_last_n_messages=10)
    assert isinstance(messages, list)
    assert any(m["content"] == "First message" for m in messages)
    # Delete messages (cleanup)
    ids = [m.get("id") for m in messages if "id" in m]
    if ids:
        del_result = memory.delete(ids)
        assert "data" in del_result 