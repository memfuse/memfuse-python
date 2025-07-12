import pytest
from memfuse import MemFuse

@pytest.mark.integration
def test_scope_real_backend():
    memfuse = MemFuse()  # Uses default base_url
    user = "integration_test_user"
    agent = "integration_test_agent"
    session = "integration_test_session"

    result = memfuse.scope(user, agent, session)
    # Adjust the following assertions based on your backend's actual response structure
    assert "data" in result
    assert "session_id" in result["data"]
    assert isinstance(result["data"]["session_id"], str)