"""Shared test fixtures and configuration for MemFuse tests."""

import pytest
from unittest.mock import Mock, AsyncMock
import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def mock_aiohttp_session():
    """Provides a mocked aiohttp ClientSession for async tests."""
    session = AsyncMock()
    
    # Mock the response context manager
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"status": "success", "data": {}})
    
    # Mock the request context manager
    session.get.return_value.__aenter__.return_value = mock_response
    session.post.return_value.__aenter__.return_value = mock_response
    session.put.return_value.__aenter__.return_value = mock_response
    session.delete.return_value.__aenter__.return_value = mock_response
    
    return session


@pytest.fixture
def mock_requests_session():
    """Provides a mocked requests Session for sync tests."""
    session = Mock()
    
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success", "data": {}}
    
    # Mock all HTTP methods
    session.get.return_value = mock_response
    session.post.return_value = mock_response
    session.put.return_value = mock_response
    session.delete.return_value = mock_response
    
    return session


@pytest.fixture
def sample_user_data():
    """Provides sample user data for testing."""
    return {
        "id": "test-user-123",
        "name": "Test User",
        "description": "A test user",
        "created_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_agent_data():
    """Provides sample agent data for testing."""
    return {
        "id": "test-agent-123",
        "name": "Test Agent",
        "description": "A test agent",
        "created_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_session_data():
    """Provides sample session data for testing."""
    return {
        "id": "test-session-123",
        "name": "Test Session",
        "user_id": "test-user-123",
        "agent_id": "test-agent-123",
        "created_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_message_data():
    """Provides sample message data for testing."""
    return {
        "id": "test-message-123",
        "content": "Hello, world!",
        "role": "user",
        "session_id": "test-session-123",
        "created_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def api_response_success():
    """Provides a standard success API response."""
    return {
        "status": "success",
        "data": {},
        "message": "Operation completed successfully"
    }


@pytest.fixture
def api_response_error():
    """Provides a standard error API response."""
    return {
        "status": "error",
        "data": None,
        "message": "An error occurred"
    }


# Environment fixtures
@pytest.fixture
def clean_environment(monkeypatch):
    """Provides a clean environment without API keys."""
    monkeypatch.delenv("MEMFUSE_API_KEY", raising=False)
    monkeypatch.delenv("MEMFUSE_BASE_URL", raising=False)
    return monkeypatch


@pytest.fixture
def mock_environment(monkeypatch):
    """Provides a mocked environment with test API key."""
    monkeypatch.setenv("MEMFUSE_API_KEY", "test-api-key")
    monkeypatch.setenv("MEMFUSE_BASE_URL", "http://test.memfuse.local")
    return monkeypatch 