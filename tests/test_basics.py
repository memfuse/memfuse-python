from memfuse import __version__
import pytest
from unittest.mock import patch, MagicMock
from memfuse import MemFuse


def test_version():
    assert isinstance(__version__, str)


def test_scope_makes_post_request():
    memfuse = MemFuse()
    user = "test_user"
    agent = "test_agent"
    session = "test_session"
    expected_url = f"{memfuse.base_url}/init"
    expected_payload = {"user": user, "agent": agent, "session": session}
    mock_response_data = {"data": {"session_id": "abc123"}}

    with patch("httpx.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = memfuse.scope(user, agent, session)

        mock_post.assert_called_once_with(expected_url, json=expected_payload)
        assert result == mock_response_data