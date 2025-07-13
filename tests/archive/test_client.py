import os
import pytest
from memfuse import MemFuseClient
from memfuse.api import MemoryApi, HealthApi
from memfuse import Memory


class TestMemFuseClientInitialization:
    def test_init_with_explicit_api_key(self):
        client = MemFuseClient(api_key="mykey", base_url="http://localhost:1234/")
        assert client.api_key == "mykey"
        assert client.base_url == "http://localhost:1234"

    def test_init_with_env_api_key(self, monkeypatch):
        monkeypatch.setenv("MEMFUSE_API_KEY", "envkey")
        client = MemFuseClient(api_key=None, base_url="http://localhost:8000/")
        assert client.api_key == "envkey"
        assert client.base_url == "http://localhost:8000"

    def test_init_with_no_api_key(self, monkeypatch):
        monkeypatch.delenv("MEMFUSE_API_KEY", raising=False)
        client = MemFuseClient(api_key=None, base_url="http://localhost:8000/")
        assert client.api_key is None
        assert client.base_url == "http://localhost:8000"

    def test_base_url_trailing_slash(self):
        client = MemFuseClient(api_key="key", base_url="http://localhost:9999/")
        assert client.base_url == "http://localhost:9999"

    def test_base_url_no_trailing_slash(self):
        client = MemFuseClient(api_key="key", base_url="http://localhost:9999")
        assert client.base_url == "http://localhost:9999"

    def test_session_is_none_on_init(self):
        client = MemFuseClient(api_key="key")
        assert client.session is None

    def test_memory_and_health_api_clients(self):
        client = MemFuseClient(api_key="key")
        assert isinstance(client.memory, MemoryApi)
        assert isinstance(client.health, HealthApi)

    def test_instance_tracking(self):
        # Clear instances for test isolation
        MemFuseClient._instances.clear()
        client = MemFuseClient(api_key="key")
        assert client in MemFuseClient._instances

class TestMemFuseClientSessionManagement:
    def test_ensure_session_creates_client(self, mocker):
        mock_client = mocker.patch("httpx.Client")
        client = MemFuseClient(api_key=None)
        client.session = None
        client._ensure_session()
        mock_client.assert_called_once()
        assert client.session == mock_client()

    def test_ensure_session_sets_auth_header(self, mocker):
        mock_client = mocker.patch("httpx.Client")
        client = MemFuseClient(api_key="secret")
        client.session = None
        client._ensure_session()
        mock_client.assert_called_once_with(headers={"Authorization": "Bearer secret"}, follow_redirects=True)

    def test_ensure_session_does_not_recreate(self, mocker):
        mock_client = mocker.patch("httpx.Client")
        client = MemFuseClient(api_key=None)
        fake_session = object()
        client.session = fake_session
        client._ensure_session()
        mock_client.assert_not_called()
        assert client.session is fake_session

    def test_close_session_closes_and_resets(self, mocker):
        client = MemFuseClient(api_key=None)
        mock_session = mocker.Mock()
        client.session = mock_session
        client._close_session()
        mock_session.close.assert_called_once()
        assert client.session is None

    def test_close_session_when_none(self):
        client = MemFuseClient(api_key=None)
        client.session = None
        # Should not raise
        client._close_session()
        assert client.session is None

class TestMemFuseClientHealthCheck:
    def test_returns_true_for_2xx(self, mocker):
        client = MemFuseClient(api_key=None)
        mock_session = mocker.Mock()
        mock_response = mocker.Mock(status_code=200)
        mock_session.get.return_value = mock_response
        client.session = mock_session
        assert client._check_server_health() is True

    def test_returns_true_for_3xx(self, mocker):
        client = MemFuseClient(api_key=None)
        mock_session = mocker.Mock()
        mock_response = mocker.Mock(status_code=302)
        mock_session.get.return_value = mock_response
        client.session = mock_session
        assert client._check_server_health() is True

    def test_returns_false_for_4xx_5xx(self, mocker):
        client = MemFuseClient(api_key=None)
        mock_session = mocker.Mock()
        mock_response = mocker.Mock(status_code=404)
        mock_session.get.return_value = mock_response
        client.session = mock_session
        assert client._check_server_health() is False
        mock_response2 = mocker.Mock(status_code=500)
        mock_session.get.return_value = mock_response2
        assert client._check_server_health() is False

    def test_returns_false_on_exception(self, mocker):
        client = MemFuseClient(api_key=None)
        mock_session = mocker.Mock()
        mock_session.get.side_effect = Exception("connection error")
        client.session = mock_session
        assert client._check_server_health() is False

    def test_ensure_session_called(self, mocker):
        client = MemFuseClient(api_key=None)
        client.session = None
        ensure_session = mocker.patch.object(client, "_ensure_session")
        mock_session = mocker.Mock()
        mock_response = mocker.Mock(status_code=200)
        # After _ensure_session, session should be set
        def set_session():
            client.session = mock_session
            mock_session.get.return_value = mock_response
        ensure_session.side_effect = set_session
        assert client._check_server_health() is True
        ensure_session.assert_called_once()

class TestMemFuseClientRequestLogic:
    def test_calls_ensure_session_and_health(self, mocker):
        client = MemFuseClient(api_key=None)
        ensure_session = mocker.patch.object(client, "_ensure_session")
        check_health = mocker.patch.object(client, "_check_server_health", return_value=True)
        mock_session = mocker.Mock()
        client.session = mock_session
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_session.get.return_value = mock_response
        result = client._request("get", "/api/v1/test")
        ensure_session.assert_called_once()
        check_health.assert_called_once()
        assert result == {"ok": True}

    def test_raises_connection_error_on_health_fail(self, mocker):
        client = MemFuseClient(api_key=None)
        mocker.patch.object(client, "_ensure_session")
        mocker.patch.object(client, "_check_server_health", return_value=False)
        with pytest.raises(ConnectionError) as exc:
            client._request("get", "/api/v1/test")
        assert "Cannot connect to MemFuse server" in str(exc.value)

    @pytest.mark.parametrize("method", ["get", "post", "put", "delete"])
    def test_uses_correct_http_method_and_endpoint(self, mocker, method):
        client = MemFuseClient(api_key=None)
        mocker.patch.object(client, "_ensure_session")
        mocker.patch.object(client, "_check_server_health", return_value=True)
        mock_session = mocker.Mock()
        client.session = mock_session
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        setattr(mock_session, method, mocker.Mock(return_value=mock_response))
        data = {"foo": "bar"}
        if method == "get":
            result = client._request(method, "/api/v1/test", data)
            getattr(mock_session, method).assert_called_once_with(f"http://localhost:8000/api/v1/test")
        else:
            result = client._request(method, "/api/v1/test", data)
            getattr(mock_session, method).assert_called_once_with(f"http://localhost:8000/api/v1/test", json=data)
        assert result == {"ok": True}

    def test_get_does_not_send_json(self, mocker):
        client = MemFuseClient(api_key=None)
        mocker.patch.object(client, "_ensure_session")
        mocker.patch.object(client, "_check_server_health", return_value=True)
        mock_session = mocker.Mock()
        client.session = mock_session
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_session.get.return_value = mock_response
        client._request("get", "/api/v1/test", {"foo": "bar"})
        mock_session.get.assert_called_once_with("http://localhost:8000/api/v1/test")

    def test_returns_parsed_json_on_success(self, mocker):
        client = MemFuseClient(api_key=None)
        mocker.patch.object(client, "_ensure_session")
        mocker.patch.object(client, "_check_server_health", return_value=True)
        mock_session = mocker.Mock()
        client.session = mock_session
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"foo": "bar"}
        mock_session.get.return_value = mock_response
        result = client._request("get", "/api/v1/test")
        assert result == {"foo": "bar"}

    def test_raises_on_non_json_response(self, mocker):
        client = MemFuseClient(api_key=None)
        mocker.patch.object(client, "_ensure_session")
        mocker.patch.object(client, "_check_server_health", return_value=True)
        mock_session = mocker.Mock()
        client.session = mock_session
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = Exception("not json")
        mock_session.get.return_value = mock_response
        with pytest.raises(Exception) as exc:
            client._request("get", "/api/v1/test")
        assert "API response is not valid JSON" in str(exc.value)

    def test_raises_on_api_error_status(self, mocker):
        client = MemFuseClient(api_key=None)
        mocker.patch.object(client, "_ensure_session")
        mocker.patch.object(client, "_check_server_health", return_value=True)
        mock_session = mocker.Mock()
        client.session = mock_session
        mock_response = mocker.Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "bad request"}
        mock_session.post.return_value = mock_response
        with pytest.raises(Exception) as exc:
            client._request("post", "/api/v1/test", {"foo": "bar"})
        assert "API request failed: bad request" in str(exc.value)

    def test_raises_connection_error_on_connect_error(self, mocker):
        import httpx
        client = MemFuseClient(api_key=None)
        mocker.patch.object(client, "_ensure_session")
        mocker.patch.object(client, "_check_server_health", return_value=True)
        mock_session = mocker.Mock()
        client.session = mock_session
        mock_session.get.side_effect = httpx.ConnectError("fail")
        with pytest.raises(ConnectionError) as exc:
            client._request("get", "/api/v1/test")
        assert "Cannot connect to MemFuse server" in str(exc.value)

class TestMemFuseClientCreateMemory:
    def test_user_and_agent_do_not_exist(self, mocker):
        client = MemFuseClient(api_key=None)
        mock_request = mocker.patch.object(client, "_request")
        # Simulate user GET returns no users
        mock_request.side_effect = [
            {"status": "error", "data": {"users": []}},  # user GET
            {"data": {"user": {"id": "user123"}}},      # user POST
            {"status": "error", "data": {"agents": []}}, # agent GET
            {"data": {"agent": {"id": "agent456"}}},    # agent POST
            {"data": {"session": {"id": "sess789", "name": "mysession"}}}, # session POST
        ]
        memory = client.create_memory(user="alice", agent="bob", session="mysession")
        assert isinstance(memory, Memory)
        assert memory.user == "alice"
        assert memory.agent == "bob"
        assert memory.session == "mysession"
        assert memory.user_id == "user123"
        assert memory.agent_id == "agent456"
        assert memory.session_id == "sess789"

    def test_user_exists_agent_does_not(self, mocker):
        client = MemFuseClient(api_key=None)
        mock_request = mocker.patch.object(client, "_request")
        mock_request.side_effect = [
            {"status": "success", "data": {"users": [{"id": "user123"}]}}, # user GET
            {"status": "error", "data": {"agents": []}}, # agent GET
            {"data": {"agent": {"id": "agent456"}}},    # agent POST
            {"data": {"session": {"id": "sess789", "name": "mysession"}}}, # session POST
        ]
        memory = client.create_memory(user="alice", agent="bob", session="mysession")
        assert memory.user_id == "user123"
        assert memory.agent_id == "agent456"
        assert memory.session_id == "sess789"

    def test_user_and_agent_exist(self, mocker):
        client = MemFuseClient(api_key=None)
        mock_request = mocker.patch.object(client, "_request")
        mock_request.side_effect = [
            {"status": "success", "data": {"users": [{"id": "user123"}]}}, # user GET
            {"status": "success", "data": {"agents": [{"id": "agent456"}]}}, # agent GET
            {"data": {"session": {"id": "sess789", "name": "mysession"}}}, # session POST
        ]
        memory = client.create_memory(user="alice", agent="bob", session="mysession")
        assert memory.user_id == "user123"
        assert memory.agent_id == "agent456"
        assert memory.session_id == "sess789"

    def test_agent_defaults_to_default_agent(self, mocker):
        client = MemFuseClient(api_key=None)
        mock_request = mocker.patch.object(client, "_request")
        mock_request.side_effect = [
            {"status": "success", "data": {"users": [{"id": "user123"}]}}, # user GET
            {"status": "error", "data": {"agents": []}}, # agent GET
            {"data": {"agent": {"id": "agent456"}}},    # agent POST
            {"data": {"session": {"id": "sess789", "name": "mysession"}}}, # session POST
        ]
        memory = client.create_memory(user="alice", agent=None, session="mysession")
        assert memory.agent == "default_agent"

    def test_session_argument_used(self, mocker):
        client = MemFuseClient(api_key=None)
        mock_request = mocker.patch.object(client, "_request")
        mock_request.side_effect = [
            {"status": "success", "data": {"users": [{"id": "user123"}]}}, # user GET
            {"status": "success", "data": {"agents": [{"id": "agent456"}]}}, # agent GET
            {"data": {"session": {"id": "sess789", "name": "customsession"}}}, # session POST
        ]
        memory = client.create_memory(user="alice", agent="bob", session="customsession")
        # Check that session name is set from argument if provided
        assert memory.session == "customsession"

    def test_raises_on_request_error(self, mocker):
        client = MemFuseClient(api_key=None)
        mock_request = mocker.patch.object(client, "_request")
        mock_request.side_effect = Exception("fail")
        with pytest.raises(Exception) as exc:
            client.create_memory(user="alice", agent="bob", session="mysession")
        assert "fail" in str(exc.value)

class TestMemFuseClientRepresentation:
    def test_repr_with_api_key(self):
        client = MemFuseClient(api_key="secret")
        rep = repr(client)
        assert rep == "MemFuseClient(api_key=*****)"

    def test_repr_with_no_api_key(self):
        client = MemFuseClient(api_key=None)
        rep = repr(client)
        assert rep == "MemFuseClient(api_key=None)"

    def test_str_delegates_to_repr(self):
        client = MemFuseClient(api_key="secret")
        assert str(client) == repr(client)
