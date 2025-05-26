import pytest
from memfuse import Memory

class TestMemoryInitialization:
    def test_init_sets_all_attributes(self):
        mock_client = object()
        memory = Memory(
            client=mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
            get_last_n_messages=42,
        )
        assert memory.client is mock_client
        assert memory.user_id == "user1"
        assert memory.agent_id == "agent1"
        assert memory.session_id == "sess1"
        assert memory.user == "alice"
        assert memory.agent == "bob"
        assert memory.session == "mysession"
        assert memory.get_last_n_messages == 42 

class TestMemoryQuery:
    def test_query_calls_client_with_correct_args(self, mocker):
        mock_client = mocker.Mock()
        memory = Memory(
            client=mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        mock_client._request.return_value = {"result": "ok"}
        result = memory.query(
            query="test query",
            top_k=7,
            store_type="foo",
            include_messages=False,
            include_knowledge=True,
        )
        mock_client._request.assert_called_once_with(
            "post",
            "/api/v1/users/user1/query",
            {
                "query": "test query",
                "top_k": 7,
                "store_type": "foo",
                "include_messages": False,
                "include_knowledge": True,
                "session_id": "sess1",
            },
        )
        assert result == {"result": "ok"}

    def test_query_defaults(self, mocker):
        mock_client = mocker.Mock()
        memory = Memory(
            client=mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        mock_client._request.return_value = {"result": "ok"}
        memory.query(query="foo")
        args, kwargs = mock_client._request.call_args
        assert args[0] == "post"
        assert args[1] == "/api/v1/users/user1/query"
        payload = args[2]
        assert payload["query"] == "foo"
        assert payload["top_k"] == 5
        assert payload["include_messages"] is True
        assert payload["include_knowledge"] is True
        assert payload["session_id"] == "sess1" 

class TestMemorySessionMessages:
    def test_get_session_messages_with_argument(self, mocker):
        mock_client = mocker.Mock()
        memory = Memory(
            client=mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
            get_last_n_messages=3,
        )
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
        mock_client._request.return_value = {"data": {"messages": messages}}
        result = memory.get_session_messages(get_last_n_messages=2)
        mock_client._request.assert_called_once_with(
            "get",
            "/api/v1/sessions/sess1/messages",
        )
        assert result == messages[-2:]

    def test_get_session_messages_default(self, mocker):
        mock_client = mocker.Mock()
        memory = Memory(
            client=mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
            get_last_n_messages=3,
        )
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
        mock_client._request.return_value = {"data": {"messages": messages}}
        result = memory.get_session_messages()
        assert result == messages[-3:] 

class TestMemoryMessageCRUD:
    def setup_method(self):
        self.mock_client = mocker = None  # Will be set in each test
        self.memory = None

    def test_add_calls_client_and_returns(self, mocker):
        self.mock_client = mocker.Mock()
        self.memory = Memory(
            client=self.mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        self.mock_client._request.return_value = {"data": "added"}
        messages = [{"role": "user", "content": "hi"}]
        result = self.memory.add(messages)
        self.mock_client._request.assert_called_once_with(
            "post",
            "/api/v1/sessions/sess1/messages",
            {"messages": messages},
        )
        assert result == {"data": "added"}

    def test_read_calls_client_and_returns(self, mocker):
        self.mock_client = mocker.Mock()
        self.memory = Memory(
            client=self.mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        self.mock_client._request.return_value = {"data": "read"}
        message_ids = ["m1", "m2"]
        result = self.memory.read(message_ids)
        self.mock_client._request.assert_called_once_with(
            "post",
            "/api/v1/sessions/sess1/messages/read",
            {"message_ids": message_ids},
        )
        assert result == {"data": "read"}

    def test_update_calls_client_and_returns(self, mocker):
        self.mock_client = mocker.Mock()
        self.memory = Memory(
            client=self.mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        self.mock_client._request.return_value = {"data": "updated"}
        message_ids = ["m1", "m2"]
        new_messages = [{"role": "user", "content": "new"}]
        result = self.memory.update(message_ids, new_messages)
        self.mock_client._request.assert_called_once_with(
            "put",
            "/api/v1/sessions/sess1/messages",
            {"message_ids": message_ids, "new_messages": new_messages},
        )
        assert result == {"data": "updated"}

    def test_delete_calls_client_and_returns(self, mocker):
        self.mock_client = mocker.Mock()
        self.memory = Memory(
            client=self.mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        self.mock_client._request.return_value = {"data": "deleted"}
        message_ids = ["m1", "m2"]
        result = self.memory.delete(message_ids)
        self.mock_client._request.assert_called_once_with(
            "delete",
            "/api/v1/sessions/sess1/messages",
            {"message_ids": message_ids},
        )
        assert result == {"data": "deleted"} 

class TestMemoryKnowledgeCRUD:
    def test_add_knowledge_calls_client_and_returns(self, mocker):
        mock_client = mocker.Mock()
        memory = Memory(
            client=mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        mock_client._request.return_value = {"data": "added_knowledge"}
        knowledge = ["fact1", "fact2"]
        result = memory.add_knowledge(knowledge)
        mock_client._request.assert_called_once_with(
            "post",
            "/api/v1/users/user1/knowledge",
            {"knowledge": knowledge},
        )
        assert result == {"data": "added_knowledge"}

    def test_read_knowledge_calls_client_and_returns(self, mocker):
        mock_client = mocker.Mock()
        memory = Memory(
            client=mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        mock_client._request.return_value = {"data": "read_knowledge"}
        knowledge_ids = ["k1", "k2"]
        result = memory.read_knowledge(knowledge_ids)
        mock_client._request.assert_called_once_with(
            "post",
            "/api/v1/users/user1/knowledge/read",
            {"knowledge_ids": knowledge_ids},
        )
        assert result == {"data": "read_knowledge"}

    def test_update_knowledge_calls_client_and_returns(self, mocker):
        mock_client = mocker.Mock()
        memory = Memory(
            client=mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        mock_client._request.return_value = {"data": "updated_knowledge"}
        knowledge_ids = ["k1", "k2"]
        new_knowledge = ["new1", "new2"]
        result = memory.update_knowledge(knowledge_ids, new_knowledge)
        mock_client._request.assert_called_once_with(
            "put",
            "/api/v1/users/user1/knowledge",
            {"knowledge_ids": knowledge_ids, "new_knowledge": new_knowledge},
        )
        assert result == {"data": "updated_knowledge"}

    def test_delete_knowledge_calls_client_and_returns(self, mocker):
        mock_client = mocker.Mock()
        memory = Memory(
            client=mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        mock_client._request.return_value = {"data": "deleted_knowledge"}
        knowledge_ids = ["k1", "k2"]
        result = memory.delete_knowledge(knowledge_ids)
        mock_client._request.assert_called_once_with(
            "delete",
            "/api/v1/users/user1/knowledge",
            {"knowledge_ids": knowledge_ids},
        )
        assert result == {"data": "deleted_knowledge"} 

class TestMemoryContextManager:
    def test_close_calls_client_close(self, mocker):
        mock_client = mocker.Mock()
        memory = Memory(
            client=mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        memory.close()
        mock_client.close.assert_called_once()

    def test_context_manager_calls_close(self, mocker):
        mock_client = mocker.Mock()
        memory = Memory(
            client=mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        with memory:
            pass
        mock_client.close.assert_called_once()

    def test_context_manager_calls_close_on_exception(self, mocker):
        mock_client = mocker.Mock()
        memory = Memory(
            client=mock_client,
            user_id="user1",
            agent_id="agent1",
            session_id="sess1",
            user="alice",
            agent="bob",
            session="mysession",
        )
        try:
            with memory:
                raise ValueError("fail")
        except ValueError:
            pass
        mock_client.close.assert_called_once() 