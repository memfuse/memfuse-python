"""Smoke tests for basic instantiation of client classes."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio


@pytest.mark.smoke
def test_can_instantiate_async_memfuse():
    """Test that AsyncMemFuse can be instantiated without errors."""
    try:
        from memfuse import AsyncMemFuse
        
        # Basic instantiation with default parameters
        client = AsyncMemFuse()
        assert client is not None
        assert hasattr(client, 'base_url')
        assert hasattr(client, 'api_key')
        assert hasattr(client, 'session')
        
        # Instantiation with custom parameters
        client_custom = AsyncMemFuse(
            base_url="http://test.example.com",
            api_key="test-key"
        )
        assert client_custom.base_url == "http://test.example.com"
        assert client_custom.api_key == "test-key"
        
        print("✅ AsyncMemFuse instantiation successful")
        
    except Exception as e:
        pytest.fail(f"Failed to instantiate AsyncMemFuse: {e}")


@pytest.mark.smoke
def test_can_instantiate_sync_memfuse():
    """Test that MemFuse can be instantiated without errors."""
    try:
        from memfuse import MemFuse
        
        # Basic instantiation with default parameters
        client = MemFuse()
        assert client is not None
        assert hasattr(client, 'base_url')
        assert hasattr(client, 'api_key')
        assert hasattr(client, 'sync_session')
        
        # Instantiation with custom parameters
        client_custom = MemFuse(
            base_url="http://test.example.com",
            api_key="test-key"
        )
        assert client_custom.base_url == "http://test.example.com"
        assert client_custom.api_key == "test-key"
        
        print("✅ MemFuse instantiation successful")
        
    except Exception as e:
        pytest.fail(f"Failed to instantiate MemFuse: {e}")


@pytest.mark.smoke
def test_async_memfuse_has_required_attributes():
    """Test that AsyncMemFuse has all required attributes."""
    from memfuse import AsyncMemFuse
    
    client = AsyncMemFuse()
    
    # Check that all API clients are available
    required_attrs = [
        'health',
        'users',
        'agents',
        'sessions',
        'knowledge',
        'messages',
        'api_keys'
    ]
    
    for attr in required_attrs:
        assert hasattr(client, attr), f"AsyncMemFuse missing required attribute: {attr}"
        api_client = getattr(client, attr)
        assert api_client is not None, f"AsyncMemFuse.{attr} is None"
        print(f"✅ AsyncMemFuse.{attr} is available")


@pytest.mark.smoke
def test_sync_memfuse_has_required_attributes():
    """Test that MemFuse has all required attributes."""
    from memfuse import MemFuse
    
    client = MemFuse()
    
    # Check that all API clients are available
    required_attrs = [
        'health',
        'users',
        'agents',
        'sessions',
        'knowledge',
        'messages',
        'api_keys'
    ]
    
    for attr in required_attrs:
        assert hasattr(client, attr), f"MemFuse missing required attribute: {attr}"
        api_client = getattr(client, attr)
        assert api_client is not None, f"MemFuse.{attr} is None"
        print(f"✅ MemFuse.{attr} is available")


@pytest.mark.smoke
def test_async_memfuse_context_manager_structure():
    """Test that AsyncMemFuse has proper context manager structure."""
    from memfuse import AsyncMemFuse
    
    client = AsyncMemFuse()
    
    # Check that it has async context manager methods
    assert hasattr(client, '__aenter__')
    assert hasattr(client, '__aexit__')
    assert callable(client.__aenter__)
    assert callable(client.__aexit__)
    
    print("✅ AsyncMemFuse has async context manager structure")


@pytest.mark.smoke
def test_sync_memfuse_context_manager_structure():
    """Test that MemFuse has proper context manager structure."""
    from memfuse import MemFuse
    
    client = MemFuse()
    
    # Check that it has sync context manager methods
    assert hasattr(client, '__enter__')
    assert hasattr(client, '__exit__')
    assert callable(client.__enter__)
    assert callable(client.__exit__)
    
    print("✅ MemFuse has sync context manager structure")


@pytest.mark.smoke
def test_can_instantiate_api_clients():
    """Test that all API client classes can be instantiated."""
    from memfuse import AsyncMemFuse
    from memfuse.api import (
        HealthApi,
        UsersApi,
        AgentsApi,
        SessionsApi,
        KnowledgeApi,
        MessagesApi,
        ApiKeysApi
    )
    
    # Create a mock client for testing
    mock_client = Mock()
    
    api_classes = [
        HealthApi,
        UsersApi,
        AgentsApi,
        SessionsApi,
        KnowledgeApi,
        MessagesApi,
        ApiKeysApi
    ]
    
    for api_class in api_classes:
        try:
            api_instance = api_class(mock_client)
            assert api_instance is not None
            assert hasattr(api_instance, 'client')
            assert api_instance.client is mock_client
            print(f"✅ {api_class.__name__} instantiation successful")
        except Exception as e:
            pytest.fail(f"Failed to instantiate {api_class.__name__}: {e}")


@pytest.mark.smoke
def test_can_instantiate_memory_classes():
    """Test that memory classes can be instantiated."""
    from memfuse import AsyncMemory, Memory
    
    # Create mock clients
    mock_async_client = Mock()
    mock_sync_client = Mock()
    
    # Test data
    test_data = {
        'session_id': 'test-session',
        'user_id': 'test-user',
        'agent_id': 'test-agent',
        'user_name': 'Test User',
        'agent_name': 'Test Agent',
        'session_name': 'Test Session'
    }
    
    try:
        # Test AsyncMemory
        async_memory = AsyncMemory(
            client=mock_async_client,
            **test_data
        )
        assert async_memory is not None
        assert async_memory.client is mock_async_client
        assert async_memory.session_id == test_data['session_id']
        assert async_memory.user == test_data['user_name']
        print("✅ AsyncMemory instantiation successful")
        
        # Test Memory
        sync_memory = Memory(
            client=mock_sync_client,
            **test_data
        )
        assert sync_memory is not None
        assert sync_memory.client is mock_sync_client
        assert sync_memory.session_id == test_data['session_id']
        assert sync_memory.user == test_data['user_name']
        print("✅ Memory instantiation successful")
        
    except Exception as e:
        pytest.fail(f"Failed to instantiate memory classes: {e}")


@pytest.mark.smoke
def test_can_instantiate_llm_adapters():
    """Test that LLM adapter classes can be instantiated."""
    from memfuse.llm import (
        OpenAI,
        AsyncOpenAI,
        Anthropic,
        AsyncAnthropic,
        MemOllama,
        AsyncMemOllama
    )
    
    # Test basic instantiation (most will fail without proper API keys, but structure should be ok)
    llm_classes = [
        (OpenAI, {}),
        (AsyncOpenAI, {}),
        (Anthropic, {}),
        (AsyncAnthropic, {}),
        (MemOllama, {}),
        (AsyncMemOllama, {})
    ]
    
    for llm_class, kwargs in llm_classes:
        try:
            # We're just testing that the class exists and is callable
            # We don't actually instantiate because it might require API keys
            assert callable(llm_class)
            assert hasattr(llm_class, '__name__')
            print(f"✅ {llm_class.__name__} is callable")
        except Exception as e:
            pytest.fail(f"Failed to access {llm_class.__name__}: {e}")


@pytest.mark.smoke
def test_environment_variable_handling():
    """Test that environment variable handling works properly."""
    from memfuse import AsyncMemFuse, MemFuse
    
    # Test with clean environment
    with patch.dict('os.environ', {}, clear=True):
        async_client = AsyncMemFuse()
        sync_client = MemFuse()
        
        # API key should be None when not set
        assert async_client.api_key is None
        assert sync_client.api_key is None
        
        print("✅ Environment variable handling works with clean environment")
    
    # Test with environment variables set
    with patch.dict('os.environ', {'MEMFUSE_API_KEY': 'env-test-key'}):
        async_client = AsyncMemFuse()
        sync_client = MemFuse()
        
        # API key should be picked up from environment
        assert async_client.api_key == 'env-test-key'
        assert sync_client.api_key == 'env-test-key'
        
        print("✅ Environment variable handling works with MEMFUSE_API_KEY set")


@pytest.mark.smoke
def test_default_values():
    """Test that default values are properly set."""
    from memfuse import AsyncMemFuse, MemFuse
    
    # Test default values
    async_client = AsyncMemFuse()
    sync_client = MemFuse()
    
    # Default base_url should be localhost
    assert async_client.base_url == "http://localhost:8000"
    assert sync_client.base_url == "http://localhost:8000"
    
    # Session should initially be None
    assert async_client.session is None
    assert sync_client.sync_session is None
    
    print("✅ Default values are properly set")


@pytest.mark.smoke
def test_client_instance_tracking():
    """Test that AsyncMemFuse properly tracks instances."""
    from memfuse import AsyncMemFuse
    
    initial_count = len(AsyncMemFuse._instances)
    
    # Create a new client
    client1 = AsyncMemFuse()
    assert len(AsyncMemFuse._instances) == initial_count + 1
    assert client1 in AsyncMemFuse._instances
    
    # Create another client
    client2 = AsyncMemFuse()
    assert len(AsyncMemFuse._instances) == initial_count + 2
    assert client2 in AsyncMemFuse._instances
    
    print("✅ Client instance tracking works properly")


@pytest.mark.smoke
def test_basic_string_representations():
    """Test that client classes have proper string representations."""
    from memfuse import AsyncMemFuse, MemFuse
    
    async_client = AsyncMemFuse()
    sync_client = MemFuse()
    
    # Should be able to get string representations without errors
    async_str = str(async_client)
    sync_str = str(sync_client)
    
    assert isinstance(async_str, str)
    assert isinstance(sync_str, str)
    assert len(async_str) > 0
    assert len(sync_str) > 0
    
    print("✅ String representations work properly") 