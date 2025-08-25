"""Smoke tests for basic connectivity structure.

Note: These tests focus on method existence and basic structure only.
No real network calls are made. Error handling and connectivity behavior 
is tested in the error_handling and integration test layers.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio


@pytest.mark.smoke
def test_basic_connectivity_structure():
    """Test that connectivity methods exist and are callable."""
    from memfuse import AsyncMemFuse, MemFuse
    
    async_client = AsyncMemFuse()
    sync_client = MemFuse()
    
    # Check that connectivity methods exist
    assert hasattr(async_client, '_check_server_health')
    assert hasattr(sync_client, '_check_server_health_sync')
    assert callable(async_client._check_server_health)
    assert callable(sync_client._check_server_health_sync)
    
    print("✅ Connectivity methods are available")


@pytest.mark.smoke
def test_request_methods_exist():
    """Test that request methods exist and are callable."""
    from memfuse import AsyncMemFuse, MemFuse
    
    async_client = AsyncMemFuse()
    sync_client = MemFuse()
    
    # Check that request methods exist
    assert hasattr(async_client, '_request')
    assert hasattr(sync_client, '_request_sync')
    assert callable(async_client._request)
    assert callable(sync_client._request_sync)
    
    print("✅ Request methods are available")


@pytest.mark.smoke
def test_session_management_methods():
    """Test that session management methods exist."""
    from memfuse import AsyncMemFuse, MemFuse
    
    async_client = AsyncMemFuse()
    sync_client = MemFuse()
    
    # Check async session management
    assert hasattr(async_client, '_ensure_session')
    assert hasattr(async_client, '_close_session')
    assert callable(async_client._ensure_session)
    assert callable(async_client._close_session)
    
    # Check sync session management
    assert hasattr(sync_client, '_ensure_sync_session')
    assert hasattr(sync_client, '_close_sync_session')
    assert callable(sync_client._ensure_sync_session)
    assert callable(sync_client._close_sync_session)
    
    print("✅ Session management methods are available")


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_async_session_creation_mocked():
    """Test that async session creation and cleanup works with mocked aiohttp."""
    from memfuse import AsyncMemFuse
    
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        
        client = AsyncMemFuse()
        
        # Test session creation
        await client._ensure_session()
        assert client.session is not None
        mock_session_class.assert_called_once()
        
        # Test session cleanup
        await client._close_session()
        assert client.session is None
        mock_session.close.assert_called_once()
        
        print("✅ Async session creation and cleanup works with mocked aiohttp")


@pytest.mark.smoke
def test_sync_session_creation_mocked():
    """Test that sync session creation and cleanup works with mocked requests."""
    from memfuse import MemFuse
    
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = MemFuse()
        
        # Test session creation
        client._ensure_sync_session()
        assert client.sync_session is not None
        mock_session_class.assert_called_once()
        
        # Test session cleanup
        client._close_sync_session()
        assert client.sync_session is None
        mock_session.close.assert_called_once()
        
        print("✅ Sync session creation and cleanup works with mocked requests")


@pytest.mark.smoke
def test_async_health_check_method_exists():
    """Test that async health check method exists and is callable."""
    from memfuse import AsyncMemFuse
    
    client = AsyncMemFuse()
    
    # For smoke tests, we only verify the method exists - no network calls
    assert hasattr(client, '_check_server_health')
    assert callable(client._check_server_health)
    
    # Verify it's an async method by checking if it's a coroutine function
    import asyncio
    assert asyncio.iscoroutinefunction(client._check_server_health)
    
    print("✅ Async health check method structure is correct")





@pytest.mark.smoke
@pytest.mark.asyncio
async def test_async_context_manager_basic():
    """Test basic async context manager functionality."""
    from memfuse import AsyncMemFuse
    
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        
        client = AsyncMemFuse()
        
        # Test context manager entry and exit
        async with client:
            assert client is not None
            # Should not raise any exceptions
            
        print("✅ Async context manager basic functionality works")


@pytest.mark.smoke
def test_sync_context_manager_basic():
    """Test basic sync context manager functionality."""
    from memfuse import MemFuse
    
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = MemFuse()
        
        # Test context manager entry and exit
        with client:
            assert client is not None
            # Should not raise any exceptions
            
        print("✅ Sync context manager basic functionality works")


@pytest.mark.smoke
def test_sync_health_check_method_exists():
    """Test that sync health check method exists and is callable."""
    from memfuse import MemFuse
    
    client = MemFuse()
    
    # For smoke tests, we only verify the method exists - no network calls
    assert hasattr(client, '_check_server_health_sync')
    assert callable(client._check_server_health_sync)
    
    # Verify it's a regular (non-async) method
    import asyncio
    assert not asyncio.iscoroutinefunction(client._check_server_health_sync)
    
    print("✅ Sync health check method structure is correct")


@pytest.mark.smoke
def test_url_construction():
    """Test that URLs are constructed correctly."""
    from memfuse import AsyncMemFuse, MemFuse
    
    # Test with default base URL
    client1 = AsyncMemFuse()
    assert client1.base_url == "http://localhost:8000"
    
    # Test with custom base URL
    client2 = AsyncMemFuse(base_url="https://api.example.com")
    assert client2.base_url == "https://api.example.com"
    
    # Test with trailing slash removal
    client3 = AsyncMemFuse(base_url="http://localhost:8000/")
    assert client3.base_url == "http://localhost:8000"
    
    # Test sync client
    sync_client = MemFuse(base_url="https://api.example.com/")
    assert sync_client.base_url == "https://api.example.com"
    
    print("✅ URL construction works correctly")


@pytest.mark.smoke
def test_api_key_handling():
    """Test that API key handling works correctly."""
    from memfuse import AsyncMemFuse, MemFuse
    
    # Test with explicit API key
    client1 = AsyncMemFuse(api_key="test-key")
    assert client1.api_key == "test-key"
    
    # Test with None API key
    client2 = AsyncMemFuse(api_key=None)
    assert client2.api_key is None
    
    # Test sync client
    sync_client = MemFuse(api_key="sync-test-key")
    assert sync_client.api_key == "sync-test-key"
    
    print("✅ API key handling works correctly")


@pytest.mark.smoke
def test_async_headers_construction():
    """Test that async headers are constructed correctly for authenticated requests."""
    from memfuse import AsyncMemFuse
    
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        
        # Test with API key
        client = AsyncMemFuse(api_key="test-key")
        asyncio.run(client._ensure_session())
        
        # Should create session with Authorization header
        mock_session_class.assert_called_once()
        call_args = mock_session_class.call_args
        headers = call_args[1]['headers']
        assert 'Authorization' in headers
        assert headers['Authorization'] == 'Bearer test-key'
        
        print("✅ Async headers construction works correctly")


@pytest.mark.smoke
def test_sync_headers_construction():
    """Test that sync headers are constructed correctly for authenticated requests."""
    from memfuse import MemFuse
    
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        # Make headers behave like a dictionary
        mock_session.headers = {}
        mock_session_class.return_value = mock_session
        
        # Test with API key
        client = MemFuse(api_key="test-key")
        client._ensure_sync_session()
        
        # Should create session and set Authorization header
        mock_session_class.assert_called_once()
        assert 'Authorization' in mock_session.headers
        assert mock_session.headers['Authorization'] == 'Bearer test-key'
        
        print("✅ Sync headers construction works correctly")


@pytest.mark.smoke
def test_cleanup_methods():
    """Test that cleanup methods are available and callable."""
    from memfuse import AsyncMemFuse, MemFuse
    
    async_client = AsyncMemFuse()
    sync_client = MemFuse()
    
    # Test that cleanup methods exist
    assert hasattr(async_client, 'close')
    assert hasattr(sync_client, 'close')
    assert callable(async_client.close)
    assert callable(sync_client.close)
    
    print("✅ Cleanup methods are available")


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_async_client_full_cleanup():
    """Test that async client full cleanup works properly."""
    from memfuse import AsyncMemFuse
    
    initial_instance_count = len(AsyncMemFuse._instances)
    
    client = AsyncMemFuse()
    
    # Client should be tracked
    assert len(AsyncMemFuse._instances) == initial_instance_count + 1
    assert client in AsyncMemFuse._instances
    
    # Close should clean up everything
    await client.close()
    
    # Client should be removed from tracking
    assert len(AsyncMemFuse._instances) == initial_instance_count
    assert client not in AsyncMemFuse._instances
    
    print("✅ Async client full cleanup works properly")


@pytest.mark.smoke
def test_sync_client_full_cleanup():
    """Test that sync client full cleanup works properly."""
    from memfuse import MemFuse
    
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        client = MemFuse()
        
        # Create a session
        client._ensure_sync_session()
        assert client.sync_session is not None
        
        # Close should clean up the session
        client.close()
        assert client.sync_session is None
        mock_session.close.assert_called_once()
        
        print("✅ Sync client full cleanup works properly") 