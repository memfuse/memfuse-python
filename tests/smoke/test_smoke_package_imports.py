"""Smoke tests for package imports and public API availability."""

import pytest
import sys
import os


@pytest.mark.smoke
def test_can_import_main_package():
    """Test that the main memfuse package can be imported."""
    try:
        import memfuse
        assert memfuse is not None
        print("✅ Successfully imported memfuse package")
    except ImportError as e:
        pytest.fail(f"Failed to import memfuse package: {e}")


@pytest.mark.smoke
def test_package_version_accessible():
    """Test that the package version is accessible."""
    import memfuse
    
    # Should have a version attribute
    assert hasattr(memfuse, '__version__')
    
    # Version should be a string
    version = memfuse.__version__
    assert isinstance(version, str)
    assert len(version) > 0
    
    print(f"✅ Package version: {version}")


@pytest.mark.smoke
def test_can_import_all_public_classes():
    """Test that all public classes from __all__ can be imported."""
    try:
        from memfuse import (
            AsyncMemFuse,
            MemFuse,
            AsyncMemory,
            Memory,
            HealthApi,
            UsersApi,
            AgentsApi,
            SessionsApi,
            KnowledgeApi,
            MessagesApi,
            ApiKeysApi
        )
        
        # Verify all classes are importable and are actual classes/types
        classes = [
            AsyncMemFuse,
            MemFuse,
            AsyncMemory,
            Memory,
            HealthApi,
            UsersApi,
            AgentsApi,
            SessionsApi,
            KnowledgeApi,
            MessagesApi,
            ApiKeysApi
        ]
        
        for cls in classes:
            assert cls is not None
            assert hasattr(cls, '__name__')
            print(f"✅ Successfully imported {cls.__name__}")
            
    except ImportError as e:
        pytest.fail(f"Failed to import public classes: {e}")


@pytest.mark.smoke
def test_package_all_attribute():
    """Test that the package __all__ attribute is properly defined."""
    import memfuse
    
    # Should have __all__ defined
    assert hasattr(memfuse, '__all__')
    
    all_items = memfuse.__all__
    assert isinstance(all_items, list)
    assert len(all_items) > 0
    
    # Check that all items in __all__ are actually importable
    for item in all_items:
        assert hasattr(memfuse, item), f"Item '{item}' in __all__ is not accessible"
        attr = getattr(memfuse, item)
        assert attr is not None, f"Item '{item}' in __all__ is None"
    
    print(f"✅ Package __all__ contains {len(all_items)} items: {all_items}")


@pytest.mark.smoke
def test_can_import_client_classes():
    """Test that client classes can be imported directly."""
    try:
        from memfuse.client import AsyncMemFuse, MemFuse
        
        assert AsyncMemFuse is not None
        assert MemFuse is not None
        
        print("✅ Successfully imported client classes")
        
    except ImportError as e:
        pytest.fail(f"Failed to import client classes: {e}")


@pytest.mark.smoke
def test_can_import_memory_classes():
    """Test that memory classes can be imported directly."""
    try:
        from memfuse.memory import AsyncMemory, Memory
        
        assert AsyncMemory is not None
        assert Memory is not None
        
        print("✅ Successfully imported memory classes")
        
    except ImportError as e:
        pytest.fail(f"Failed to import memory classes: {e}")


@pytest.mark.smoke
def test_can_import_api_classes():
    """Test that API classes can be imported directly."""
    try:
        from memfuse.api import (
            HealthApi,
            UsersApi,
            AgentsApi,
            SessionsApi,
            KnowledgeApi,
            MessagesApi,
            ApiKeysApi
        )
        
        api_classes = [
            HealthApi,
            UsersApi,
            AgentsApi,
            SessionsApi,
            KnowledgeApi,
            MessagesApi,
            ApiKeysApi
        ]
        
        for cls in api_classes:
            assert cls is not None
            print(f"✅ Successfully imported {cls.__name__}")
            
    except ImportError as e:
        pytest.fail(f"Failed to import API classes: {e}")


@pytest.mark.smoke
def test_can_import_llm_adapters():
    """Test that LLM adapter classes can be imported."""
    try:
        from memfuse.llm import (
            OpenAI,
            AsyncOpenAI,
            Anthropic,
            AsyncAnthropic,
            GeminiClient,
            AsyncGeminiClient,
            MemOllama,
            AsyncMemOllama
        )
        
        llm_classes = [
            OpenAI,
            AsyncOpenAI,
            Anthropic,
            AsyncAnthropic,
            GeminiClient,
            AsyncGeminiClient,
            MemOllama,
            AsyncMemOllama
        ]
        
        for cls in llm_classes:
            assert cls is not None
            print(f"✅ Successfully imported {cls.__name__}")
            
    except ImportError as e:
        pytest.fail(f"Failed to import LLM adapter classes: {e}")


@pytest.mark.smoke  
def test_can_import_utility_functions():
    """Test that utility functions can be imported."""
    try:
        from memfuse.utils import (
            handle_server_connection,
            run_async,
            run_with_error_handling
        )
        
        utils = [
            handle_server_connection,
            run_async,
            run_with_error_handling
        ]
        
        for util in utils:
            assert util is not None
            assert callable(util)
            print(f"✅ Successfully imported {util.__name__}")
            
    except ImportError as e:
        pytest.fail(f"Failed to import utility functions: {e}")


@pytest.mark.smoke
def test_can_import_models():
    """Test that model classes can be imported."""
    try:
        from memfuse.models import (
            Message,
            InitRequest,
            QueryRequest,
            AddRequest,
            ReadRequest,
            UpdateRequest,
            DeleteRequest,
            AddKnowledgeRequest,
            ReadKnowledgeRequest,
            UpdateKnowledgeRequest,
            DeleteKnowledgeRequest,
            ErrorDetail,
            ApiResponse,
        )
        
        model_classes = [
            Message,
            InitRequest,
            QueryRequest,
            AddRequest,
            ReadRequest,
            UpdateRequest,
            DeleteRequest,
            AddKnowledgeRequest,
            ReadKnowledgeRequest,
            UpdateKnowledgeRequest,
            DeleteKnowledgeRequest,
            ErrorDetail,
            ApiResponse,
        ]
        
        for cls in model_classes:
            assert cls is not None
            print(f"✅ Successfully imported {cls.__name__}")
            
    except ImportError as e:
        pytest.fail(f"Failed to import model classes: {e}") 