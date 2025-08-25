"""Smoke tests for required dependencies availability."""

import pytest
import sys
import importlib


@pytest.mark.smoke
def test_can_import_aiohttp():
    """Test that aiohttp is available for async HTTP operations."""
    try:
        import aiohttp
        assert aiohttp is not None
        
        # Check that we can access key classes
        assert hasattr(aiohttp, 'ClientSession')
        assert hasattr(aiohttp, 'ClientConnectorError')
        
        print(f"✅ aiohttp {aiohttp.__version__} is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import aiohttp: {e}")


@pytest.mark.smoke
def test_can_import_requests():
    """Test that requests is available for sync HTTP operations."""
    try:
        import requests
        assert requests is not None
        
        # Check that we can access key classes
        assert hasattr(requests, 'Session')
        assert hasattr(requests, 'RequestException')
        
        print(f"✅ requests {requests.__version__} is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import requests: {e}")


@pytest.mark.smoke
def test_can_import_pydantic():
    """Test that pydantic is available for data validation."""
    try:
        import pydantic
        assert pydantic is not None
        
        # Check that we can access key classes
        assert hasattr(pydantic, 'BaseModel')
        assert hasattr(pydantic, 'ValidationError')
        
        print(f"✅ pydantic {pydantic.__version__} is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import pydantic: {e}")


@pytest.mark.smoke
def test_can_import_openai():
    """Test that openai is available for OpenAI LLM integration."""
    try:
        import openai
        assert openai is not None
        
        # Check that we can access key classes
        assert hasattr(openai, 'OpenAI')
        assert hasattr(openai, 'AsyncOpenAI')
        
        print(f"✅ openai {openai.__version__} is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import openai: {e}")


@pytest.mark.smoke
def test_can_import_anthropic():
    """Test that anthropic is available for Anthropic LLM integration."""
    try:
        import anthropic
        assert anthropic is not None
        
        # Check that we can access key classes
        assert hasattr(anthropic, 'Anthropic')
        assert hasattr(anthropic, 'AsyncAnthropic')
        
        print(f"✅ anthropic {anthropic.__version__} is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import anthropic: {e}")


@pytest.mark.smoke
def test_can_import_google_genai():
    """Test that google-genai is available for Google Gemini integration."""
    try:
        import google.genai
        assert google.genai is not None
        
        # Check that we can access key classes
        assert hasattr(google.genai, 'Client')
        
        print("✅ google-genai is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import google-genai: {e}")


@pytest.mark.smoke
def test_can_import_ollama():
    """Test that ollama is available for Ollama LLM integration."""
    try:
        import ollama
        assert ollama is not None
        
        # Check that we can access key classes
        assert hasattr(ollama, 'Client')
        
        # Try to get version, but don't fail if it doesn't exist
        version = getattr(ollama, '__version__', 'unknown')
        print(f"✅ ollama {version} is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import ollama: {e}")


@pytest.mark.smoke
def test_can_import_jinja2():
    """Test that jinja2 is available for template rendering."""
    try:
        import jinja2
        assert jinja2 is not None
        
        # Check that we can access key classes
        assert hasattr(jinja2, 'Environment')
        assert hasattr(jinja2, 'Template')
        
        print(f"✅ jinja2 {jinja2.__version__} is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import jinja2: {e}")


@pytest.mark.smoke
def test_can_import_httpx():
    """Test that httpx is available as an HTTP client option."""
    try:
        import httpx
        assert httpx is not None
        
        # Check that we can access key classes
        assert hasattr(httpx, 'Client')
        assert hasattr(httpx, 'AsyncClient')
        
        print(f"✅ httpx {httpx.__version__} is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import httpx: {e}")


@pytest.mark.smoke
def test_can_import_python_frontmatter():
    """Test that python-frontmatter is available for frontmatter parsing."""
    try:
        import frontmatter
        assert frontmatter is not None
        
        # Check that we can access key functions
        assert hasattr(frontmatter, 'loads')
        assert hasattr(frontmatter, 'dumps')
        
        print("✅ python-frontmatter is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import python-frontmatter: {e}")


@pytest.mark.smoke
def test_can_import_python_dotenv():
    """Test that python-dotenv is available for environment variable loading."""
    try:
        import dotenv
        assert dotenv is not None
        
        # Check that we can access key functions
        assert hasattr(dotenv, 'load_dotenv')
        assert hasattr(dotenv, 'find_dotenv')
        
        print("✅ python-dotenv is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import python-dotenv: {e}")


@pytest.mark.smoke
def test_can_import_gradio():
    """Test that gradio is available for UI components."""
    try:
        import gradio
        assert gradio is not None
        
        # Check that we can access key classes
        assert hasattr(gradio, 'Interface')
        assert hasattr(gradio, 'Blocks')
        
        print(f"✅ gradio {gradio.__version__} is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import gradio: {e}")


@pytest.mark.smoke
def test_standard_library_dependencies():
    """Test that required standard library modules are available."""
    standard_modules = [
        'os',
        'sys',
        'json',
        'asyncio',
        'typing',
        'pathlib',
        'collections',
        'functools',
        'importlib',
        'datetime',
        'uuid',
        'urllib',
        'logging'
    ]
    
    for module_name in standard_modules:
        try:
            module = importlib.import_module(module_name)
            assert module is not None
            print(f"✅ {module_name} is available")
        except ImportError as e:
            pytest.fail(f"Failed to import standard library module {module_name}: {e}")


@pytest.mark.smoke
def test_pytest_available():
    """Test that pytest is available for testing."""
    try:
        import pytest
        assert pytest is not None
        
        # Check that we can access key functions
        assert hasattr(pytest, 'mark')
        assert hasattr(pytest, 'fixture')
        assert hasattr(pytest, 'raises')
        
        print(f"✅ pytest {pytest.__version__} is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import pytest: {e}")


@pytest.mark.smoke
def test_unittest_mock_available():
    """Test that unittest.mock is available for mocking."""
    try:
        from unittest.mock import Mock, AsyncMock, patch
        
        assert Mock is not None
        assert AsyncMock is not None
        assert patch is not None
        
        print("✅ unittest.mock is available")
        
    except ImportError as e:
        pytest.fail(f"Failed to import unittest.mock: {e}")


@pytest.mark.smoke
def test_dependency_versions_accessible():
    """Test that we can access version information from dependencies."""
    dependencies_with_versions = [
        'aiohttp',
        'requests', 
        'pydantic',
        'openai',
        'anthropic',
        'ollama',
        'jinja2',
        'httpx',
        'gradio',
        'pytest'
    ]
    
    for dep in dependencies_with_versions:
        try:
            module = importlib.import_module(dep)
            
            # Try to get version information
            version = None
            for attr in ['__version__', 'version', 'VERSION']:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    break
                    
            if version:
                print(f"✅ {dep} version: {version}")
            else:
                print(f"⚠️  {dep} version not accessible")
                
        except ImportError:
            # This test should not fail if import fails - that's covered by other tests
            pass 