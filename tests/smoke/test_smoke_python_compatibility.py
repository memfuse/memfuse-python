"""Smoke tests for Python version compatibility."""

import sys
import pytest


@pytest.mark.smoke
def test_python_version_compatibility():
    """Test that we're running on a supported Python version.
    
    The project requires Python 3.10 or higher according to pyproject.toml.
    """
    major, minor = sys.version_info[:2]
    
    # Check minimum Python version
    assert major >= 3, f"Python {major}.{minor} is not supported. Minimum Python 3.10 required."
    
    if major == 3:
        assert minor >= 10, f"Python {major}.{minor} is not supported. Minimum Python 3.10 required."
    
    # Log the Python version for debugging
    print(f"✅ Running on Python {major}.{minor}.{sys.version_info.micro}")


@pytest.mark.smoke
def test_python_version_info_accessible():
    """Test that we can access Python version information."""
    # Basic smoke test to ensure sys.version_info is accessible
    assert hasattr(sys, 'version_info')
    assert len(sys.version_info) >= 2
    assert isinstance(sys.version_info.major, int)
    assert isinstance(sys.version_info.minor, int)


@pytest.mark.smoke
def test_python_platform_info():
    """Test basic platform information access."""
    import platform
    
    # Smoke test - should not raise exceptions
    system = platform.system()
    assert isinstance(system, str)
    assert len(system) > 0
    
    # Should be able to get platform info
    machine = platform.machine()
    assert isinstance(machine, str)
    
    print(f"✅ Running on {system} {machine}")


@pytest.mark.smoke
def test_python_import_mechanism():
    """Test that Python import mechanism works correctly."""
    # Test dynamic import
    import importlib
    
    # Should be able to import built-in modules
    sys_module = importlib.import_module('sys')
    assert sys_module is sys
    
    # Should be able to import from standard library
    json_module = importlib.import_module('json')
    assert hasattr(json_module, 'loads')
    assert hasattr(json_module, 'dumps') 