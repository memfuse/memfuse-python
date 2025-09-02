"""Unit tests for version compatibility utilities."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from memfuse.utils.version_compatibility import (
    parse_semantic_version,
    compare_versions,
    extract_version_from_health_response,
    check_version_compatibility
)


class TestVersionParsing:
    """Test semantic version parsing functionality."""

    def test_parse_valid_semantic_versions(self):
        """Test parsing of valid semantic version strings."""
        # Standard semantic versions
        assert parse_semantic_version("1.2.3") == (1, 2, 3)
        assert parse_semantic_version("0.1.0") == (0, 1, 0)
        assert parse_semantic_version("10.20.30") == (10, 20, 30)
        
        # Versions with 'v' prefix
        assert parse_semantic_version("v1.2.3") == (1, 2, 3)
        assert parse_semantic_version("v0.1.0") == (0, 1, 0)
        
        # Versions with pre-release/build metadata
        assert parse_semantic_version("1.2.3-alpha") == (1, 2, 3)
        assert parse_semantic_version("1.2.3+build.1") == (1, 2, 3)
        assert parse_semantic_version("1.2.3-alpha+build.1") == (1, 2, 3)
        
        # Python development versions (PEP 440 style)
        assert parse_semantic_version("0.3.0.post16.dev0+899f75a") == (0, 3, 0)
        assert parse_semantic_version("1.0.0.dev123") == (1, 0, 0)
        assert parse_semantic_version("2.1.0.post1") == (2, 1, 0)
        
        # Complex versions that should extract base version
        assert parse_semantic_version("1.2.3.4") == (1, 2, 3)  # Treats .4 as suffix

    def test_parse_invalid_versions(self):
        """Test parsing of invalid version strings."""
        # Empty or None
        assert parse_semantic_version("") is None
        assert parse_semantic_version(None) is None
        
        # Invalid formats (not enough version components)
        assert parse_semantic_version("1.2") is None
        assert parse_semantic_version("v1.2") is None
        assert parse_semantic_version("abc.def.ghi") is None
        assert parse_semantic_version("1.2.x") is None


class TestVersionComparison:
    """Test version comparison logic."""

    def test_compatible_versions(self):
        """Test that versions with same minor version are compatible."""
        # Same minor version should be compatible
        assert compare_versions("0.1.0", "0.1.5") is None
        assert compare_versions("0.2.1", "0.2.0") is None
        assert compare_versions("1.0.0", "1.0.10") is None
        assert compare_versions("2.5.3", "2.5.1") is None

    def test_sdk_behind_server(self):
        """Test warning when SDK minor version is behind server."""
        warning = compare_versions("0.1.5", "0.2.1")
        assert warning is not None
        assert "SDK version: 0.1.5" in warning
        assert "Server version: 0.2.1" in warning
        assert "Please upgrade your SDK" in warning
        assert "pip install --upgrade memfuse" in warning
        assert "github.com/memfuse/memfuse-python" in warning
        assert "github.com/memfuse/memfuse" in warning

    def test_server_behind_sdk(self):
        """Test warning when server minor version is behind SDK."""
        warning = compare_versions("0.3.1", "0.2.5")
        assert warning is not None
        assert "SDK version: 0.3.1" in warning
        assert "Server version: 0.2.5" in warning
        assert "Please upgrade your server" in warning
        assert "server version (0.2.5) is behind the SDK version (0.3.1)" in warning
        assert "github.com/memfuse/memfuse-python" in warning
        assert "github.com/memfuse/memfuse" in warning

    def test_invalid_version_strings(self):
        """Test that invalid version strings return None (no warning)."""
        assert compare_versions("invalid", "0.1.0") is None
        assert compare_versions("0.1.0", "invalid") is None
        assert compare_versions("", "0.1.0") is None
        assert compare_versions("0.1.0", "") is None


class TestHealthResponseParsing:
    """Test extraction of version info from health responses."""

    def test_extract_version_from_standard_response(self):
        """Test extracting version from standard MemFuse health response."""
        health_data = {
            "status": "success",
            "code": 200,
            "data": {
                "status": "ok",
                "version": "0.3.1",
                "python_version": "3.11.9"
            }
        }
        assert extract_version_from_health_response(health_data) == "0.3.1"

    def test_extract_version_from_top_level(self):
        """Test extracting version from top-level fields."""
        health_data = {"version": "1.2.3", "status": "ok"}
        assert extract_version_from_health_response(health_data) == "1.2.3"
        
        health_data = {"server_version": "2.0.1"}
        assert extract_version_from_health_response(health_data) == "2.0.1"

    def test_extract_version_with_whitespace(self):
        """Test that version extraction handles whitespace."""
        health_data = {"data": {"version": "  0.3.1  "}}
        assert extract_version_from_health_response(health_data) == "0.3.1"

    def test_no_version_found(self):
        """Test handling when no version is found."""
        # Empty response
        assert extract_version_from_health_response({}) is None
        
        # No version field
        health_data = {"status": "ok", "data": {"status": "healthy"}}
        assert extract_version_from_health_response(health_data) is None
        
        # Invalid data type
        assert extract_version_from_health_response("invalid") is None
        assert extract_version_from_health_response(None) is None

    def test_extract_from_nested_structures(self):
        """Test extracting version from various nested structures."""
        # Note: Current implementation only checks specific nested keys
        # Test the ones that are actually implemented
        health_data = {"data": {"version": "1.0.0"}}
        assert extract_version_from_health_response(health_data) == "1.0.0"


class TestVersionCompatibilityIntegration:
    """Test the main version compatibility check function."""

    def test_check_compatibility_success(self):
        """Test successful compatibility check."""
        health_data = {
            "status": "success",
            "data": {"version": "0.2.1"}
        }
        
        # Compatible versions
        warning = check_version_compatibility("0.2.0", health_data)
        assert warning is None
        
        # Incompatible versions
        warning = check_version_compatibility("0.1.5", health_data)
        assert warning is not None
        assert "SDK version: 0.1.5" in warning
        assert "Server version: 0.2.1" in warning

    def test_check_compatibility_missing_data(self):
        """Test handling of missing or invalid data."""
        # Missing SDK version
        assert check_version_compatibility("", {"data": {"version": "0.1.0"}}) is None
        assert check_version_compatibility(None, {"data": {"version": "0.1.0"}}) is None
        
        # Missing health data
        assert check_version_compatibility("0.1.0", {}) is None
        assert check_version_compatibility("0.1.0", None) is None
        
        # No version in health data
        health_data = {"status": "success", "data": {"status": "ok"}}
        assert check_version_compatibility("0.1.0", health_data) is None


class TestClientVersionCheckIntegration:
    """Test version checking integration with MemFuse clients."""

    @pytest.mark.asyncio
    async def test_async_client_version_check_method_exists(self):
        """Test that async client has version checking method."""
        from memfuse import AsyncMemFuse
        
        client = AsyncMemFuse()
        assert hasattr(client, '_check_version_compatibility')
        assert callable(client._check_version_compatibility)
        
        # Should be async
        import asyncio
        assert asyncio.iscoroutinefunction(client._check_version_compatibility)

    def test_sync_client_version_check_method_exists(self):
        """Test that sync client has version checking method."""
        from memfuse import MemFuse
        
        client = MemFuse()
        assert hasattr(client, '_check_version_compatibility_sync')
        assert callable(client._check_version_compatibility_sync)
        
        # Should be sync
        import asyncio
        assert not asyncio.iscoroutinefunction(client._check_version_compatibility_sync)

    @pytest.mark.asyncio
    async def test_async_version_check_with_mocked_health(self):
        """Test async version checking behavior."""
        from memfuse import AsyncMemFuse
        
        client = AsyncMemFuse()
        
        # Test that method executes without errors when mocked
        with patch.object(client.health, 'health_check', return_value={"data": {"version": "0.3.1"}}):
            # Should execute without raising exceptions
            await client._check_version_compatibility()

    def test_sync_version_check_with_mocked_health(self):
        """Test sync version checking behavior."""
        from memfuse import MemFuse
        
        client = MemFuse()
        
        # Test that method executes without errors when mocked
        with patch.object(client.health, 'health_check_sync', return_value={"data": {"version": "0.2.1"}}):
            # Should execute without raising exceptions
            client._check_version_compatibility_sync()

    @pytest.mark.asyncio
    async def test_async_version_check_handles_errors(self):
        """Test that async version check handles errors gracefully."""
        from memfuse import AsyncMemFuse
        
        client = AsyncMemFuse()
        
        # Mock health check to raise an exception
        with patch.object(client.health, 'health_check', side_effect=Exception("Health check failed")):
            # Should not raise an exception
            try:
                await client._check_version_compatibility()
            except Exception:
                pytest.fail("Version check should handle errors gracefully")

    def test_sync_version_check_handles_errors(self):
        """Test that sync version check handles errors gracefully."""
        from memfuse import MemFuse
        
        client = MemFuse()
        
        # Mock health check to raise an exception
        with patch.object(client.health, 'health_check_sync', side_effect=Exception("Health check failed")):
            # Should not raise an exception
            try:
                client._check_version_compatibility_sync()
            except Exception:
                pytest.fail("Version check should handle errors gracefully")

    def test_version_check_functionality_basic(self):
        """Test basic version checking functionality without complex mocking."""
        # This test focuses on the core logic without the complex import mocking
        from memfuse.utils.version_compatibility import check_version_compatibility
        
        # Test cases that should trigger warnings
        health_data = {"data": {"version": "0.3.1"}}
        
        # SDK behind case
        warning = check_version_compatibility("0.2.0", health_data)
        assert warning is not None
        assert "SDK version: 0.2.0" in warning
        assert "Server version: 0.3.1" in warning
        
        # Compatible case
        warning = check_version_compatibility("0.3.0", health_data)
        assert warning is None