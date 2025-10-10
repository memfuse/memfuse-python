"""Version compatibility utilities for MemFuse SDK."""

import re
from loguru import logger
from typing import Optional, Tuple, Dict, Any

# GitHub URLs
SDK_GITHUB_URL = "https://github.com/memfuse/memfuse-python"
SERVER_GITHUB_URL = "https://github.com/memfuse/memfuse"


def parse_semantic_version(version_string: str) -> Optional[Tuple[int, int, int]]:
    """Parse a semantic version string into (major, minor, patch) tuple.
    
    Args:
        version_string: Version string like "1.2.3" or "v1.2.3" or "1.2.3.post16.dev0+hash"
        
    Returns:
        Tuple of (major, minor, patch) integers, or None if parsing fails
    """
    if not version_string:
        return None
    
    # Remove 'v' prefix if present
    version_string = version_string.lstrip('v')
    
    # Match the first three numeric components and ignore any trailing identifiers
    # Handles: 1.2.3, 1.2.3-alpha, 1.2.3+build.1, 1.2.3.post16.dev0+hash, 1.2.3rc1, etc.
    # Examples like "0.3.23" and "0.3.23rc1" both parse as (0, 3, 23).
    pattern = r'^(\d+)\.(\d+)\.(\d+)'
    match = re.match(pattern, version_string)
    
    if not match:
        logger.debug(f"Failed to parse version string: {version_string}")
        return None
    
    try:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    except ValueError:
        logger.debug(f"Failed to convert version components to integers: {version_string}")
        return None


def compare_versions(sdk_version: str, server_version: str) -> Optional[str]:
    """Compare SDK and server versions and return recommendation if needed.
    
    Args:
        sdk_version: SDK version string
        server_version: Server version string
        
    Returns:
        Warning message if versions are incompatible, None if compatible or parsing fails
    """
    sdk_parsed = parse_semantic_version(sdk_version)
    server_parsed = parse_semantic_version(server_version)
    
    if not sdk_parsed or not server_parsed:
        logger.debug(f"Could not parse versions - SDK: {sdk_version}, Server: {server_version}")
        return None
    
    sdk_major, sdk_minor, sdk_patch = sdk_parsed
    server_major, server_minor, server_patch = server_parsed
    
    # Only compare minor versions as requested
    if sdk_minor == server_minor:
        return None
    
    # Generate appropriate warning message
    if sdk_minor < server_minor:
        recommendation = (
            f"Please upgrade your SDK to match the server version:\n"
            f"   pip install --upgrade memfuse"
        )
    else:
        recommendation = (
            f"Please upgrade your server to match the SDK version.\n"
            f"   Your server version ({server_version}) is behind the SDK version ({sdk_version})."
        )
    
    warning_message = (
        f"⚠️  Version mismatch detected:\n"
        f"   SDK version: {sdk_version}\n"
        f"   Server version: {server_version}\n\n"
        f"   {recommendation}\n\n"
        f"   SDK: {SDK_GITHUB_URL}\n"
        f"   Server: {SERVER_GITHUB_URL}"
    )
    
    return warning_message


def extract_version_from_health_response(health_data: Dict[str, Any]) -> Optional[str]:
    """Extract version information from health check response.
    
    Based on the actual MemFuse server response format:
    {
        "status": "success",
        "code": 200,
        "data": {
            "status": "ok",
            "version": "0.3.1",
            ...
        }
    }
    
    Args:
        health_data: Response data from /api/v1/health endpoint
        
    Returns:
        Version string if found, None otherwise
    """
    if not isinstance(health_data, dict):
        return None
    
    # Check the known structure: data.version
    if 'data' in health_data and isinstance(health_data['data'], dict):
        if 'version' in health_data['data']:
            version = health_data['data']['version']
            if isinstance(version, str) and version.strip():
                return version.strip()
    
    # Fallback: try common fields at top level
    version_fields = ['version', 'server_version', 'api_version']
    for field in version_fields:
        if field in health_data:
            version = health_data[field]
            if isinstance(version, str) and version.strip():
                return version.strip()
    
    logger.debug(f"No version found in health response keys: {list(health_data.keys())}")
    return None


def check_version_compatibility(sdk_version: str, health_data: Dict[str, Any]) -> Optional[str]:
    """Check version compatibility and return warning if needed.
    
    Args:
        sdk_version: Current SDK version
        health_data: Response data from health check endpoint
        
    Returns:
        Warning message if versions are incompatible, None if compatible
    """
    if not sdk_version or not health_data:
        return None
    
    server_version = extract_version_from_health_response(health_data)
    
    if not server_version:
        logger.debug("Could not extract server version from health response")
        return None
    
    return compare_versions(sdk_version, server_version)
