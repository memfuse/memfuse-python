"""Health API client for MemFuse."""

from typing import Dict, Any


class HealthApi:
    """Health API client for MemFuse."""
    
    # Shared endpoints configuration
    ENDPOINTS = {
        'health_check': {'method': 'GET', 'path': '/api/v1/health/'},
    }
    
    def __init__(self, client):
        """Initialize the health API client.
        
        Args:
            client: The MemFuse client
        """
        self.client = client

    def _build_url(self, endpoint_key: str, **kwargs) -> str:
        """Build URL from endpoint configuration and parameters."""
        endpoint = self.ENDPOINTS[endpoint_key]
        return endpoint['path'].format(**kwargs)

    def _get_method(self, endpoint_key: str) -> str:
        """Get HTTP method for endpoint."""
        return self.ENDPOINTS[endpoint_key]['method']
    
    # Async methods
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the server.
        
        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('health_check'),
            self._build_url('health_check')
        )

    # Sync methods
    def health_check_sync(self) -> Dict[str, Any]:
        """Check the health of the server (sync version).
        
        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('health_check'),
            self._build_url('health_check')
        )
