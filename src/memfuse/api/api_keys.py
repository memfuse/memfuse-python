"""API Keys API client for MemFuse."""

from typing import Dict, Optional, Any


class ApiKeysApi:
    """API Keys API client for MemFuse."""

    # Shared endpoints configuration
    ENDPOINTS = {
        'list': {'method': 'GET', 'path': '/api/v1/users/{user_id}/api-keys'},
        'create': {'method': 'POST', 'path': '/api/v1/users/{user_id}/api-keys'},
        'delete': {'method': 'DELETE', 'path': '/api/v1/users/{user_id}/api-keys/{api_key_id}'},
    }

    def __init__(self, client):
        """Initialize the API keys API client.

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

    def _build_create_data(
        self,
        name: Optional[str] = None,
        key: Optional[str] = None,
        permissions: Optional[Dict[str, Any]] = None,
        expires_at: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build data for create request."""
        data = {}
        if name is not None:
            data["name"] = name
        if key is not None:
            data["key"] = key
        if permissions is not None:
            data["permissions"] = permissions
        if expires_at is not None:
            data["expires_at"] = expires_at
        return data

    # Async methods
    async def list(self, user_id: str) -> Dict[str, Any]:
        """List all API keys for a user.

        Args:
            user_id: User ID

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('list'),
            self._build_url('list', user_id=user_id)
        )

    async def create(
        self, 
        user_id: str, 
        name: Optional[str] = None,
        key: Optional[str] = None,
        permissions: Optional[Dict[str, Any]] = None,
        expires_at: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new API key for a user.

        Args:
            user_id: User ID
            name: Name of the API key (optional)
            key: Custom API key value (optional)
            permissions: Permissions for the API key (optional)
            expires_at: Expiration date for the API key (optional)

        Returns:
            Response data
        """
        data = self._build_create_data(name, key, permissions, expires_at)
        return await self.client._request(
            self._get_method('create'),
            self._build_url('create', user_id=user_id),
            data or None,
        )

    async def delete(self, user_id: str, api_key_id: str) -> Dict[str, Any]:
        """Delete an API key.

        Args:
            user_id: User ID
            api_key_id: API key ID

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('delete'),
            self._build_url('delete', user_id=user_id, api_key_id=api_key_id),
        )

    # Sync methods
    def list_sync(self, user_id: str) -> Dict[str, Any]:
        """List all API keys for a user (sync version).

        Args:
            user_id: User ID

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('list'),
            self._build_url('list', user_id=user_id)
        )

    def create_sync(
        self, 
        user_id: str, 
        name: Optional[str] = None,
        key: Optional[str] = None,
        permissions: Optional[Dict[str, Any]] = None,
        expires_at: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new API key for a user (sync version).

        Args:
            user_id: User ID
            name: Name of the API key (optional)
            key: Custom API key value (optional)
            permissions: Permissions for the API key (optional)
            expires_at: Expiration date for the API key (optional)

        Returns:
            Response data
        """
        data = self._build_create_data(name, key, permissions, expires_at)
        return self.client._request_sync(
            self._get_method('create'),
            self._build_url('create', user_id=user_id),
            data or None,
        )

    def delete_sync(self, user_id: str, api_key_id: str) -> Dict[str, Any]:
        """Delete an API key (sync version).

        Args:
            user_id: User ID
            api_key_id: API key ID

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('delete'),
            self._build_url('delete', user_id=user_id, api_key_id=api_key_id),
        )
