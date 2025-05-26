"""Knowledge API client for MemFuse."""

from typing import Dict, List, Optional, Any


class KnowledgeApi:
    """Knowledge API client for MemFuse."""

    # Shared endpoints configuration
    ENDPOINTS = {
        'list': {'method': 'GET', 'path': '/api/v1/users/{user_id}/knowledge'},
        'add': {'method': 'POST', 'path': '/api/v1/users/{user_id}/knowledge'},
        'read': {'method': 'POST', 'path': '/api/v1/users/{user_id}/knowledge/read'},
        'update': {'method': 'PUT', 'path': '/api/v1/users/{user_id}/knowledge'},
        'delete': {'method': 'DELETE', 'path': '/api/v1/users/{user_id}/knowledge'},
    }

    def __init__(self, client):
        """Initialize the knowledge API client.

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
    async def list(self, user_id: str) -> Dict[str, Any]:
        """List all knowledge items for a user.

        Args:
            user_id: User ID

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('list'),
            self._build_url('list', user_id=user_id)
        )

    async def add(self, user_id: str, knowledge: List[str]) -> Dict[str, Any]:
        """Add knowledge to a user.

        Args:
            user_id: User ID
            knowledge: List of knowledge strings

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('add'),
            self._build_url('add', user_id=user_id),
            {
                "knowledge": knowledge,
            },
        )

    async def read(self, user_id: str, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Read knowledge from a user.

        Args:
            user_id: User ID
            knowledge_ids: List of knowledge IDs

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('read'),
            self._build_url('read', user_id=user_id),
            {
                "knowledge_ids": knowledge_ids,
            },
        )

    async def update(
        self, user_id: str, knowledge_ids: List[str], new_knowledge: List[str]
    ) -> Dict[str, Any]:
        """Update knowledge for a user.

        Args:
            user_id: User ID
            knowledge_ids: List of knowledge IDs
            new_knowledge: List of new knowledge strings

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('update'),
            self._build_url('update', user_id=user_id),
            {
                "knowledge_ids": knowledge_ids,
                "new_knowledge": new_knowledge,
            },
        )

    async def delete(self, user_id: str, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Delete knowledge from a user.

        Args:
            user_id: User ID
            knowledge_ids: List of knowledge IDs

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('delete'),
            self._build_url('delete', user_id=user_id),
            {
                "knowledge_ids": knowledge_ids,
            },
        )

    # Sync methods
    def list_sync(self, user_id: str) -> Dict[str, Any]:
        """List all knowledge items for a user (sync version).

        Args:
            user_id: User ID

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('list'),
            self._build_url('list', user_id=user_id)
        )

    def add_sync(self, user_id: str, knowledge: List[str]) -> Dict[str, Any]:
        """Add knowledge to a user (sync version).

        Args:
            user_id: User ID
            knowledge: List of knowledge strings

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('add'),
            self._build_url('add', user_id=user_id),
            {
                "knowledge": knowledge,
            },
        )

    def read_sync(self, user_id: str, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Read knowledge from a user (sync version).

        Args:
            user_id: User ID
            knowledge_ids: List of knowledge IDs

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('read'),
            self._build_url('read', user_id=user_id),
            {
                "knowledge_ids": knowledge_ids,
            },
        )

    def update_sync(
        self, user_id: str, knowledge_ids: List[str], new_knowledge: List[str]
    ) -> Dict[str, Any]:
        """Update knowledge for a user (sync version).

        Args:
            user_id: User ID
            knowledge_ids: List of knowledge IDs
            new_knowledge: List of new knowledge strings

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('update'),
            self._build_url('update', user_id=user_id),
            {
                "knowledge_ids": knowledge_ids,
                "new_knowledge": new_knowledge,
            },
        )

    def delete_sync(self, user_id: str, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Delete knowledge from a user (sync version).

        Args:
            user_id: User ID
            knowledge_ids: List of knowledge IDs

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('delete'),
            self._build_url('delete', user_id=user_id),
            {
                "knowledge_ids": knowledge_ids,
            },
        )
