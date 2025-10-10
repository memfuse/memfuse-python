"""Users API client for MemFuse."""

from typing import Dict, List, Optional, Any
import warnings


class UsersApi:
    """Users API client for MemFuse."""

    # Shared endpoints configuration
    ENDPOINTS = {
        'list': {'method': 'GET', 'path': '/api/v1/users'},
        'get': {'method': 'GET', 'path': '/api/v1/users/{user_id}'},
        'get_by_name': {'method': 'GET', 'path': '/api/v1/users?name={name}'},
        'create': {'method': 'POST', 'path': '/api/v1/users'},
        'update': {'method': 'PUT', 'path': '/api/v1/users/{user_id}'},
        'delete': {'method': 'DELETE', 'path': '/api/v1/users/{user_id}'},
        'query': {'method': 'POST', 'path': '/api/v1/users/{user_id}/query'},
    }

    def __init__(self, client):
        """Initialize the users API client.

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
    async def list(self) -> Dict[str, Any]:
        """List all users.

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('list'), 
            self._build_url('list')
        )

    async def get(self, user_id: str) -> Dict[str, Any]:
        """Get a user by ID.

        Args:
            user_id: User ID

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('get'), 
            self._build_url('get', user_id=user_id)
        )

    async def get_by_name(self, name: str) -> Dict[str, Any]:
        """Get a user by name.

        Args:
            name: User name

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('get_by_name'), 
            self._build_url('get_by_name', name=name)
        )

    async def create(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a new user.

        Args:
            name: User name
            description: User description (optional)

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('create'),
            self._build_url('create'),
            {
                "name": name,
                "description": description,
            },
        )

    async def update(
        self, user_id: str, name: Optional[str] = None, description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update a user.

        Args:
            user_id: User ID
            name: New user name (optional)
            description: New user description (optional)

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('update'),
            self._build_url('update', user_id=user_id),
            {
                "name": name,
                "description": description,
            },
        )

    async def delete(self, user_id: str) -> Dict[str, Any]:
        """Delete a user.

        Args:
            user_id: User ID

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('delete'), 
            self._build_url('delete', user_id=user_id)
        )

    async def query(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
        # Deprecated: kept for backwards compatibility; ignored in request payload
        store_type: Optional[str] = None,
        include_messages: bool = True,
        include_knowledge: bool = True,
    ) -> Dict[str, Any]:
        """Query memory across all sessions for a user.

        Args:
            user_id: User ID
            query: Query string
            session_id: Session ID (optional)
            agent_id: Agent ID (optional)
            top_k: Number of results to return
            metadata: Optional metadata to provide additional query context (e.g., {"task": "...", "mode": "..."})
            store_type: Deprecated; ignored
            include_messages: Deprecated; ignored
            include_knowledge: Deprecated; ignored

        Returns:
            Response data
        """
        # Emit deprecation warnings if legacy args are passed explicitly
        if store_type is not None or not include_messages or not include_knowledge:
            warnings.warn(
                "users.query: 'store_type', 'include_messages', and 'include_knowledge' are deprecated and ignored in request payload.",
                DeprecationWarning,
                stacklevel=2,
            )

        payload: Dict[str, Any] = {
            "query": query,
            "session_id": session_id,
            "agent_id": agent_id,
            "top_k": top_k,
        }
        if metadata is not None:
            payload["metadata"] = metadata

        return await self.client._request(
            self._get_method('query'),
            self._build_url('query', user_id=user_id),
            payload,
        )

    # Sync methods
    def list_sync(self) -> Dict[str, Any]:
        """List all users (sync version).

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('list'), 
            self._build_url('list')
        )

    def get_sync(self, user_id: str) -> Dict[str, Any]:
        """Get a user by ID (sync version).

        Args:
            user_id: User ID

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('get'), 
            self._build_url('get', user_id=user_id)
        )

    def get_by_name_sync(self, name: str) -> Dict[str, Any]:
        """Get a user by name (sync version).

        Args:
            name: User name

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('get_by_name'), 
            self._build_url('get_by_name', name=name)
        )

    def create_sync(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a new user (sync version).

        Args:
            name: User name
            description: User description (optional)

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('create'),
            self._build_url('create'),
            {
                "name": name,
                "description": description,
            },
        )

    def update_sync(
        self, user_id: str, name: Optional[str] = None, description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update a user (sync version).

        Args:
            user_id: User ID
            name: New user name (optional)
            description: New user description (optional)

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('update'),
            self._build_url('update', user_id=user_id),
            {
                "name": name,
                "description": description,
            },
        )

    def delete_sync(self, user_id: str) -> Dict[str, Any]:
        """Delete a user (sync version).

        Args:
            user_id: User ID

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('delete'), 
            self._build_url('delete', user_id=user_id)
        )

    def query_sync(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
        # Deprecated: kept for backwards compatibility; ignored in request payload
        store_type: Optional[str] = None,
        include_messages: bool = True,
        include_knowledge: bool = True,
    ) -> Dict[str, Any]:
        """Query memory across all sessions for a user (sync version).

        Args:
            user_id: User ID
            query: Query string
            session_id: Session ID (optional)
            agent_id: Agent ID (optional)
            top_k: Number of results to return
            metadata: Optional metadata to provide additional query context (e.g., {"task": "...", "mode": "..."})
            store_type: Deprecated; ignored
            include_messages: Deprecated; ignored
            include_knowledge: Deprecated; ignored

        Returns:
            Response data
        """
        # Emit deprecation warnings if legacy args are passed explicitly
        if store_type is not None or not include_messages or not include_knowledge:
            warnings.warn(
                "users.query_sync: 'store_type', 'include_messages', and 'include_knowledge' are deprecated and ignored in request payload.",
                DeprecationWarning,
                stacklevel=2,
            )

        payload: Dict[str, Any] = {
            "query": query,
            "session_id": session_id,
            "agent_id": agent_id,
            "top_k": top_k,
        }
        if metadata is not None:
            payload["metadata"] = metadata

        return self.client._request_sync(
            self._get_method('query'),
            self._build_url('query', user_id=user_id),
            payload,
        )
