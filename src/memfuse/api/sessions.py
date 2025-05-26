"""Sessions API client for MemFuse."""

from typing import Dict, List, Optional, Any


class SessionsApi:
    """Sessions API client for MemFuse."""

    # Shared endpoints configuration
    ENDPOINTS = {
        'list': {'method': 'GET', 'path': '/api/v1/sessions'},
        'get': {'method': 'GET', 'path': '/api/v1/sessions/{session_id}'},
        'get_by_name': {'method': 'GET', 'path': '/api/v1/sessions?name={name}'},
        'create': {'method': 'POST', 'path': '/api/v1/sessions'},
        'update': {'method': 'PUT', 'path': '/api/v1/sessions/{session_id}'},
        'delete': {'method': 'DELETE', 'path': '/api/v1/sessions/{session_id}'},
        'add_messages': {'method': 'POST', 'path': '/api/v1/sessions/{session_id}/messages'},
        'read_messages': {'method': 'POST', 'path': '/api/v1/sessions/{session_id}/messages/read'},
        'update_messages': {'method': 'PUT', 'path': '/api/v1/sessions/{session_id}/messages'},
        'delete_messages': {'method': 'DELETE', 'path': '/api/v1/sessions/{session_id}/messages'},
    }

    def __init__(self, client):
        """Initialize the sessions API client.

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

    def _build_list_url(self, user_id: Optional[str] = None, agent_id: Optional[str] = None) -> str:
        """Build URL for list endpoint with optional query parameters."""
        params = []
        if user_id:
            params.append(f"user_id={user_id}")
        if agent_id:
            params.append(f"agent_id={agent_id}")

        url = self._build_url('list')
        if params:
            url += "?" + "&".join(params)
        return url

    # Async methods
    async def list(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """List sessions, optionally filtered by user and/or agent.

        Args:
            user_id: User ID (optional)
            agent_id: Agent ID (optional)

        Returns:
            Response data
        """
        url = self._build_list_url(user_id, agent_id)
        return await self.client._request(self._get_method('list'), url)

    async def get(self, session_id: str) -> Dict[str, Any]:
        """Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('get'),
            self._build_url('get', session_id=session_id)
        )

    async def get_by_name(self, name: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a session by name.
        
        Args:
            name: Session name
            user_id: User ID for user-scoped lookup (optional, defaults to global lookup)
        """
        # Build URL with optional user_id parameter
        params = [f"name={name}"]
        if user_id:
            params.append(f"user_id={user_id}")
        
        url = "/api/v1/sessions?" + "&".join(params)
        return await self.client._request("GET", url)

    async def create(
        self, user_id: str, agent_id: str, name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new session.

        Args:
            user_id: User ID
            agent_id: Agent ID
            name: Session name (optional, will be auto-generated if not provided)

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('create'),
            self._build_url('create'),
            {
                "user_id": user_id,
                "agent_id": agent_id,
                "name": name,
            },
        )

    async def update(self, session_id: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Update a session.

        Args:
            session_id: Session ID
            name: New session name (optional)

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('update'),
            self._build_url('update', session_id=session_id),
            {
                "name": name,
            },
        )

    async def delete(self, session_id: str) -> Dict[str, Any]:
        """Delete a session.

        Args:
            session_id: Session ID

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('delete'),
            self._build_url('delete', session_id=session_id)
        )

    async def add_messages(self, session_id: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Add messages to a session.

        Args:
            session_id: Session ID
            messages: List of message dictionaries with role and content

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('add_messages'),
            self._build_url('add_messages', session_id=session_id),
            {
                "messages": messages,
            },
        )

    async def read_messages(self, session_id: str, message_ids: List[str]) -> Dict[str, Any]:
        """Read messages from a session.

        Args:
            session_id: Session ID
            message_ids: List of message IDs

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('read_messages'),
            self._build_url('read_messages', session_id=session_id),
            {
                "message_ids": message_ids,
            },
        )

    async def update_messages(
        self, session_id: str, message_ids: List[str], new_messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Update messages in a session.

        Args:
            session_id: Session ID
            message_ids: List of message IDs
            new_messages: List of new message dictionaries

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('update_messages'),
            self._build_url('update_messages', session_id=session_id),
            {
                "message_ids": message_ids,
                "new_messages": new_messages,
            },
        )

    async def delete_messages(self, session_id: str, message_ids: List[str]) -> Dict[str, Any]:
        """Delete messages from a session.

        Args:
            session_id: Session ID
            message_ids: List of message IDs

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('delete_messages'),
            self._build_url('delete_messages', session_id=session_id),
            {
                "message_ids": message_ids,
            },
        )

    # Sync methods
    def list_sync(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """List sessions, optionally filtered by user and/or agent (sync version).

        Args:
            user_id: User ID (optional)
            agent_id: Agent ID (optional)

        Returns:
            Response data
        """
        url = self._build_list_url(user_id, agent_id)
        return self.client._request_sync(self._get_method('list'), url)

    def get_sync(self, session_id: str) -> Dict[str, Any]:
        """Get a session by ID (sync version).

        Args:
            session_id: Session ID

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('get'),
            self._build_url('get', session_id=session_id)
        )

    def get_by_name_sync(self, name: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a session by name (sync version).

        Args:
            name: Session name
            user_id: User ID for user-scoped lookup (optional, defaults to global lookup)

        Returns:
            Response data
        """
        # Build URL with optional user_id parameter
        params = [f"name={name}"]
        if user_id:
            params.append(f"user_id={user_id}")
        
        url = "/api/v1/sessions?" + "&".join(params)
        return self.client._request_sync("GET", url)

    def create_sync(
        self, user_id: str, agent_id: str, name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new session (sync version).

        Args:
            user_id: User ID
            agent_id: Agent ID
            name: Session name (optional, will be auto-generated if not provided)

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('create'),
            self._build_url('create'),
            {
                "user_id": user_id,
                "agent_id": agent_id,
                "name": name,
            },
        )

    def update_sync(self, session_id: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Update a session (sync version).

        Args:
            session_id: Session ID
            name: New session name (optional)

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('update'),
            self._build_url('update', session_id=session_id),
            {
                "name": name,
            },
        )

    def delete_sync(self, session_id: str) -> Dict[str, Any]:
        """Delete a session (sync version).

        Args:
            session_id: Session ID

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('delete'),
            self._build_url('delete', session_id=session_id)
        )

    def add_messages_sync(self, session_id: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Add messages to a session (sync version).

        Args:
            session_id: Session ID
            messages: List of message dictionaries with role and content

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('add_messages'),
            self._build_url('add_messages', session_id=session_id),
            {
                "messages": messages,
            },
        )

    def read_messages_sync(self, session_id: str, message_ids: List[str]) -> Dict[str, Any]:
        """Read messages from a session (sync version).

        Args:
            session_id: Session ID
            message_ids: List of message IDs

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('read_messages'),
            self._build_url('read_messages', session_id=session_id),
            {
                "message_ids": message_ids,
            },
        )

    def update_messages_sync(
        self, session_id: str, message_ids: List[str], new_messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Update messages in a session (sync version).

        Args:
            session_id: Session ID
            message_ids: List of message IDs
            new_messages: List of new message dictionaries

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('update_messages'),
            self._build_url('update_messages', session_id=session_id),
            {
                "message_ids": message_ids,
                "new_messages": new_messages,
            },
        )

    def delete_messages_sync(self, session_id: str, message_ids: List[str]) -> Dict[str, Any]:
        """Delete messages from a session (sync version).

        Args:
            session_id: Session ID
            message_ids: List of message IDs

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('delete_messages'),
            self._build_url('delete_messages', session_id=session_id),
            {
                "message_ids": message_ids,
            },
        )
