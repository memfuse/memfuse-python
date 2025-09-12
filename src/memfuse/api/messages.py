"""Messages API client for MemFuse."""

from typing import Dict, List, Optional, Any


class MessagesApi:
    """Messages API client for MemFuse."""

    # Shared endpoints configuration
    ENDPOINTS = {
        'list': {'method': 'GET', 'path': '/api/v1/sessions/{session_id}/messages'},
        'add': {'method': 'POST', 'path': '/api/v1/sessions/{session_id}/messages'},
        'read': {'method': 'POST', 'path': '/api/v1/sessions/{session_id}/messages/read'},
        'update': {'method': 'PUT', 'path': '/api/v1/sessions/{session_id}/messages'},
        'delete': {'method': 'DELETE', 'path': '/api/v1/sessions/{session_id}/messages'},
    }

    def __init__(self, client):
        """Initialize the messages API client.

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

    def _build_list_url(
        self,
        session_id: str,
        limit: Optional[int] = None,
        sort_by: Optional[str] = None,
        order: Optional[str] = None,
        buffer_only: Optional[bool] = None
    ) -> str:
        """Build URL for list endpoint with optional query parameters."""
        query_params = []
        if limit is not None:
            query_params.append(f"limit={limit}")
        if sort_by is not None:
            query_params.append(f"sort_by={sort_by}")
        if order is not None:
            query_params.append(f"order={order}")
        if buffer_only is not None:
            query_params.append(f"buffer_only={str(buffer_only).lower()}")

        query_string = "&".join(query_params)
        endpoint = self._build_url('list', session_id=session_id)
        if query_string:
            endpoint = f"{endpoint}?{query_string}"
        return endpoint

    # Async methods
    async def list(
        self,
        session_id: str,
        limit: Optional[int] = 20,
        sort_by: Optional[str] = "timestamp",
        order: Optional[str] = "desc",
        buffer_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """List all messages in a session.

        Args:
            session_id: Session ID
            limit: Maximum number of messages to return. Defaults to 20.
            sort_by: Field to sort messages by (e.g., "timestamp", "id"). Defaults to "timestamp".
            order: Sort order ("asc" or "desc"). Defaults to "desc".
            buffer_only: If True, only return RoundBuffer data; if False, return HybridBuffer + SQLite data excluding RoundBuffer

        Returns:
            Response data
        """
        url = self._build_list_url(session_id, limit, sort_by, order, buffer_only)
        return await self.client._request(self._get_method('list'), url)

    async def add(self, session_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add messages to a session.

        Args:
            session_id: Session ID
            messages: List of message dictionaries with role, content, and optional metadata

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('add'),
            self._build_url('add', session_id=session_id),
            {
                "messages": messages,
            },
        )

    async def read(self, session_id: str, message_ids: List[str]) -> Dict[str, Any]:
        """Read messages from a session.

        Args:
            session_id: Session ID
            message_ids: List of message IDs

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('read'),
            self._build_url('read', session_id=session_id),
            {
                "message_ids": message_ids,
            },
        )

    async def update(
        self, session_id: str, message_ids: List[str], new_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update messages in a session.

        Args:
            session_id: Session ID
            message_ids: List of message IDs
            new_messages: List of new message dictionaries (role, content, optional metadata)

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('update'),
            self._build_url('update', session_id=session_id),
            {
                "message_ids": message_ids,
                "new_messages": new_messages,
            },
        )

    async def delete(self, session_id: str, message_ids: List[str]) -> Dict[str, Any]:
        """Delete messages from a session.

        Args:
            session_id: Session ID
            message_ids: List of message IDs

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('delete'),
            self._build_url('delete', session_id=session_id),
            {
                "message_ids": message_ids,
            },
        )

    # Sync methods
    def list_sync(
        self,
        session_id: str,
        limit: Optional[int] = 20,
        sort_by: Optional[str] = "timestamp",
        order: Optional[str] = "desc",
        buffer_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """List all messages in a session (sync version).

        Args:
            session_id: Session ID
            limit: Maximum number of messages to return. Defaults to 20.
            sort_by: Field to sort messages by (e.g., "timestamp", "id"). Defaults to "timestamp".
            order: Sort order ("asc" or "desc"). Defaults to "desc".
            buffer_only: If True, only return RoundBuffer data; if False, return HybridBuffer + SQLite data excluding RoundBuffer

        Returns:
            Response data
        """
        url = self._build_list_url(session_id, limit, sort_by, order, buffer_only)
        return self.client._request_sync(self._get_method('list'), url)

    def add_sync(self, session_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add messages to a session (sync version).

        Args:
            session_id: Session ID
            messages: List of message dictionaries with role, content, and optional metadata

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('add'),
            self._build_url('add', session_id=session_id),
            {
                "messages": messages,
            },
        )

    def read_sync(self, session_id: str, message_ids: List[str]) -> Dict[str, Any]:
        """Read messages from a session (sync version).

        Args:
            session_id: Session ID
            message_ids: List of message IDs

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('read'),
            self._build_url('read', session_id=session_id),
            {
                "message_ids": message_ids,
            },
        )

    def update_sync(
        self, session_id: str, message_ids: List[str], new_messages: List[Dict[str, Any]]
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
            self._get_method('update'),
            self._build_url('update', session_id=session_id),
            {
                "message_ids": message_ids,
                "new_messages": new_messages,
            },
        )

    def delete_sync(self, session_id: str, message_ids: List[str]) -> Dict[str, Any]:
        """Delete messages from a session (sync version).

        Args:
            session_id: Session ID
            message_ids: List of message IDs

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('delete'),
            self._build_url('delete', session_id=session_id),
            {
                "message_ids": message_ids,
            },
        )
