"""Agents API client for MemFuse."""

from typing import Dict, List, Optional, Any


class AgentsApi:
    """Agents API client for MemFuse."""

    # Shared endpoints configuration
    ENDPOINTS = {
        'list': {'method': 'GET', 'path': '/api/v1/agents'},
        'get': {'method': 'GET', 'path': '/api/v1/agents/{agent_id}'},
        'get_by_name': {'method': 'GET', 'path': '/api/v1/agents?name={name}'},
        'create': {'method': 'POST', 'path': '/api/v1/agents'},
        'update': {'method': 'PUT', 'path': '/api/v1/agents/{agent_id}'},
        'delete': {'method': 'DELETE', 'path': '/api/v1/agents/{agent_id}'},
    }

    def __init__(self, client):
        """Initialize the agents API client.

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
        """List all agents.

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('list'),
            self._build_url('list')
        )

    async def get(self, agent_id: str) -> Dict[str, Any]:
        """Get an agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('get'),
            self._build_url('get', agent_id=agent_id)
        )

    async def get_by_name(self, name: str) -> Dict[str, Any]:
        """Get an agent by name.

        Args:
            name: Agent name

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('get_by_name'),
            self._build_url('get_by_name', name=name)
        )

    async def create(
        self,
        name: str,
        description: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new agent.

        Args:
            name: Agent name
            description: Agent description (optional)

        Returns:
            Response data
        """
        extra_headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return await self.client._request(
            self._get_method('create'),
            self._build_url('create'),
            {
                "name": name,
                "description": description,
            },
            extra_headers=extra_headers,
        )

    async def update(
        self, agent_id: str, name: Optional[str] = None, description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update an agent.

        Args:
            agent_id: Agent ID
            name: New agent name (optional)
            description: New agent description (optional)

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('update'),
            self._build_url('update', agent_id=agent_id),
            {
                "name": name,
                "description": description,
            },
        )

    async def delete(self, agent_id: str) -> Dict[str, Any]:
        """Delete an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Response data
        """
        return await self.client._request(
            self._get_method('delete'),
            self._build_url('delete', agent_id=agent_id)
        )

    # Sync methods
    def list_sync(self) -> Dict[str, Any]:
        """List all agents (sync version).

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('list'),
            self._build_url('list')
        )

    def get_sync(self, agent_id: str) -> Dict[str, Any]:
        """Get an agent by ID (sync version).

        Args:
            agent_id: Agent ID

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('get'),
            self._build_url('get', agent_id=agent_id)
        )

    def get_by_name_sync(self, name: str) -> Dict[str, Any]:
        """Get an agent by name (sync version).

        Args:
            name: Agent name

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('get_by_name'),
            self._build_url('get_by_name', name=name)
        )

    def create_sync(
        self,
        name: str,
        description: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new agent (sync version).

        Args:
            name: Agent name
            description: Agent description (optional)

        Returns:
            Response data
        """
        extra_headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return self.client._request_sync(
            self._get_method('create'),
            self._build_url('create'),
            {
                "name": name,
                "description": description,
            },
            extra_headers=extra_headers,
        )

    def update_sync(
        self, agent_id: str, name: Optional[str] = None, description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update an agent (sync version).

        Args:
            agent_id: Agent ID
            name: New agent name (optional)
            description: New agent description (optional)

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('update'),
            self._build_url('update', agent_id=agent_id),
            {
                "name": name,
                "description": description,
            },
        )

    def delete_sync(self, agent_id: str) -> Dict[str, Any]:
        """Delete an agent (sync version).

        Args:
            agent_id: Agent ID

        Returns:
            Response data
        """
        return self.client._request_sync(
            self._get_method('delete'),
            self._build_url('delete', agent_id=agent_id)
        )
