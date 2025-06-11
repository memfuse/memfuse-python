"""MemFuse client implementation."""

import os
import aiohttp
from typing import Dict, Optional, Any, List

from .memory import AsyncMemory
from .api import (
    HealthApi,
    UsersApi,
    AgentsApi,
    SessionsApi,
    KnowledgeApi,
    MessagesApi,
    ApiKeysApi
)


class AsyncMemFuse:
    """MemFuse client for communicating with the MemFuse server."""

    # Class variable to track all instances
    _instances = set()

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """Initialize the MemFuse client.

        Args:
            base_url: URL of the MemFuse server API
            api_key: API key for authentication (optional for local usage)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("MEMFUSE_API_KEY")
        self.session = None

        # Initialize ASYNC API clients using the classes from .api
        self.health = HealthApi(self)
        self.users = UsersApi(self)
        self.agents = AgentsApi(self)
        self.sessions = SessionsApi(self)
        self.knowledge = KnowledgeApi(self)
        self.messages = MessagesApi(self)
        self.api_keys = ApiKeysApi(self)

        # Add self to instances
        AsyncMemFuse._instances.add(self)

    async def _ensure_session(self):
        """Ensure that an HTTP session exists."""
        if self.session is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self.session = aiohttp.ClientSession(headers=headers)

    async def _close_session(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def _check_server_health(self) -> bool:
        """Check if the server is running.

        Returns:
            True if the server is running, False otherwise
        """
        await self._ensure_session()
        try:
            url = f"{self.base_url}/api/v1/health"
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    return True
                return False
        except Exception:
            return False

    async def _request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a request to the MemFuse server.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request data

        Returns:
            Response data

        Raises:
            ConnectionError: If the server is not running, with a helpful error message
        """
        await self._ensure_session()

        # Check if the server is running
        try:
            if not await self._check_server_health():
                # Instead of raising a plain ConnectionError, we raise a custom exception
                # with a helpful error message that includes instructions on how to start the server
                raise ConnectionError(
                    f"Cannot connect to MemFuse server at {self.base_url}. "
                    "Please make sure the server is running.\n\n"
                    "You can start the server with:\n"
                    "  poetry run memfuse-core"
                )

            url = f"{self.base_url}{endpoint}"

            async with getattr(self.session, method.lower())(url, json=data) as response:
                response_data = await response.json()
                if response.status >= 400:
                    error_message = response_data.get("message", "Unknown error")
                    raise Exception(f"API request failed: {error_message}")
                return response_data

        except aiohttp.ClientConnectorError as e:
            # Catch aiohttp connection errors and convert them to our custom ConnectionError
            # Use "raise ... from e" to maintain the exception chain
            raise ConnectionError(
                f"Cannot connect to MemFuse server at {self.base_url}. "
                "Please make sure the server is running.\n\n"
                "You can start the server with:\n"
                "  poetry run memfuse-core"
            ) from e

    async def init(
        self,
        user: str,
        agent: Optional[str] = None,
        session: Optional[str] = None,
    ) -> AsyncMemory:
        """Initialize a memory instance.

        Args:
            user: User name (required)
            agent: Agent name (optional)
            session: Session name (optional, will be auto-generated if not provided)

        Returns:
            ClientMemory: A client memory instance for the specified user, agent, and session
        """
        # Get or create user
        user_name = user
        user_response = await self.users.get_by_name(user_name)

        if user_response["status"] == "error" or not user_response.get("data", {}).get("users"):
            # User doesn't exist, create it
            user_response = await self.users.create(
                name=user_name,
                description="User created by MemFuse client"
            )
            user_id = user_response["data"]["user"]["id"]
        else:
            # User exists, get the first one
            user_id = user_response["data"]["users"][0]["id"]

        # Get or create agent
        agent_name = agent or "agent_default"
        agent_response = await self.agents.get_by_name(agent_name)

        if agent_response["status"] == "error" or not agent_response.get("data", {}).get("agents"):
            # Agent doesn't exist, create it
            agent_response = await self.agents.create(
                name=agent_name,
                description="Agent created by MemFuse client"
            )
            agent_id = agent_response["data"]["agent"]["id"]
        else:
            # Agent exists, get the first one
            agent_id = agent_response["data"]["agents"][0]["id"]

        # Check if session with the given name already exists
        session_name = session
        if session_name:
            session_response = await self.sessions.get_by_name(session_name, user_id=user_id)
            if session_response["status"] == "success" and session_response.get("data", {}).get("sessions"):
                # Session exists, get the first one
                session_data = session_response["data"]["sessions"][0]
                session_id = session_data["id"]
            else:
                # Session doesn't exist, create it
                session_response = await self.sessions.create(
                    user_id=user_id,
                    agent_id=agent_id,
                    name=session_name
                )
                session_data = session_response["data"]["session"]
                session_id = session_data["id"]
        else:
            # No session name provided, create a new session
            session_response = await self.sessions.create(
                user_id=user_id,
                agent_id=agent_id
            )
            session_data = session_response["data"]["session"]
            session_id = session_data["id"]
            session_name = session_data["name"]

        # Create ClientMemory with all necessary parameters
        memory = AsyncMemory(
            client=self,
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            user_name=user_name,
            agent_name=agent_name,
            session_name=session_name
        )

        return memory

    async def close(self):
        """Close the client session.

        This method should be called when the client is no longer needed
        to properly clean up resources.
        """
        await self._close_session()

        # Remove self from instances
        if self in AsyncMemFuse._instances:
            AsyncMemFuse._instances.remove(self)

    async def __aenter__(self):
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager.
        
        This method is called when the context is exited, either normally
        or due to an exception. It closes the client session.
        """
        await self.close()

    async def _thread_safe_coro_runner(self, coro):
        """Runs a coroutine with session management, suitable for asyncio.run() in a new thread."""
        await self._ensure_session()
        try:
            return await coro
        finally:
            await self._close_session()


class MemFuse:
    """Synchronous MemFuse client for communicating with the MemFuse server."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """Initialize the synchronous MemFuse client.

        Args:
            base_url: URL of the MemFuse server API
            api_key: API key for authentication (optional for local usage)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("MEMFUSE_API_KEY")
        self.sync_session = None  # requests session for sync requests

        # Initialize API clients using the classes from .api
        self.health = HealthApi(self)
        self.users = UsersApi(self)
        self.agents = AgentsApi(self)
        self.sessions = SessionsApi(self)
        self.knowledge = KnowledgeApi(self)
        self.messages = MessagesApi(self)
        self.api_keys = ApiKeysApi(self)

    def _ensure_sync_session(self):
        """Ensure that a sync HTTP session exists."""
        if self.sync_session is None:
            import requests
            self.sync_session = requests.Session()
            if self.api_key:
                self.sync_session.headers["Authorization"] = f"Bearer {self.api_key}"

    def _close_sync_session(self):
        """Close the sync HTTP session."""
        if self.sync_session:
            self.sync_session.close()
            self.sync_session = None

    def _check_server_health_sync(self) -> bool:
        """Check if the server is running (sync version).

        Returns:
            True if the server is running, False otherwise
        """
        self._ensure_sync_session()
        try:
            url = f"{self.base_url}/api/v1/health"
            response = self.sync_session.get(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def _request_sync(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a sync request to the MemFuse server.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request data

        Returns:
            Response data

        Raises:
            ConnectionError: If the server is not running, with a helpful error message
        """
        import requests
        self._ensure_sync_session()

        # Check if the server is running
        try:
            if not self._check_server_health_sync():
                raise ConnectionError(
                    f"Cannot connect to MemFuse server at {self.base_url}. "
                    "Please make sure the server is running.\n\n"
                    "You can start the server with:\n"
                    "  poetry run memfuse-core"
                )

            url = f"{self.base_url}{endpoint}"

            response = getattr(self.sync_session, method.lower())(url, json=data, timeout=10)
            response_data = response.json()
            if response.status_code >= 400:
                error_message = response_data.get("message", "Unknown error")
                raise Exception(f"API request failed: {error_message}")
            return response_data

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to MemFuse server at {self.base_url}. "
                "Please make sure the server is running.\n\n"
                "You can start the server with:\n"
                "  poetry run memfuse-core"
            ) from e

    def init(
        self,
        user: str,
        agent: Optional[str] = None,
        session: Optional[str] = None,
    ) -> 'Memory': 
        """Initialize a synchronous memory instance.

        Args:
            user: User name (required)
            agent: Agent name (optional)
            session: Session name (optional, will be auto-generated if not provided)

        Returns:
            Memory: A synchronous memory instance.
        """
        # Get or create user
        user_name = user
        user_response = self.users.get_by_name_sync(user_name)

        if user_response["status"] == "error" or not user_response.get("data", {}).get("users"):
            # User doesn't exist, create it
            user_response = self.users.create_sync(
                name=user_name,
                description="User created by MemFuse client"
            )
            user_id = user_response["data"]["user"]["id"]
        else:
            # User exists, get the first one
            user_id = user_response["data"]["users"][0]["id"]

        # Get or create agent
        agent_name = agent or "agent_default"
        agent_response = self.agents.get_by_name_sync(agent_name)

        if agent_response["status"] == "error" or not agent_response.get("data", {}).get("agents"):
            # Agent doesn't exist, create it
            agent_response = self.agents.create_sync(
                name=agent_name,
                description="Agent created by MemFuse client"
            )
            agent_id = agent_response["data"]["agent"]["id"]
        else:
            # Agent exists, get the first one
            agent_id = agent_response["data"]["agents"][0]["id"]

        # Check if session with the given name already exists
        session_name = session
        if session_name:
            session_response = self.sessions.get_by_name_sync(session_name, user_id=user_id)
            if session_response["status"] == "success" and session_response.get("data", {}).get("sessions"):
                # Session exists, get the first one
                session_data = session_response["data"]["sessions"][0]
                session_id = session_data["id"]
            else:
                # Session doesn't exist, create it
                session_response = self.sessions.create_sync(
                    user_id=user_id,
                    agent_id=agent_id,
                    name=session_name
                )
                session_data = session_response["data"]["session"]
                session_id = session_data["id"]
        else:
            # No session name provided, create a new session
            session_response = self.sessions.create_sync(
                user_id=user_id,
                agent_id=agent_id
            )
            session_data = session_response["data"]["session"]
            session_id = session_data["id"]
            session_name = session_data["name"]

        from .memory import Memory 
        return Memory(
            client=self,
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            user_name=user_name,
            agent_name=agent_name,
            session_name=session_name
        )

    def close(self):
        """Close the client and its underlying sessions."""
        self._close_sync_session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
