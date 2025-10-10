"""MemFuse client implementation."""

import os
import asyncio
import threading
import aiohttp
import json
from typing import Dict, Optional, Any
import uuid
import time
from loguru import logger

from .memory import AsyncMemory
from .utils import MemFuseHTTPError, check_version_compatibility
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

    # Instance-scoped singleflight state (initialized lazily per event loop)
    _agent_creation_futures: Optional[Dict[str, asyncio.Task]] = None
    _agent_creation_lock: Optional[asyncio.Lock] = None

    def __init__(self, base_url: str = "http://localhost:8765", api_key: Optional[str] = None, timeout: int = 10):
        """Initialize the MemFuse client.

        Args:
            base_url: URL of the MemFuse server API
            api_key: API key for authentication (optional for local usage)
            timeout: Request timeout in seconds (default: 10)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("MEMFUSE_API_KEY")
        self.timeout = timeout
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
        # Defer singleflight structures to runtime to ensure loop affinity
        self._agent_creation_futures = None
        self._agent_creation_lock = None

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
            async with self.session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    return True
                return False
        except Exception:
            return False

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
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

        # Debug logging
        if os.getenv("MEMFUSE_DEBUG") == "1":
            url = f"{self.base_url}{endpoint}"
            logger.debug(f"[MEMFUSE API] {method} {url}")
            if data:
                logger.debug(f"[MEMFUSE API] Request body: {json.dumps(data, indent=2)}")
            if extra_headers:
                logger.debug(f"[MEMFUSE API] Extra headers: {extra_headers}")

        # Perform the request (health check is handled during init)
        try:
            url = f"{self.base_url}{endpoint}"

            async with getattr(self.session, method.lower())(
                url,
                json=data,
                headers=extra_headers,
                timeout=self.timeout,
            ) as response:
                response_data = await response.json()
                if response.status >= 400:
                    error_message = response_data.get("message", "Unknown error")
                    raise MemFuseHTTPError(f"API request failed: {error_message}", response.status, response_data)
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

    async def _check_version_compatibility(self):
        """Check SDK and server version compatibility and display warnings if needed."""
        try:
            # Get SDK version
            from . import __version__
            sdk_version = __version__
            
            # If version is placeholder, try to get it from installed package metadata
            if not sdk_version or sdk_version == "{{version}}":
                try:
                    # Try to get version from installed package
                    from importlib.metadata import version, PackageNotFoundError
                    try:
                        sdk_version = version('memfuse')
                    except PackageNotFoundError:
                        # Package not installed via pip, skip version check
                        pass
                except ImportError:
                    # Python < 3.8, try backport
                    try:
                        from importlib_metadata import version, PackageNotFoundError
                        try:
                            sdk_version = version('memfuse')
                        except PackageNotFoundError:
                            pass
                    except ImportError:
                        pass
            
            if not sdk_version or sdk_version == "{{version}}":
                logger.debug("SDK version not available, skipping version check")
                return
            
            # Get server health information
            health_data = await self.health.health_check()
            
            # Check compatibility and print warning if needed
            warning = check_version_compatibility(sdk_version, health_data)
            if warning:
                print(warning)
                
        except Exception as e:
            # Don't fail init if version check fails
            logger.debug(f"Version compatibility check failed: {e}")

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
        # Ensure session and validate server health once up front
        await self._ensure_session()
        if not await self._check_server_health():
            raise ConnectionError(
                f"Cannot connect to MemFuse server at {self.base_url}. "
                "Please make sure the server is running.\n\n"
                "You can start the server with:\n"
                "  poetry run memfuse-core"
            )

        # Check version compatibility
        await self._check_version_compatibility()
        
        # Get or create user
        user_name = user
        try:
            user_response = await self.users.get_by_name(user_name)
            # If we get here, the user exists
            user_id = user_response["data"]["users"][0]["id"]
        except MemFuseHTTPError as e:
            if e.status_code == 404:
                # User doesn't exist, create it
                user_response = await self.users.create(
                    name=user_name,
                    description="User created by MemFuse client"
                )
                user_id = user_response["data"]["user"]["id"]
            else:
                # Re-raise other HTTP errors
                raise

        # Get or create agent using optimized singleflight mechanism
        agent_name = agent or "agent_default"
        agent_id = await self._get_or_create_agent(agent_name)

        # Check if session with the given name already exists
        session_name = session
        if session_name:
            try:
                session_response = await self.sessions.get_by_name(session_name, user_id=user_id)
                # If we get here, the session exists
                session_data = session_response["data"]["sessions"][0]
                session_id = session_data["id"]
            except MemFuseHTTPError as e:
                if e.status_code == 404:
                    # Session doesn't exist, create it
                    session_response = await self.sessions.create(
                        user_id=user_id,
                        agent_id=agent_id,
                        name=session_name
                    )
                    session_data = session_response["data"]["session"]
                    session_id = session_data["id"]
                else:
                    # Re-raise other HTTP errors
                    raise
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

    async def _get_or_create_agent(self, agent_name: str) -> str:
        """Get or create an agent with singleflight pattern to avoid concurrent creation.

        This method implements the recommended SDK-side optimizations:
        1. Singleflight pattern: merge concurrent requests for the same agent name
        2. POST-first flow leveraging server-side idempotent create
        3. Fallback handling: if POST fails with 400/409, GET the existing agent

        Args:
            agent_name: Name of the agent to get or create

        Returns:
            Agent ID
        """
        # Lazily initialize per-instance state bound to the current loop
        if self._agent_creation_lock is None:
            self._agent_creation_lock = asyncio.Lock()
        if self._agent_creation_futures is None:
            self._agent_creation_futures = {}

        # Singleflight pattern: reuse in-flight task without awaiting while holding the lock
        async with self._agent_creation_lock:
            existing_future = self._agent_creation_futures.get(agent_name)
            if existing_future is None:
                future = asyncio.create_task(self._do_get_or_create_agent(agent_name))
                self._agent_creation_futures[agent_name] = future
            else:
                future = existing_future

        try:
            return await future
        finally:
            # Only the creator removes the future
            async with self._agent_creation_lock:
                if self._agent_creation_futures.get(agent_name) is future:
                    self._agent_creation_futures.pop(agent_name, None)

    async def _do_get_or_create_agent(self, agent_name: str) -> str:
        """Actually perform the get-or-create operation for an agent.

        Preferred flow aligned with server semantics:
        1. Try to POST create (server is idempotent and may return 200 or 201)
        2. If POST fails with 400/409 (already exists), GET by name and return

        Args:
            agent_name: Name of the agent to get or create

        Returns:
            Agent ID
        """
        # Step 1: Try to create the agent (server handles get-or-create semantics)
        idempotency_key = f"agent-create:{agent_name}:{uuid.uuid4().hex}"
        max_attempts = 3
        base_delay_seconds = 0.2

        for attempt_index in range(max_attempts):
            try:
                agent_response = await self.agents.create(
                    name=agent_name,
                    description="Agent created by MemFuse client",
                    idempotency_key=idempotency_key,
                )
                if (
                    agent_response
                    and agent_response.get("data")
                    and agent_response["data"].get("agent")
                ):
                    return agent_response["data"]["agent"]["id"]
                raise ValueError(
                    f"Invalid response format when creating agent {agent_name}"
                )

            except MemFuseHTTPError as create_error:
                # 400/409: already exists -> fetch
                if (
                    create_error.status_code in [400, 409]
                    or "already exists" in str(create_error).lower()
                ):
                    # GET fallback with a couple of retries for transient issues
                    for get_attempt in range(2):
                        try:
                            agent_response = await self.agents.get_by_name(agent_name)
                            if (
                                agent_response
                                and agent_response.get("data")
                                and agent_response["data"].get("agents")
                                and len(agent_response["data"]["agents"]) > 0
                            ):
                                return agent_response["data"]["agents"][0]["id"]
                            raise ValueError(
                                f"Agent {agent_name} should exist but was not found"
                            )
                        except MemFuseHTTPError as get_error:
                            # If GET returns 404 immediately after a conflict, wait briefly and retry
                            if get_error.status_code >= 500 and get_attempt < 1:
                                await asyncio.sleep(0.2)
                                continue
                            raise ValueError(
                                f"Failed to get agent {agent_name} after creation conflict. "
                                f"Create error: {create_error}, Get error: {get_error}"
                            ) from get_error
                        except (aiohttp.ClientError, asyncio.TimeoutError):
                            if get_attempt < 1:
                                await asyncio.sleep(0.2)
                                continue
                            raise
                    # Should have returned by now
                    raise ValueError(
                        f"Agent {agent_name} should exist but was not retrievable"
                    )

                # Retry on 5xx for idempotent POST
                if create_error.status_code >= 500 and attempt_index < max_attempts - 1:
                    await asyncio.sleep(base_delay_seconds * (2 ** attempt_index))
                    continue
                # Other creation errors, re-raise
                raise

            except (aiohttp.ClientError, asyncio.TimeoutError):
                # Network errors: retry a few times since POST is idempotent on server
                if attempt_index < max_attempts - 1:
                    await asyncio.sleep(base_delay_seconds * (2 ** attempt_index))
                    continue
                raise

    async def _thread_safe_coro_runner(self, coro):
        """Runs a coroutine with session management, suitable for asyncio.run() in a new thread."""
        await self._ensure_session()
        try:
            return await coro
        finally:
            await self._close_session()


class MemFuse:
    """Synchronous MemFuse client for communicating with the MemFuse server."""

    # Class-level locks for agent creation (sync version)
    _agent_creation_locks = {}
    _agent_creation_lock = threading.Lock()

    def __init__(self, base_url: str = "http://localhost:8765", api_key: Optional[str] = None, timeout: int = 10):
        """Initialize the synchronous MemFuse client.

        Args:
            base_url: URL of the MemFuse server API
            api_key: API key for authentication (optional for local usage)
            timeout: Request timeout in seconds (default: 10)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("MEMFUSE_API_KEY")
        self.timeout = timeout
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
            response = self.sync_session.get(url, timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False

    def _request_sync(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
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

        # Debug logging
        if os.getenv("MEMFUSE_DEBUG") == "1":
            url = f"{self.base_url}{endpoint}"
            logger.debug(f"[MEMFUSE API] {method} {url}")
            if data:
                logger.debug(f"[MEMFUSE API] Request body: {json.dumps(data, indent=2)}")
            if extra_headers:
                logger.debug(f"[MEMFUSE API] Extra headers: {extra_headers}")

        # Perform the request (health check is handled during init)
        try:
            url = f"{self.base_url}{endpoint}"

            response = getattr(self.sync_session, method.lower())(
                url,
                json=data,
                headers=extra_headers,
                timeout=self.timeout,
            )
            response_data = response.json()
            if response.status_code >= 400:
                error_message = response_data.get("message", "Unknown error")
                raise MemFuseHTTPError(f"API request failed: {error_message}", response.status_code, response_data)
            return response_data

        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to MemFuse server at {self.base_url}. "
                "Please make sure the server is running.\n\n"
                "You can start the server with:\n"
                "  poetry run memfuse-core"
            ) from e

    def _check_version_compatibility_sync(self):
        """Check SDK and server version compatibility and display warnings if needed (sync version)."""
        try:
            # Get SDK version
            from . import __version__
            sdk_version = __version__
            
            # If version is placeholder, try to get it from installed package metadata
            if not sdk_version or sdk_version == "{{version}}":
                try:
                    # Try to get version from installed package
                    from importlib.metadata import version, PackageNotFoundError
                    try:
                        sdk_version = version('memfuse')
                    except PackageNotFoundError:
                        # Package not installed via pip, skip version check
                        pass
                except ImportError:
                    # Python < 3.8, try backport
                    try:
                        from importlib_metadata import version, PackageNotFoundError
                        try:
                            sdk_version = version('memfuse')
                        except PackageNotFoundError:
                            pass
                    except ImportError:
                        pass
            
            if not sdk_version or sdk_version == "{{version}}":
                logger.debug("SDK version not available, skipping version check")
                return
            
            # Get server health information
            health_data = self.health.health_check_sync()
            
            # Check compatibility and print warning if needed
            warning = check_version_compatibility(sdk_version, health_data)
            if warning:
                print(warning)
                
        except Exception as e:
            # Don't fail init if version check fails
            logger.debug(f"Version compatibility check failed: {e}")

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
        # Ensure session and validate server health once up front
        self._ensure_sync_session()
        if not self._check_server_health_sync():
            raise ConnectionError(
                f"Cannot connect to MemFuse server at {self.base_url}. "
                "Please make sure the server is running.\n\n"
                "You can start the server with:\n"
                "  poetry run memfuse-core"
            )

        # Check version compatibility
        self._check_version_compatibility_sync()
        
        # Get or create user
        user_name = user
        try:
            user_response = self.users.get_by_name_sync(user_name)
            # If we get here, the user exists
            user_id = user_response["data"]["users"][0]["id"]
        except MemFuseHTTPError as e:
            if e.status_code == 404:
                # User doesn't exist, create it
                user_response = self.users.create_sync(
                    name=user_name,
                    description="User created by MemFuse client"
                )
                user_id = user_response["data"]["user"]["id"]
            else:
                # Re-raise other HTTP errors
                raise

        # Get or create agent using optimized approach
        agent_name = agent or "agent_default"
        agent_id = self._get_or_create_agent_sync(agent_name)

        # Check if session with the given name already exists
        session_name = session
        if session_name:
            try:
                session_response = self.sessions.get_by_name_sync(session_name, user_id=user_id)
                # If we get here, the session exists
                session_data = session_response["data"]["sessions"][0]
                session_id = session_data["id"]
            except MemFuseHTTPError as e:
                if e.status_code == 404:
                    # Session doesn't exist, create it
                    session_response = self.sessions.create_sync(
                        user_id=user_id,
                        agent_id=agent_id,
                        name=session_name
                    )
                    session_data = session_response["data"]["session"]
                    session_id = session_data["id"]
                else:
                    # Re-raise other HTTP errors
                    raise
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

    def _get_or_create_agent_sync(self, agent_name: str) -> str:
        """Get or create an agent with thread-safe locking (sync version).

        This implements the same optimizations as the async version but for sync usage:
        1. Per-agent-name locking to avoid concurrent creation
        2. Fallback handling: if POST fails with 400/409, GET the existing agent
        3. Proper error handling for server-side idempotent behavior

        Args:
            agent_name: Name of the agent to get or create

        Returns:
            Agent ID
        """
        # Get or create a per-agent lock
        with MemFuse._agent_creation_lock:
            if agent_name not in MemFuse._agent_creation_locks:
                MemFuse._agent_creation_locks[agent_name] = threading.Lock()
            agent_lock = MemFuse._agent_creation_locks[agent_name]

        # Use the per-agent lock to serialize creation attempts
        with agent_lock:
            return self._do_get_or_create_agent_sync(agent_name)

    def _do_get_or_create_agent_sync(self, agent_name: str) -> str:
        """Actually perform the get-or-create operation for an agent (sync version).

        Preferred flow aligned with server semantics:
        1. Try to POST create (server is idempotent and may return 200 or 201)
        2. If POST fails with 400/409 (already exists), GET by name and return

        Args:
            agent_name: Name of the agent to get or create

        Returns:
            Agent ID
        """
        # Step 1: Try to create the agent (server handles get-or-create semantics)
        idempotency_key = f"agent-create:{agent_name}:{uuid.uuid4().hex}"
        max_attempts = 3
        base_delay_seconds = 0.2

        import requests
        for attempt_index in range(max_attempts):
            try:
                agent_response = self.agents.create_sync(
                    name=agent_name,
                    description="Agent created by MemFuse client",
                    idempotency_key=idempotency_key,
                )
                if (
                    agent_response
                    and agent_response.get("data")
                    and agent_response["data"].get("agent")
                ):
                    return agent_response["data"]["agent"]["id"]
                raise ValueError(
                    f"Invalid response format when creating agent {agent_name}"
                )

            except MemFuseHTTPError as create_error:
                if (
                    create_error.status_code in [400, 409]
                    or "already exists" in str(create_error).lower()
                ):
                    # GET fallback with short retry for transient failures
                    for get_attempt in range(2):
                        try:
                            agent_response = self.agents.get_by_name_sync(agent_name)
                            if (
                                agent_response
                                and agent_response.get("data")
                                and agent_response["data"].get("agents")
                                and len(agent_response["data"]["agents"]) > 0
                            ):
                                return agent_response["data"]["agents"][0]["id"]
                            raise ValueError(
                                f"Agent {agent_name} should exist but was not found"
                            )
                        except MemFuseHTTPError as get_error:
                            if get_error.status_code >= 500 and get_attempt < 1:
                                time.sleep(0.2)
                                continue
                            raise ValueError(
                                f"Failed to get agent {agent_name} after creation conflict. "
                                f"Create error: {create_error}, Get error: {get_error}"
                            ) from get_error
                        except requests.exceptions.RequestException:
                            if get_attempt < 1:
                                time.sleep(0.2)
                                continue
                            raise
                    raise ValueError(
                        f"Agent {agent_name} should exist but was not retrievable"
                    )

                if create_error.status_code >= 500 and attempt_index < max_attempts - 1:
                    time.sleep(base_delay_seconds * (2 ** attempt_index))
                    continue
                raise

            except requests.exceptions.RequestException:
                if attempt_index < max_attempts - 1:
                    time.sleep(base_delay_seconds * (2 ** attempt_index))
                    continue
                raise

    def close(self):
        """Close the client and its underlying sessions."""
        self._close_sync_session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
