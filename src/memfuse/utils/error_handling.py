"""Error handling utilities for MemFuse client."""

import asyncio
import functools
import sys
from typing import Callable, TypeVar, Any, Awaitable, Dict

T = TypeVar('T')


class MemFuseHTTPError(Exception):
    """Exception raised for HTTP errors from the MemFuse API."""
    
    def __init__(self, message: str, status_code: int, response_data: Dict[str, Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


def handle_server_connection(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorator to handle server connection errors.
    
    This decorator catches ConnectionError exceptions and provides a helpful
    error message to the user, suggesting how to start the server.
    
    Args:
        func: The async function to decorate
        
    Returns:
        The decorated function
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return await func(*args, **kwargs)
        except ConnectionError as e:
            print(f"\nConnection Error: {e}")
            print("\nPlease make sure the MemFuse server is running. You can start it with:")
            print("  poetry run memfuse-core")
            sys.exit(1)
    return wrapper


async def run_with_error_handling(coro: Awaitable[T]) -> T:
    """Run a coroutine with error handling.
    
    This function runs the given coroutine and handles any ConnectionError
    exceptions, providing a helpful error message to the user.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
    """
    try:
        return await coro
    except ConnectionError as e:
        print(f"\nConnection Error: {e}")
        print("\nPlease make sure the MemFuse server is running. You can start it with:")
        print("  poetry run memfuse-core")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


def run_async(coro: Awaitable[T]) -> T:
    """Run an async function in the event loop.
    
    This function runs the given coroutine in the event loop and handles
    any ConnectionError exceptions, providing a helpful error message to the user.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
    """
    try:
        return asyncio.run(run_with_error_handling(coro))
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
