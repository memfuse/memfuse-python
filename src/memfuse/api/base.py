"""Base API client with shared endpoint configuration."""

from typing import Dict, List, Optional, Any, Callable
import functools


class BaseApi:
    """Base API client that generates sync and async methods from endpoint configurations."""

    ENDPOINTS: Dict[str, Dict[str, Any]] = {}

    def __init__(self, client):
        """Initialize the API client.

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

    def _make_request_async(self, endpoint_key: str, url_params: Dict[str, Any] = None, body: Dict[str, Any] = None):
        """Make an async request using endpoint configuration."""
        url_params = url_params or {}
        method = self._get_method(endpoint_key)
        url = self._build_url(endpoint_key, **url_params)
        
        if body is not None:
            return self.client._request(method, url, body)
        else:
            return self.client._request(method, url)

    def _make_request_sync(self, endpoint_key: str, url_params: Dict[str, Any] = None, body: Dict[str, Any] = None):
        """Make a sync request using endpoint configuration."""
        url_params = url_params or {}
        method = self._get_method(endpoint_key)
        url = self._build_url(endpoint_key, **url_params)
        
        if body is not None:
            return self.client._request_sync(method, url, body)
        else:
            return self.client._request_sync(method, url)

    @classmethod
    def _create_method_pair(cls, endpoint_key: str, doc_template: str = None):
        """Create both async and sync versions of a method.
        
        This is a class method that can be used to dynamically create method pairs.
        """
        def async_method(self, **kwargs):
            url_params = kwargs.pop('_url_params', {})
            body = kwargs.pop('_body', None)
            if kwargs:  # If there are remaining kwargs, they become the body
                body = kwargs
            return self._make_request_async(endpoint_key, url_params, body)

        def sync_method(self, **kwargs):
            url_params = kwargs.pop('_url_params', {})
            body = kwargs.pop('_body', None) 
            if kwargs:  # If there are remaining kwargs, they become the body
                body = kwargs
            return self._make_request_sync(endpoint_key, url_params, body)

        # Set docstrings if provided
        if doc_template:
            async_method.__doc__ = doc_template
            sync_method.__doc__ = doc_template + " (sync version)"

        return async_method, sync_method 