#!/usr/bin/env python
"""
Unified LLM client supporting multiple providers (OpenAI and Gemini).
Supports both synchronous and asynchronous operations.
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass

# OpenAI imports
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Gemini imports
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)


class Provider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class LLMResponse:
    """Unified response structure for all providers."""
    text: str
    provider: Provider
    model: str
    usage: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None


class UnifiedLLM:
    """
    Unified LLM client that supports multiple providers.
    
    Supports:
    - OpenAI (using OPENAI_API_KEY and OPENAI_BASE_URL)
    - Gemini (using GEMINI_API_KEY and GEMINI_BASE_URL)
    - Both sync and async operations
    """
    
    def __init__(self, provider: Union[str, Provider], model: Optional[str] = None):
        """
        Initialize the unified LLM client.
        
        Args:
            provider: The LLM provider to use ("openai" or "gemini")
            model: Optional model name. If not provided, uses defaults.
        """
        self.provider = Provider(provider) if isinstance(provider, str) else provider
        self.model = model
        self.client = None
        self.async_client = None
        
        # Initialize the appropriate clients
        if self.provider == Provider.OPENAI:
            self._init_openai_client()
        elif self.provider == Provider.GEMINI:
            self._init_gemini_client()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_openai_client(self):
        """Initialize OpenAI sync and async clients with environment variables."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
        
        # Set default model if not provided
        if not self.model:
            self.model = "gpt-4o"
        
        logger.info(f"Initialized OpenAI sync and async clients with model: {self.model}")
        if base_url:
            logger.info(f"Using custom base URL: {base_url}")
    
    def _init_gemini_client(self):
        """Initialize Gemini sync and async clients with environment variables."""
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI library not available. Install with: pip install google-genai")
        
        api_key = os.getenv("GEMINI_API_KEY")
        base_url = os.getenv("GEMINI_BASE_URL")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        try:
            # Create client with API key
            client_kwargs = {"api_key": api_key}
            
            # Handle custom base URL through http_options if provided
            if base_url:
                from google.genai import types
                http_options = types.HttpOptions(base_url=base_url)
                client_kwargs["http_options"] = http_options
                logger.info(f"Using custom base URL: {base_url}")
            
            self.client = genai.Client(**client_kwargs)
            # Gemini client supports both sync and async operations through the same client
            # The async operations are accessed via client.aio
            self.async_client = self.client
            
            # Set default model if not provided
            if not self.model:
                self.model = "gemini-2.0-flash-001"
            
            logger.info(f"Initialized Gemini sync and async clients with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def generate_text(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text using the configured provider (synchronous).
        
        Args:
            prompt: The text prompt to send to the model
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse: Unified response object
        """
        if self.provider == Provider.OPENAI:
            return self._generate_openai(prompt, **kwargs)
        elif self.provider == Provider.GEMINI:
            return self._generate_gemini(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def agenerate_text(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text using the configured provider (asynchronous).
        
        Args:
            prompt: The text prompt to send to the model
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse: Unified response object
        """
        if self.provider == Provider.OPENAI:
            return await self._agenerate_openai(prompt, **kwargs)
        elif self.provider == Provider.GEMINI:
            return await self._agenerate_gemini(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_openai(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using OpenAI."""
        try:
            # Extract OpenAI-specific parameters
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )
            
            text = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None
            
            return LLMResponse(
                text=text,
                provider=self.provider,
                model=self.model,
                usage=usage,
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def _agenerate_openai(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using OpenAI (asynchronous)."""
        try:
            # Extract OpenAI-specific parameters
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )
            
            text = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None
            
            return LLMResponse(
                text=text,
                provider=self.provider,
                model=self.model,
                usage=usage,
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"OpenAI async generation failed: {e}")
            raise
    
    def _generate_gemini(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Gemini."""
        try:
            # Extract Gemini-specific parameters
            temperature = kwargs.get("temperature", 0.7)
            max_output_tokens = kwargs.get("max_tokens", kwargs.get("max_output_tokens", 1000))
            
            # Build generation config
            config_kwargs = {}
            if "temperature" in kwargs or temperature != 0.7:
                config_kwargs["temperature"] = temperature
            if "max_tokens" in kwargs or "max_output_tokens" in kwargs:
                config_kwargs["max_output_tokens"] = max_output_tokens
            
            # Generate content
            if config_kwargs:
                from google.genai import types
                config = types.GenerateContentConfig(**config_kwargs)
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
            
            text = response.text
            
            # Extract usage information if available
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            return LLMResponse(
                text=text,
                provider=self.provider,
                model=self.model,
                usage=usage,
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    async def _agenerate_gemini(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text using Gemini (asynchronous)."""
        try:
            # Extract Gemini-specific parameters
            temperature = kwargs.get("temperature", 0.7)
            max_output_tokens = kwargs.get("max_tokens", kwargs.get("max_output_tokens", 1000))
            
            # Build generation config
            config_kwargs = {}
            if "temperature" in kwargs or temperature != 0.7:
                config_kwargs["temperature"] = temperature
            if "max_tokens" in kwargs or "max_output_tokens" in kwargs:
                config_kwargs["max_output_tokens"] = max_output_tokens
            
            # Generate content asynchronously
            if config_kwargs:
                from google.genai import types
                config = types.GenerateContentConfig(**config_kwargs)
                response = await self.async_client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config
                )
            else:
                response = await self.async_client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
            
            text = response.text
            
            # Extract usage information if available
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            return LLMResponse(
                text=text,
                provider=self.provider,
                model=self.model,
                usage=usage,
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Gemini async generation failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if the provider client is properly initialized."""
        return self.client is not None
    
    @classmethod
    def list_available_providers(cls) -> Dict[str, bool]:
        """List which providers are available based on installed dependencies."""
        return {
            "openai": OPENAI_AVAILABLE,
            "gemini": GEMINI_AVAILABLE
        }