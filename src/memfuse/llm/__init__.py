# src/memfuse/llm/__init__.py
import os
from typing import TYPE_CHECKING

# Choose which backend to expose under the public name `OpenAI`
if os.getenv("MEMFUSE_OPENAI_PASSTHROUGH") == "1":
    # 1️⃣  Optional escape hatch – give users the *real* SDK
    from openai import OpenAI as _OpenAICore       # type: ignore
    from openai import AsyncOpenAI as _AsyncCore
else:
    # 2️⃣  Default to your memory‑augmented adapter
    from .openai_adapter import MemOpenAI as _OpenAICore
    from .openai_adapter import AsyncMemOpenAI as _AsyncCore

# Choose which backend to expose under the public name `Anthropic`
if os.getenv("MEMFUSE_ANTHROPIC_PASSTHROUGH") == "1":
    # 1️⃣  Optional escape hatch – give users the *real* SDK
    from anthropic import Anthropic as _AnthropicCore       # type: ignore
    from anthropic import AsyncAnthropic as _AsyncAnthropicCore
else:
    # 2️⃣  Default to your memory‑augmented adapter
    from .anthropic_adapter import MemAnthropic as _AnthropicCore
    from .anthropic_adapter import AsyncMemAnthropic as _AsyncAnthropicCore

# Choose which backend to expose under the public name `GeminiClient`
if os.getenv("MEMFUSE_GEMINI_PASSTHROUGH") == "1":
    # 1️⃣  Optional escape hatch – give users the *real* SDK
    from google.genai import Client as _GeminiClientCore       # type: ignore
    # For async, we use the same Client but users can access .aio for async operations
    from google.genai import Client as _AsyncGeminiClientCore  # type: ignore
else:
    # 2️⃣  Default to your memory‑augmented adapter
    from .gemini_adapter import MemorableGoogleGenerativeAI as _GeminiClientCore
    from .gemini_adapter import AsyncMemorableGoogleGenerativeAI as _AsyncGeminiClientCore

# Import other adapters normally
from .ollama_adapter import MemOllama as _MemOllamaCore
from .ollama_adapter import AsyncMemOllama as _AsyncMemOllamaCore

# Public names
OpenAI: type[_OpenAICore] = _OpenAICore
AsyncOpenAI: type[_AsyncCore] = _AsyncCore
Anthropic: type[_AnthropicCore] = _AnthropicCore
AsyncAnthropic: type[_AsyncAnthropicCore] = _AsyncAnthropicCore
GeminiClient: type[_GeminiClientCore] = _GeminiClientCore
AsyncGeminiClient: type[_AsyncGeminiClientCore] = _AsyncGeminiClientCore
MemOllama: type[_MemOllamaCore] = _MemOllamaCore
AsyncMemOllama: type[_AsyncMemOllamaCore] = _AsyncMemOllamaCore

__all__ = [
    "OpenAI", "AsyncOpenAI", 
    "Anthropic", "AsyncAnthropic", 
    "GeminiClient", "AsyncGeminiClient",
    "MemOllama", "AsyncMemOllama"
]

# Optional: help static analysers
if TYPE_CHECKING:  # pragma: no cover
    from openai import OpenAI as _OpenAIStub
    from openai import AsyncOpenAI as _AsyncStub
    from anthropic import Anthropic as _AnthropicStub
    from anthropic import AsyncAnthropic as _AsyncAnthropicStub
    from google.genai import Client as _GeminiClientStub
    from ollama import Client as _OllamaClientStub

    OpenAI = _OpenAIStub
    AsyncOpenAI = _AsyncStub
    Anthropic = _AnthropicStub
    AsyncAnthropic = _AsyncAnthropicStub
    GeminiClient = _GeminiClientStub
    AsyncGeminiClient = _GeminiClientStub  # Same as sync for type checking
    MemOllama = _OllamaClientStub
    AsyncMemOllama = _OllamaClientStub