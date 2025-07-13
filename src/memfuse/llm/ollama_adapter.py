# src/memfuse/llm/ollama_adapter.py
from __future__ import annotations
from typing import Any, Dict, List, Callable, Coroutine
import inspect, functools
import logging

from ollama import Client, AsyncClient # type: ignore
# Ollama's response for chat is a dict, let's define a type alias if specific structure is known
# For now, using Any. Example response: {'model': 'llama2', 'created_at': '...', 'message': {'role': 'assistant', 'content': '...'}, ...}
OllamaChatResponse = Dict[str, Any]
OllamaAsyncChatResponse = Coroutine[Any, Any, OllamaChatResponse] # for async streaming
OllamaChatStreamChunk = Dict[str, Any] # Example chunk: {'model': 'llama2', 'created_at': '...', 'message': {'role': 'assistant', 'content': '...'}, 'done': False}

from memfuse import Memory
from memfuse.prompts import PromptContext, PromptFormatter

# Set up logger for this module
logger = logging.getLogger(__name__)

def _wrap_chat(
    chat_fn: Callable[..., OllamaChatResponse | OllamaAsyncChatResponse], # Synchronous or Asynchronous chat function
    memory: Memory,
    is_async: bool = False,
) -> Callable[..., OllamaChatResponse | OllamaAsyncChatResponse]:
    """
    Returns a function with the *exact same* signature as `chat_fn`
    but that transparently injects conversational memory.
    Works for both ollama.Client.chat and ollama.AsyncClient.chat.
    """
    sig = inspect.signature(chat_fn)

    # The wrapper needs to be async if the original function is async
    if is_async:
        @functools.wraps(chat_fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any: # Actual return type depends on stream or not
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # ------- 1. Extract the messages list & augment with history --------
            query_messages: List[Dict[str, Any]] = bound.arguments["messages"]

            # ------- 2. Get the last n messages (Commented out, kept from original) -------
            # max_chat_history = memory.max_chat_history
            # retrieved_chat_history = await memory.alist_messages( # Assuming async memory operation
            #     session_id=memory.session_id, # session_id needs to be available
            #     limit=max_chat_history,
            # )
            # TODO: Remove this defensive handling once the server API is fixed to consistently return
            # the same format for memory.list_messages() - it should always return {"data": {"messages": [...]}}
            # Currently it sometimes returns a list directly, causing "list indices must be integers or slices, not str" errors
            # When re-enabling memory operations, add defensive handling like other adapters:
            # if isinstance(retrieved_chat_history, dict) and "data" in retrieved_chat_history:
            #     chat_history = retrieved_chat_history["data"]["messages"][::-1]
            # elif isinstance(retrieved_chat_history, list):
            #     chat_history = retrieved_chat_history[::-1]
            # else:
            #     chat_history = []

            # ------- 3. Retrieve memories (Commented out, kept from original) -------
            query_string = PromptFormatter.messages_to_query(query_messages)
            # query_response = await memory.aquery(query_string) # Assuming async memory operation
            # retrieved_memories = query_response["data"]["results"]
            retrieved_memories = None # Placeholder

            # ------- 4. Compose the prompt --------------------------
            prompt_context = PromptContext(
                query_messages=query_messages,
                retrieved_memories=retrieved_memories, # Using None for now
                retrieved_chat_history=None, # Using None for now
                max_chat_history=20, # Placeholder
            )
            full_msg_list = prompt_context.compose_openai()
            bound.arguments["messages"] = full_msg_list

            # ------- 5. Forward the call to the real chat function ---------------------
            # Ollama's chat can be streaming or non-streaming.
            # We need to handle both cases.
            is_streaming = bound.arguments.get("stream", False)

            if is_streaming:
                # The wrapped function will also become an async generator
                async def stream_wrapper() -> Any: # OllamaAsyncChatStream:
                    response_stream = chat_fn(*bound.args, **bound.kwargs)

                    assistant_response_content = ""
                    full_response_message_dict = {}

                    async for chunk in response_stream: # type: ignore
                        # Example chunk: {'model': 'llama2', 'created_at': '...', 'message': {'role': 'assistant', 'content': 'Why is the sky blue?'}, 'done': False}
                        # Final chunk:   {'model': 'llama2', 'created_at': '...', 'message': {'role': 'assistant', 'content': ''}, 'done': True, ... metrics ...}
                        if chunk.get("message") and "content" in chunk["message"]:
                            assistant_response_content += chunk["message"]["content"]
                        if chunk.get("done") and chunk.get("message"): # Capture the final message structure for persistence
                            full_response_message_dict = chunk.get("message", {}) # role, content (which is empty for final chunk)
                        yield chunk

                    messages_to_persist = list(query_messages)
                    if assistant_response_content: # Persist the accumulated content
                         messages_to_persist.append({
                            "role": full_response_message_dict.get("role", "assistant"), # Role from final chunk or default
                            "content": assistant_response_content.strip()
                        })

                    if messages_to_persist:
                        # result = await memory.aadd(messages=messages_to_persist) # Assuming async memory.aadd
                        result = await memory.add(messages=messages_to_persist) # Using async add
                        if result and result.get("data") and result["data"].get("message_ids"):
                            logger.info(f"Persisted (async stream) Ollama message IDs: {result['data']['message_ids']}")
                        else:
                            logger.error("Failed to persist (async stream) Ollama messages or no message IDs returned.")

                return stream_wrapper() # Return the async generator
            else:
                response = await chat_fn(*bound.args, **bound.kwargs) # type: ignore

                # ------- 6. Persist *only* the new interaction ----------------------
                messages_to_persist = list(query_messages)

                if response and response.get("message") and response["message"].get("content"):
                    assistant_message = response["message"]
                    messages_to_persist.append({
                        "role": assistant_message.get("role", "assistant"),
                        "content": assistant_message.get("content")
                    })

                if messages_to_persist:
                    # result = await memory.aadd(messages=messages_to_persist) # Assuming async memory.aadd
                    result = await memory.add(messages=messages_to_persist) # Using async add
                    if result and result.get("data") and result["data"].get("message_ids"):
                        logger.info(f"Persisted (async) Ollama message IDs: {result['data']['message_ids']}")
                    else:
                        logger.error("Failed to persist (async) Ollama messages or no message IDs returned.")
                return response

        # Replace wrapper's __signature__ so help(), IDEs, and type checkers show the original signature.
        wrapper.__signature__ = sig # type: ignore[attr-defined]
        return wrapper
    else: # Synchronous wrapper
        @functools.wraps(chat_fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any: # Actual return type depends on stream or not
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            query_messages: List[Dict[str, Any]] = bound.arguments["messages"]
            query_string = PromptFormatter.messages_to_query(query_messages)
            retrieved_memories = None
            
            # TODO: Remove this defensive handling once the server API is fixed to consistently return
            # the same format for memory.list_messages() - it should always return {"data": {"messages": [...]}}
            # Currently it sometimes returns a list directly, causing "list indices must be integers or slices, not str" errors
            # When implementing memory operations, add defensive handling like other adapters

            prompt_context = PromptContext(
                query_messages=query_messages,
                retrieved_memories=retrieved_memories,
                retrieved_chat_history=None,
                max_chat_history=20,
            )
            full_msg_list = prompt_context.compose_openai()
            bound.arguments["messages"] = full_msg_list

            is_streaming = bound.arguments.get("stream", False)

            if is_streaming:
                def stream_wrapper() -> Any: # OllamaChatStream
                    response_stream = chat_fn(*bound.args, **bound.kwargs)
                    assistant_response_content = ""
                    full_response_message_dict = {}

                    for chunk in response_stream: # type: ignore
                        if chunk.get("message") and "content" in chunk["message"]:
                            assistant_response_content += chunk["message"]["content"]
                        if chunk.get("done") and chunk.get("message"):
                            full_response_message_dict = chunk.get("message", {})
                        yield chunk

                    messages_to_persist = list(query_messages)
                    if assistant_response_content:
                         messages_to_persist.append({
                            "role": full_response_message_dict.get("role", "assistant"),
                            "content": assistant_response_content.strip()
                        })

                    if messages_to_persist:
                        result = memory.add(messages=messages_to_persist)
                        if result and result.get("data") and result["data"].get("message_ids"):
                            logger.info(f"Persisted (sync stream) Ollama message IDs: {result['data']['message_ids']}")
                        else:
                            logger.error("Failed to persist (sync stream) Ollama messages or no message IDs returned.")
                return stream_wrapper()
            else:
                response = chat_fn(*bound.args, **bound.kwargs)

                messages_to_persist = list(query_messages)
                if response and response.get("message") and response["message"].get("content"):
                    assistant_message = response["message"]
                    messages_to_persist.append({
                        "role": assistant_message.get("role", "assistant"),
                        "content": assistant_message.get("content")
                    })

                if messages_to_persist:
                    result = memory.add(messages=messages_to_persist)
                    if result and result.get("data") and result["data"].get("message_ids"):
                        logger.info(f"Persisted (sync) Ollama message IDs: {result['data']['message_ids']}")
                    else:
                        logger.error("Failed to persist (sync) Ollama messages or no message IDs returned.")
                return response

        wrapper.__signature__ = sig # type: ignore[attr-defined]
        return wrapper


class MemOllama(Client):
    """
    Public adapter that *looks identical* to `ollama.Client`.
    Memory is applied to the `chat` method.
    """
    def __init__(
        self,
        host: str | None = None,
        memory: Memory | None = None,
        **kwargs: Any, # To catch other httpx.Client params like timeout, headers, etc.
    ):
        super().__init__(host=host, **kwargs)
        self.memory_instance = memory # Renamed to avoid conflict if 'memory' is a valid Client kwarg

        if self.memory_instance:
            original_chat = self.chat
            self.chat = _wrap_chat(original_chat, self.memory_instance, is_async=False)


class AsyncMemOllama(AsyncClient):
    """
    Async version of MemOllama, an adapter for `ollama.AsyncClient`.
    """
    def __init__(
        self,
        host: str | None = None,
        memory: Memory | None = None,
        **kwargs: Any,
    ):
        super().__init__(host=host, **kwargs)
        self.memory_instance = memory

        if self.memory_instance:
            original_chat = self.chat
            # The original AsyncClient.chat is already an async def method or returns a coroutine/async generator
            self.chat = _wrap_chat(original_chat, self.memory_instance, is_async=True) # type: ignore[assignment] 