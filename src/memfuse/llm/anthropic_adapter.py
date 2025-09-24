# src/memfuse/llm/anthropic_adapter.py
from __future__ import annotations
from typing import Any, Dict, List, Callable, Optional
import inspect, functools
import logging
import contextvars
import time

from anthropic import Anthropic, AsyncAnthropic

from memfuse import Memory
from memfuse.prompts import PromptContext, PromptFormatter

# Set up logger for this module
logger = logging.getLogger(__name__)

# Context variables to capture query responses, timing, and prompt context for debugging (invisible to SDK users)
_debug_query_response: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar('debug_query_response', default=None)
_debug_query_time: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar('debug_query_time', default=None)
_debug_prompt_context: contextvars.ContextVar[Optional[PromptContext]] = contextvars.ContextVar('debug_prompt_context', default=None)


def _extract_text_from_content(content: Any) -> str:
    """Helper function to extract text from various content formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text += item.get("text", "") + " "
        return text.strip()
    return ""


def _wrap_create(
    create_fn: Callable[..., Any],
    memory: Memory,
) -> Callable[..., Any]:
    """
    Returns a function with the *exact same* signature as `create_fn`
    but that transparently injects conversational memory.
    """
    sig = inspect.signature(create_fn)

    @functools.wraps(create_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # signature replaced below
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # ------- 1. Extract the messages list & augment with history --------
        # NB: v1 SDK uses keyword‐only 'messages'
        query_messages: List[Dict[str, Any]] = bound.arguments["messages"]
        
        # ------- 2. Get the last n messages ----------------------------------
        max_chat_history = memory.max_chat_history

        in_buffer_chat_history = memory.list_messages(
            limit=max_chat_history,
            buffer_only=True,
        )

        in_buffer_messages_length = len(in_buffer_chat_history["data"]["messages"])

        if in_buffer_messages_length < max_chat_history:
            in_db_chat_history = memory.list_messages(
                limit=max_chat_history - in_buffer_messages_length,
                buffer_only=False,
            )
        else:
            in_db_chat_history = []

        # TODO: Remove this defensive handling once the server API is fixed to consistently return
        # the same format for memory.list_messages() - it should always return {"data": {"messages": [...]}}
        # Currently it sometimes returns a list directly, causing "list indices must be integers or slices, not str" errors
        # Handle both dict and list formats for chat history
        if isinstance(in_db_chat_history, dict) and "data" in in_db_chat_history:
            db_messages = in_db_chat_history["data"]["messages"]
        elif isinstance(in_db_chat_history, list):
            db_messages = in_db_chat_history
        else:
            db_messages = []

        chat_history = [{"role": message["role"], "content": message["content"]} for message in db_messages[::-1]] + [{"role": message["role"], "content": message["content"]} for message in in_buffer_chat_history["data"]["messages"][::-1]]

        # ------- 3. Retrieve memories ---------------------------------------
        # Convert Anthropic formatted messages to a string for querying
        query_string = PromptFormatter.messages_to_query(chat_history + query_messages)
        
        # Query memories if we have a non-empty query string
        retrieved_memories = []
        query_response = None
        if query_string.strip():
            # Measure query time
            query_start_time = time.perf_counter()
            query_response = memory.query_session(query_string)
            query_end_time = time.perf_counter()
            query_duration = query_end_time - query_start_time
            
            # Store query response and timing for debugging access (invisible to normal users)
            _debug_query_response.set(query_response)
            _debug_query_time.set(query_duration)

            # Compact diagnostics for the new envelope
            if isinstance(query_response, dict):
                status = query_response.get("status")
                code = query_response.get("code")
                message = query_response.get("message")
                errors = query_response.get("errors")
                total = (
                    query_response.get("data", {}).get("total")
                    if isinstance(query_response.get("data"), dict)
                    else None
                )
                logger.info(f"Query status={status} code={code} total={total} message={message}")
                if errors:
                    logger.warning(f"Query errors: {errors}")

            # Safely extract results with fallback
            retrieved_memories = (
                query_response.get("data", {}).get("results", [])
                if isinstance(query_response, dict)
                else []
            )
            logger.info(f"Query took {query_duration * 1000:.2f} ms; {len(retrieved_memories)} memories")
        
        # ------- 4. Compose the prompt for Anthropic API format --------------------------
        prompt_context = PromptContext(
            query_messages=query_messages,
            retrieved_memories=retrieved_memories,
            retrieved_chat_history=chat_history,
            max_chat_history=max_chat_history,
        )
        
        # Store prompt context for debugging access (invisible to normal users)
        _debug_prompt_context.set(prompt_context)

        # Get system message and formatted messages for Anthropic
        system_prompt, anthropic_messages = prompt_context.compose_for_anthropic()
        
        # Update the bound arguments
        bound.arguments["messages"] = anthropic_messages
        
        # Add system prompt if it's not already provided in the kwargs
        if "system" not in bound.arguments:
            bound.arguments["system"] = system_prompt

        # ------- 5. Forward the call to the real create ---------------------
        response = create_fn(*bound.args, **bound.kwargs)

        # ------- 6. Persist *only* the new interaction ----------------------
        # Prepare the messages to persist - convert Anthropic format back to standard format
        messages_to_persist = []
        
        # Add user messages
        for msg in query_messages:
            if msg.get("role") == "user":
                content = _extract_text_from_content(msg.get("content", ""))
                if content:
                    messages_to_persist.append({
                        "role": "user",
                        "content": content
                    })
        
        # Add assistant response
        if response and response.content and isinstance(response.content, list):
            for content_item in response.content:
                if hasattr(content_item, 'text') and content_item.text:
                    messages_to_persist.append({
                        "role": "assistant",
                        "content": content_item.text
                    })
            
        if messages_to_persist: # Only add if there's something to add
            result = memory.add(messages=messages_to_persist)
            if result and result.get("data") and result["data"].get("message_ids"):
                message_ids = result["data"]["message_ids"]
                logger.info(f"Persisted message IDs: {message_ids}")
            else:
                logger.error("Failed to persist messages or no message IDs returned.")
        else:
            logger.info("No messages to persist for this interaction.")
        
        return response

    # ★ Replace wrapper's __signature__ so help(), IDEs, and type checkers
    #   all show the *original* Anthropic signature.
    wrapper.__signature__ = sig  # type: ignore[attr-defined]
    return wrapper


def _wrap_create_async(
    create_fn: Callable[..., Any],
    memory: Memory,
) -> Callable[..., Any]:
    """
    Async version of _wrap_create that properly handles async memory operations.
    """
    sig = inspect.signature(create_fn)

    @functools.wraps(create_fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:  # signature replaced below
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # ------- 1. Extract the messages list & augment with history --------
        query_messages: List[Dict[str, Any]] = bound.arguments["messages"]
        
        # ------- 2. Get the last n messages ----------------------------------
        max_chat_history = memory.max_chat_history

        in_buffer_chat_history = await memory.list_messages(
            limit=max_chat_history,
            buffer_only=True,
        )

        in_buffer_messages_length = len(in_buffer_chat_history["data"]["messages"])

        if in_buffer_messages_length < max_chat_history:
            in_db_chat_history = await memory.list_messages(
                limit=max_chat_history - in_buffer_messages_length,
                buffer_only=False,
            )
        else:
            in_db_chat_history = []

        # TODO: Remove this defensive handling once the server API is fixed to consistently return
        # the same format for memory.list_messages() - it should always return {"data": {"messages": [...]}}
        # Currently it sometimes returns a list directly, causing "list indices must be integers or slices, not str" errors
        # Handle both dict and list formats for chat history
        if isinstance(in_db_chat_history, dict) and "data" in in_db_chat_history:
            db_messages = in_db_chat_history["data"]["messages"]
        elif isinstance(in_db_chat_history, list):
            db_messages = in_db_chat_history
        else:
            db_messages = []

        chat_history = [{"role": message["role"], "content": message["content"]} for message in db_messages[::-1]] + [{"role": message["role"], "content": message["content"]} for message in in_buffer_chat_history["data"]["messages"][::-1]]

        # ------- 3. Retrieve memories ---------------------------------------
        query_string = PromptFormatter.messages_to_query(chat_history + query_messages)
        
        # Query memories if we have a non-empty query string (ASYNC)
        retrieved_memories = []
        query_response = None
        if query_string.strip():
            # Measure query time
            query_start_time = time.perf_counter()
            query_response = await memory.query_session(query_string)
            query_end_time = time.perf_counter()
            query_duration = query_end_time - query_start_time
            
            # Store query response and timing for debugging access (invisible to normal users)
            _debug_query_response.set(query_response)
            _debug_query_time.set(query_duration)

            # Compact diagnostics for the new envelope
            if isinstance(query_response, dict):
                status = query_response.get("status")
                code = query_response.get("code")
                message = query_response.get("message")
                errors = query_response.get("errors")
                total = (
                    query_response.get("data", {}).get("total")
                    if isinstance(query_response.get("data"), dict)
                    else None
                )
                logger.info(f"Query status={status} code={code} total={total} message={message}")
                if errors:
                    logger.warning(f"Query errors: {errors}")

            # Safely extract results with fallback
            retrieved_memories = (
                query_response.get("data", {}).get("results", [])
                if isinstance(query_response, dict)
                else []
            )
            logger.info(f"Query took {query_duration * 1000:.2f} ms; {len(retrieved_memories)} memories")
        
        # ------- 4. Compose the prompt for Anthropic API format --------------------------
        prompt_context = PromptContext(
            query_messages=query_messages,
            retrieved_memories=retrieved_memories,
            retrieved_chat_history=chat_history,
            max_chat_history=max_chat_history,
        )
        
        # Store prompt context for debugging access (invisible to normal users)
        _debug_prompt_context.set(prompt_context)

        # Get system message and formatted messages for Anthropic
        system_prompt, anthropic_messages = prompt_context.compose_for_anthropic()
        
        # Update the bound arguments
        bound.arguments["messages"] = anthropic_messages
        
        # Add system prompt if it's not already provided in the kwargs
        if "system" not in bound.arguments:
            bound.arguments["system"] = system_prompt

        # ------- 5. Forward the call to the real create ---------------------
        response = await create_fn(*bound.args, **bound.kwargs)

        # ------- 6. Persist *only* the new interaction ----------------------
        messages_to_persist = []
        
        # Add user messages
        for msg in query_messages:
            if msg.get("role") == "user":
                content = _extract_text_from_content(msg.get("content", ""))
                if content:
                    messages_to_persist.append({
                        "role": "user",
                        "content": content
                    })
        
        # Add assistant response
        if response and response.content and isinstance(response.content, list):
            for content_item in response.content:
                if hasattr(content_item, 'text') and content_item.text:
                    messages_to_persist.append({
                        "role": "assistant",
                        "content": content_item.text
                    })
            
        if messages_to_persist: # Only add if there's something to add (ASYNC)
            result = await memory.add(messages=messages_to_persist)
            if result and result.get("data") and result["data"].get("message_ids"):
                message_ids = result["data"]["message_ids"]
                logger.info(f"Persisted async message IDs: {message_ids}")
            else:
                logger.error("Failed to persist async messages or no message IDs returned.")
        else:
            logger.info("No messages to persist for this interaction.")
        
        return response

    # ★ Replace wrapper's __signature__ so help(), IDEs, and type checkers
    #   all show the *original* Anthropic signature.
    wrapper.__signature__ = sig  # type: ignore[attr-defined]
    return wrapper


class MemAnthropic(Anthropic):
    """
    Public adapter that *looks identical* to `anthropic.Anthropic`.
    Memory is applied only to messages for now, but the pattern
    is reusable for other methods.
    """

    def __init__(
        self,
        *args: Any,
        memory: Memory | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.memory = memory

        # ---- dynamically monkey‑patch the Messages.create method ----
        original_create = self.messages.create
        self.messages.create = _wrap_create(original_create, self.memory)


class AsyncMemAnthropic(AsyncAnthropic):
    """Async version with identical trick."""

    def __init__(
        self,
        *args: Any,
        memory: Memory | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._mem = memory

        original_create = self.messages.create
        self.messages.create = _wrap_create_async(original_create, self._mem)


# Debug utilities (for benchmarks and debugging)
def get_last_query_response() -> Optional[Dict[str, Any]]:
    """
    Get the last query response for debugging purposes.
    
    This function allows access to the intermediate query response from memory.query_session()
    that occurs within the Anthropic adapter. This is useful for benchmarks and debugging to 
    inspect what memories were retrieved.
    
    Returns:
        The last query response dict if available, None otherwise.
        
    Note:
        This is intended for debugging/benchmarking use only and should not be used
        in production code as it relies on internal implementation details.
    """
    return _debug_query_response.get(None)


def get_last_query_time() -> Optional[float]:
    """
    Get the last query timing for debugging purposes.
    
    Returns:
        The last query duration in seconds if available, None otherwise.
        
    Note:
        This is intended for debugging/benchmarking use only and should not be used
        in production code as it relies on internal implementation details.
    """
    return _debug_query_time.get(None)


def get_last_prompt_context() -> Optional[PromptContext]:
    """
    Get the last prompt context for debugging purposes.
    
    This function allows access to the PromptContext object that was used to compose
    the final prompt sent to the LLM. This is useful for debugging prompt composition.
    
    Returns:
        The last PromptContext object if available, None otherwise.
        
    Note:
        This is intended for debugging/benchmarking use only and should not be used
        in production code as it relies on internal implementation details.
    """
    return _debug_prompt_context.get(None) 
