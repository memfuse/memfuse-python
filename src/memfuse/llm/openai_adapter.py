# src/memfuse/llm/openai_adapter.py
from __future__ import annotations
from typing import Any, Dict, List, Callable, Optional
import inspect, functools
import logging
import contextvars
import time

from openai import OpenAI, AsyncOpenAI

from memfuse import Memory, AsyncMemory
from memfuse.prompts import PromptContext, PromptFormatter

# Set up logger for this module
logger = logging.getLogger(__name__)

# Context variables to capture query responses, timing, and prompt context for debugging (invisible to SDK users)
_debug_query_response: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar('debug_query_response', default=None)
_debug_query_time: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar('debug_query_time', default=None)
_debug_prompt_context: contextvars.ContextVar[Optional[PromptContext]] = contextvars.ContextVar('debug_prompt_context', default=None)


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

        # ------- 1. Extract the messages list --------
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
        query_string = PromptFormatter.messages_to_query(chat_history + query_messages)
        logger.info(f"Query string for memory: {query_string}")
        
        # Measure query time
        query_start_time = time.perf_counter()
        query_response = memory.query_session(query_string)
        query_end_time = time.perf_counter()
        query_duration = query_end_time - query_start_time
        
        # Store query response and timing for debugging access (invisible to normal users)
        _debug_query_response.set(query_response)
        _debug_query_time.set(query_duration)
        
        logger.info(f"Query response type: {type(query_response)}")
        logger.info(f"Query response content: {query_response}")
        logger.info(f"Query took {query_duration * 1000:.2f} ms")

        # Log the structure of the response to understand the format
        if isinstance(query_response, dict):
            logger.info(f"Query response keys: {list(query_response.keys())}")
            if "data" in query_response:
                logger.info(f"query_response['data'] type: {type(query_response['data'])}")
                logger.info(f"query_response['data'] content: {query_response['data']}")
                if isinstance(query_response["data"], dict):
                    logger.info(f"query_response['data'] keys: {list(query_response['data'].keys())}")

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
        logger.info(f"Successfully retrieved {len(retrieved_memories)} memories")

        # ------- 4. Compose the prompt --------------------------
        prompt_context = PromptContext(
            query_messages=query_messages,
            retrieved_memories=retrieved_memories,
            retrieved_chat_history=chat_history,
            max_chat_history=max_chat_history,
        )
        
        # Store prompt context for debugging access (invisible to normal users)
        _debug_prompt_context.set(prompt_context)

        full_msg = prompt_context.compose_for_openai()

        logger.info(full_msg)

        bound.arguments["messages"] = full_msg

        # ------- 5. Forward the call to the real create ---------------------
        is_streaming = bound.arguments.get("stream", False)
        
        if is_streaming:
            # Handle streaming response
            def stream_wrapper():
                response_stream = create_fn(*bound.args, **bound.kwargs)
                assistant_response_content = ""
                
                for chunk in response_stream:
                    # Extract content from the streaming chunk
                    if (hasattr(chunk, 'choices') and chunk.choices and 
                        len(chunk.choices) > 0 and hasattr(chunk.choices[0], 'delta') and 
                        chunk.choices[0].delta and hasattr(chunk.choices[0].delta, 'content') and 
                        chunk.choices[0].delta.content):
                        assistant_response_content += chunk.choices[0].delta.content
                    yield chunk
                
                # After streaming is complete, persist the messages
                # Extract only the new user messages (excluding system messages and chat history)
                messages_to_persist = []
                for msg in query_messages:
                    if msg.get("role") == "user":
                        messages_to_persist.append({
                            "role": "user",
                            "content": msg.get("content", "")
                        })
                
                if assistant_response_content.strip():
                    messages_to_persist.append({
                        "role": "assistant",
                        "content": assistant_response_content.strip()
                    })
                
                if messages_to_persist:
                    logger.info(f"Persisting streaming messages: {messages_to_persist}")
                    result = memory.add(messages=messages_to_persist)
                    if result and result.get("data") and result["data"].get("message_ids"):
                        message_ids = result["data"]["message_ids"]
                        logger.info(f"Persisted streaming message IDs: {message_ids}")
                    else:
                        logger.info("Failed to persist streaming messages or no message IDs returned.")
                else:
                    logger.info("No streaming messages to persist for this interaction.")
            
            return stream_wrapper()
        else:
            # Handle non-streaming response (original logic)
            response = create_fn(*bound.args, **bound.kwargs)

            # ------- 6. Persist *only* the new interaction ----------------------
            # Extract only the new user messages (excluding system messages and chat history)
            messages_to_persist = []
            for msg in query_messages:
                if msg.get("role") == "user":
                    messages_to_persist.append({
                        "role": "user",
                        "content": msg.get("content", "")
                    })
            
            if response and response.choices and response.choices[0].message:
                assistant_message = response.choices[0].message
                messages_to_persist.append({
                    "role": assistant_message.role,
                    "content": assistant_message.content
                })
                
            if messages_to_persist: # Only add if there's something to add
                logger.info(f"Persisting messages: {messages_to_persist}")
                try:
                    logger.info(f"About to call memory.add with messages: {messages_to_persist}")
                    result = memory.add(messages=messages_to_persist)
                    logger.info(f"Memory.add result type: {type(result)}")
                    logger.info(f"Memory.add result content: {result}")
                    if result and result.get("data") and result["data"].get("message_ids"):
                        message_ids = result["data"]["message_ids"]
                        logger.info(f"Persisted message IDs: {message_ids}")
                    else:
                        logger.info("Failed to persist messages or no message IDs returned.")
                except Exception as e:
                    logger.error(f"Error in memory.add: {e}")
                    logger.error(f"Exception type: {type(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise
            else:
                logger.info("No messages to persist for this interaction.")
            
            return response

    # ★ Replace wrapper's __signature__ so help(), IDEs, and type checkers
    #   all show the *original* OpenAI signature.
    wrapper.__signature__ = sig  # type: ignore[attr-defined]
    return wrapper


def _async_wrap_create(
    create_fn: Callable[..., Any],
    memory: AsyncMemory,
) -> Callable[..., Any]:
    """
    Async version that works with AsyncMemory objects.
    Returns a function with the *exact same* signature as `create_fn`
    but that transparently injects conversational memory.
    """
    sig = inspect.signature(create_fn)

    @functools.wraps(create_fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:  # signature replaced below
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # ------- 1. Extract the messages list & augment with history --------
        # NB: v1 SDK uses keyword‐only 'messages'
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
        logger.info(f"Query string for memory: {query_string}")
        
        # Measure query time
        query_start_time = time.perf_counter()
        query_response = await memory.query_session(query_string)
        query_end_time = time.perf_counter()
        query_duration = query_end_time - query_start_time
        
        # Store query response and timing for debugging access (invisible to normal users)
        _debug_query_response.set(query_response)
        _debug_query_time.set(query_duration)
        
        logger.info(f"Query response type: {type(query_response)}")
        logger.info(f"Query response content: {query_response}")
        logger.info(f"Query took {query_duration * 1000:.2f} ms")

        # Log the structure of the response to understand the format
        if isinstance(query_response, dict):
            logger.info(f"Query response keys: {list(query_response.keys())}")
            if "data" in query_response:
                logger.info(f"query_response['data'] type: {type(query_response['data'])}")
                logger.info(f"query_response['data'] content: {query_response['data']}")
                if isinstance(query_response["data"], dict):
                    logger.info(f"query_response['data'] keys: {list(query_response['data'].keys())}")

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
        logger.info(f"Successfully retrieved {len(retrieved_memories)} memories")

        # ------- 4. Compose the prompt --------------------------
        prompt_context = PromptContext(
            query_messages=query_messages,
            retrieved_memories=retrieved_memories,
            retrieved_chat_history=chat_history,
            max_chat_history=max_chat_history,
        )
        
        # Store prompt context for debugging access (invisible to normal users)
        _debug_prompt_context.set(prompt_context)

        full_msg = prompt_context.compose_for_openai()

        logger.info(full_msg)

        bound.arguments["messages"] = full_msg

        # ------- 5. Forward the call to the real create ---------------------
        is_streaming = bound.arguments.get("stream", False)
        
        if is_streaming:
            # Handle streaming response
            async def stream_wrapper():
                response_stream = await create_fn(*bound.args, **bound.kwargs)
                assistant_response_content = ""
                
                async for chunk in response_stream:
                    # Extract content from the streaming chunk
                    if (hasattr(chunk, 'choices') and chunk.choices and 
                        len(chunk.choices) > 0 and hasattr(chunk.choices[0], 'delta') and 
                        chunk.choices[0].delta and hasattr(chunk.choices[0].delta, 'content') and 
                        chunk.choices[0].delta.content):
                        assistant_response_content += chunk.choices[0].delta.content
                    yield chunk
                
                # After streaming is complete, persist the messages
                # Extract only the new user messages (excluding system messages and chat history)
                messages_to_persist = []
                for msg in query_messages:
                    if msg.get("role") == "user":
                        messages_to_persist.append({
                            "role": "user",
                            "content": msg.get("content", "")
                        })
                
                if assistant_response_content.strip():
                    messages_to_persist.append({
                        "role": "assistant",
                        "content": assistant_response_content.strip()
                    })
                
                if messages_to_persist:
                    logger.info(f"Persisting async streaming messages: {messages_to_persist}")
                    result = await memory.add(messages=messages_to_persist)
                    if result and result.get("data") and result["data"].get("message_ids"):
                        message_ids = result["data"]["message_ids"]
                        logger.info(f"Persisted async streaming message IDs: {message_ids}")
                    else:
                        logger.info("Failed to persist async streaming messages or no message IDs returned.")
                else:
                    logger.info("No async streaming messages to persist for this interaction.")
            
            return stream_wrapper()
        else:
            # Handle non-streaming response (original logic)
            response = await create_fn(*bound.args, **bound.kwargs)

            # ------- 6. Persist *only* the new interaction ----------------------
            # Extract only the new user messages (excluding system messages and chat history)
            messages_to_persist = []
            for msg in query_messages:
                if msg.get("role") == "user":
                    messages_to_persist.append({
                        "role": "user",
                        "content": msg.get("content", "")
                    })
            
            if response and response.choices and response.choices[0].message:
                assistant_message = response.choices[0].message
                messages_to_persist.append({
                    "role": assistant_message.role,
                    "content": assistant_message.content
                })
                
            if messages_to_persist: # Only add if there's something to add
                logger.info(f"Persisting messages: {messages_to_persist}")
                result = await memory.add(messages=messages_to_persist)
                if result and result.get("data") and result["data"].get("message_ids"):
                    message_ids = result["data"]["message_ids"]
                    logger.info(f"Persisted message IDs: {message_ids}")
                else:
                    logger.info("Failed to persist messages or no message IDs returned.")
            else:
                logger.info("No messages to persist for this interaction.")
            
            return response

    # ★ Replace wrapper's __signature__ so help(), IDEs, and type checkers
    #   all show the *original* OpenAI signature.
    wrapper.__signature__ = sig  # type: ignore[attr-defined]
    return wrapper


class MemOpenAI(OpenAI):
    """
    Public adapter that *looks identical* to `openai.OpenAI`.
    Memory is applied only to chat completions for now, but the pattern
    is reusable for Embeddings, Images, etc.
    """

    def __init__(
        self,
        *args: Any,
        memory: Memory | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.memory = memory

        # ---- dynamically monkey‑patch the ChatCompletions.create method ----
        original_create = self.chat.completions.create
        self.chat.completions.create = _wrap_create(original_create, self.memory)  # type: ignore[assignment]


class AsyncMemOpenAI(AsyncOpenAI):
    """Async version that works with AsyncMemory objects."""

    def __init__(
        self,
        *args: Any,
        memory: AsyncMemory | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.memory = memory

        original_create = self.chat.completions.create
        self.chat.completions.create = _async_wrap_create(original_create, self.memory)  # type: ignore[assignment]


# Debug utilities (for benchmarks and debugging)
def get_last_query_response() -> Optional[Dict[str, Any]]:
    """
    Get the last query response for debugging purposes.
    
    This function allows access to the intermediate query response from memory.query_session()
    that occurs within the OpenAI adapter. This is useful for benchmarks and debugging to 
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
