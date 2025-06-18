# src/memfuse/llm/openai_adapter.py
from __future__ import annotations
from typing import Any, Dict, List, Callable
import inspect, functools
import logging

from openai import OpenAI, AsyncOpenAI

from memfuse import Memory, AsyncMemory
from memfuse.prompts import PromptContext, PromptFormatter

# Set up logger for this module
logger = logging.getLogger(__name__)


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

        chat_history = [{"role": message["role"], "content": message["content"]} for message in in_db_chat_history["data"]["messages"][::-1]] + [{"role": message["role"], "content": message["content"]} for message in in_buffer_chat_history["data"]["messages"][::-1]]

        # ------- 3. Retrieve memories ---------------------------------------
        query_string = PromptFormatter.messages_to_query(chat_history + query_messages)
        query_response = memory.query_session(query_string)
        retrieved_memories = query_response["data"]["results"]

        # ------- 4. Compose the prompt --------------------------
        prompt_context = PromptContext(
            query_messages=query_messages,
            retrieved_memories=retrieved_memories,
            retrieved_chat_history=chat_history,
            max_chat_history=max_chat_history,
        )

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
                messages_to_persist = list(query_messages)
                if assistant_response_content.strip():
                    messages_to_persist.append({
                        "role": "assistant",
                        "content": assistant_response_content.strip()
                    })
                
                if messages_to_persist:
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
            messages_to_persist = list(query_messages) # Start with the original user messages for this turn
            
            if response and response.choices and response.choices[0].message:
                assistant_message = response.choices[0].message
                messages_to_persist.append({
                    "role": assistant_message.role,
                    "content": assistant_message.content
                })
                
            if messages_to_persist: # Only add if there's something to add
                result = memory.add(messages=messages_to_persist)
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

        chat_history = [{"role": message["role"], "content": message["content"]} for message in in_db_chat_history["data"]["messages"][::-1]] + [{"role": message["role"], "content": message["content"]} for message in in_buffer_chat_history["data"]["messages"][::-1]]

        # ------- 3. Retrieve memories ---------------------------------------
        query_string = PromptFormatter.messages_to_query(chat_history + query_messages)
        query_response = await memory.query_session(query_string)
        retrieved_memories = query_response["data"]["results"]

        logger.info(retrieved_memories)
        # ------- 4. Compose the prompt --------------------------
        prompt_context = PromptContext(
            query_messages=query_messages,
            retrieved_memories=retrieved_memories,
            retrieved_chat_history=chat_history,
            max_chat_history=max_chat_history,
        )

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
                messages_to_persist = list(query_messages)
                if assistant_response_content.strip():
                    messages_to_persist.append({
                        "role": "assistant",
                        "content": assistant_response_content.strip()
                    })
                
                if messages_to_persist:
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
            messages_to_persist = list(query_messages) # Start with the original user messages for this turn
            
            if response and response.choices and response.choices[0].message:
                assistant_message = response.choices[0].message
                messages_to_persist.append({
                    "role": assistant_message.role,
                    "content": assistant_message.content
                })
                
            if messages_to_persist: # Only add if there's something to add
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