# src/memfuse/llm/gemini_adapter.py
from __future__ import annotations
from typing import Any, Callable, List, Dict, Union, cast, Coroutine, Optional
import inspect, functools
import logging
import contextvars
import time

from google import genai
from google.genai import types # types.Content, types.Part, types.GenerateContentResponse
# Assuming AsyncClient is available, if not, this might need adjustment based on SDK specifics
# from google.genai import AsyncClient as AsyncGeminiClient # google.genai.Client can be used with an async transport

from memfuse import Memory
from memfuse.prompts import PromptContext, PromptFormatter

# Set up logger for this module
logger = logging.getLogger(__name__)

# Context variables to capture query responses, timing, and prompt context for debugging (invisible to SDK users)
_debug_query_response: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar('debug_query_response', default=None)
_debug_query_time: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar('debug_query_time', default=None)
_debug_prompt_context: contextvars.ContextVar[Optional[PromptContext]] = contextvars.ContextVar('debug_prompt_context', default=None)

# Type aliases for clarity, based on google.genai.types
ContentDict = Dict[str, Any] # Represents types.Content or similar structure for internal processing
PartDict = Dict[str, Any]    # Represents types.Part or similar structure
# GenerateContentResponse = types.GenerateContentResponse # Already a type

# Helper to convert PromptFormatter's output to SDK's types.Content list
def _prepare_contents_for_sdk(structured_prompt_parts: List[Dict[str, str]]) -> List[types.Content]:
    sdk_contents: List[types.Content] = []
    for msg_dict in structured_prompt_parts:
        # Ensure role is valid for Gemini (user or model, typically)
        # PromptFormatter might use 'assistant', which maps to 'model' for Gemini.
        role = msg_dict['role']
        if role == 'assistant':
            role = 'model' 
        sdk_contents.append(types.Content(role=role, parts=[types.Part(text=msg_dict['content'])]))
    return sdk_contents

# Helper to extract query messages for memory processing from SDK's contents
def _extract_query_messages_from_sdk_contents(
    sdk_contents: Union[str, List[Union[str, types.Content, PartDict, ContentDict]]] # Input can be varied
) -> List[Dict[str, Any]]:
    query_messages: List[Dict[str, Any]] = []
    
    # Normalize input to List[types.Content] or compatible dicts
    normalized_contents: List[Union[types.Content, ContentDict]] = []
    if isinstance(sdk_contents, str):
        normalized_contents.append(types.Content(role="user", parts=[types.Part(text=sdk_contents)]))
    elif isinstance(sdk_contents, list):
        for item in sdk_contents:
            if isinstance(item, str):
                normalized_contents.append(types.Content(role="user", parts=[types.Part(text=item)]))
            elif isinstance(item, types.Content):
                normalized_contents.append(item)
            elif isinstance(item, dict) and 'parts' in item: # Assumed to be ContentDict-like
                 # Map 'assistant' role if present from older direct dict usage
                role = item.get('role', 'user')
                if role == 'assistant':
                    role = 'model'
                
                # Ensure parts are correctly structured for types.Part
                processed_parts = []
                for part_item in item.get('parts', []):
                    if isinstance(part_item, str): # simple text part
                        processed_parts.append(types.Part(text=part_item))
                    elif isinstance(part_item, types.Part):
                        processed_parts.append(part_item)
                    elif isinstance(part_item, dict) and 'text' in part_item: # PartDict-like
                        processed_parts.append(types.Part(text=part_item['text']))
                    # TODO: Handle other part types like inline_data if necessary
                if processed_parts:
                    normalized_contents.append(types.Content(role=role, parts=processed_parts))
            # PartDict at top level is not typical for 'contents' but handle defensively
            elif isinstance(item, dict) and 'text' in item: # Assumed to be PartDict-like, wrap in user Content
                normalized_contents.append(types.Content(role="user", parts=[types.Part(text=item['text'])]))


    for content_item in normalized_contents:
        # content_item is now types.Content or a ContentDict-like structure
        role = content_item.role if isinstance(content_item, types.Content) else content_item.get('role', 'user')
        # Map 'model' role back to 'assistant' for internal PromptFormatter consistency
        if role == 'model':
            role = 'assistant' 
        
        combined_text_content = ""
        parts = content_item.parts if isinstance(content_item, types.Content) else content_item.get('parts', [])
        for part in parts:
            if isinstance(part, types.Part) and part.text:
                combined_text_content += part.text + " "
            elif isinstance(part, dict) and 'text' in part: # For ContentDict compatibility
                 combined_text_content += part['text'] + " "
            # TODO: Handle other part types like inline_data if needed for query extraction
        
        if combined_text_content.strip():
            query_messages.append({"role": role, "content": combined_text_content.strip()})
            
    return query_messages


def _instrument_generate_content_common(
    original_query_contents: Union[str, List[Union[str, types.Content, PartDict, ContentDict]]],
    memory: Memory,
    is_async: bool = False
) -> Union[List[types.Content], Coroutine[Any, Any, List[types.Content]]]:
    """
    Common logic for instrumenting generate_content and generate_content_async.
    Prepares contents for SDK, retrieves memories, composes prompt, and returns final contents.
    If is_async is True, returns a coroutine that must be awaited.
    """
    if is_async:
        return _instrument_generate_content_async(original_query_contents, memory)
    else:
        return _instrument_generate_content_sync(original_query_contents, memory)


def _instrument_generate_content_sync(
    original_query_contents: Union[str, List[Union[str, types.Content, PartDict, ContentDict]]],
    memory: Memory
) -> List[types.Content]:
    """Synchronous version of content instrumentation."""
    # 1. Extract query messages for memory processing and PromptFormatter
    gemini_query_messages = _extract_query_messages_from_sdk_contents(original_query_contents)

    if not gemini_query_messages:
        logger.info("Gemini query messages are empty after processing, using original contents.")
        if isinstance(original_query_contents, str):
            return [types.Content(role="user", parts=[types.Part(text=original_query_contents)])]
        elif isinstance(original_query_contents, list):
            return _prepare_contents_for_sdk(
                _extract_query_messages_from_sdk_contents(original_query_contents)
            ) if not all(isinstance(c, types.Content) for c in original_query_contents) else original_query_contents
        return original_query_contents

    # 2. Retrieve memories
    latest_user_query_message = None
    if gemini_query_messages and gemini_query_messages[-1]["role"] == "user":
        latest_user_query_message = gemini_query_messages[-1]
    
    retrieved_memories = None
    chat_history = None
    
    if latest_user_query_message:    
        # Get chat history
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

        # Retrieve memories
        query_string = PromptFormatter.messages_to_query(chat_history + gemini_query_messages)
        
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

    # 3. Compose the prompt context for PromptFormatter
    prompt_context = PromptContext(
        query_messages=gemini_query_messages,
        retrieved_memories=retrieved_memories,
        retrieved_chat_history=chat_history,
        max_chat_history=memory.max_chat_history,
    )
    
    # Store prompt context for debugging access (invisible to normal users)
    _debug_prompt_context.set(prompt_context)
    logger.info(f"Stored prompt context in ContextVar: {prompt_context}")
    
    # PromptFormatter composes the full prompt, including history, query, memories.
    full_structured_prompt_parts: List[Dict[str, str]] = prompt_context.compose_for_gemini()
    logger.info(f"Composed prompt parts: {len(full_structured_prompt_parts)} parts")
    
    # Debug: Log the actual prompt being sent
    for i, part in enumerate(full_structured_prompt_parts):
        logger.info(f"PROMPT PART {i+1} [{part['role'].upper()}]: {part['content'][:1000]}...")

    # 4. Convert to SDK's List[types.Content]
    final_contents_for_sdk = _prepare_contents_for_sdk(full_structured_prompt_parts)
    
    return final_contents_for_sdk


async def _instrument_generate_content_async(
    original_query_contents: Union[str, List[Union[str, types.Content, PartDict, ContentDict]]],
    memory: Memory
) -> List[types.Content]:
    """Asynchronous version of content instrumentation."""
    # 1. Extract query messages for memory processing and PromptFormatter
    gemini_query_messages = _extract_query_messages_from_sdk_contents(original_query_contents)

    if not gemini_query_messages:
        logger.info("Gemini query messages are empty after processing, using original contents.")
        if isinstance(original_query_contents, str):
            return [types.Content(role="user", parts=[types.Part(text=original_query_contents)])]
        elif isinstance(original_query_contents, list):
            return _prepare_contents_for_sdk(
                _extract_query_messages_from_sdk_contents(original_query_contents)
            ) if not all(isinstance(c, types.Content) for c in original_query_contents) else original_query_contents
        return original_query_contents

    # 2. Retrieve memories
    latest_user_query_message = None
    if gemini_query_messages and gemini_query_messages[-1]["role"] == "user":
        latest_user_query_message = gemini_query_messages[-1]
    
    retrieved_memories = None
    retrieved_chat_history = None
    
    if latest_user_query_message:    
        # Get chat history
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

        retrieved_chat_history = [{"role": message["role"], "content": message["content"]} for message in db_messages[::-1]] + [{"role": message["role"], "content": message["content"]} for message in in_buffer_chat_history["data"]["messages"][::-1]]

        # Retrieve memories
        query_string = PromptFormatter.messages_to_query(retrieved_chat_history + gemini_query_messages)
        
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

    # 3. Compose the prompt context for PromptFormatter
    prompt_context = PromptContext(
        query_messages=gemini_query_messages,
        retrieved_memories=retrieved_memories,
        retrieved_chat_history=retrieved_chat_history,
        max_chat_history=memory.max_chat_history,
    )
    
    # Store prompt context for debugging access (invisible to normal users)
    _debug_prompt_context.set(prompt_context)
    logger.info(f"Stored prompt context in ContextVar: {prompt_context}")
    
    # PromptFormatter composes the full prompt, including history, query, memories.
    full_structured_prompt_parts: List[Dict[str, str]] = prompt_context.compose_for_gemini()
    logger.info(f"Composed prompt parts: {len(full_structured_prompt_parts)} parts")
    
    # Debug: Log the actual prompt being sent
    for i, part in enumerate(full_structured_prompt_parts):
        logger.info(f"PROMPT PART {i+1} [{part['role'].upper()}]: {part['content'][:1000]}...")

    # 4. Convert to SDK's List[types.Content]
    final_contents_for_sdk = _prepare_contents_for_sdk(full_structured_prompt_parts)
    
    return final_contents_for_sdk


def _persist_interaction_common(
    memory: Memory,
    # Original query messages (user's input to generate_content)
    # This should be the messages *before* memory augmentation for accurate logging
    original_input_messages_for_formatter: List[Dict[str, Any]], 
    response: types.GenerateContentResponse,
    is_async: bool = False
):
    """
    Common logic for persisting the user query and model response.
    Simplified to match OpenAI and Anthropic adapter patterns.
    """
    if is_async:
        return _persist_interaction_async(memory, original_input_messages_for_formatter, response)
    else:
        return _persist_interaction_sync(memory, original_input_messages_for_formatter, response)


def _persist_interaction_sync(
    memory: Memory,
    original_input_messages_for_formatter: List[Dict[str, Any]], 
    response: types.GenerateContentResponse
):
    """Synchronous version of interaction persistence."""
    # Start with the original user messages for this turn
    messages_to_persist = list(original_input_messages_for_formatter)
    
    # Add assistant response
    if response and response.candidates:
        assistant_response_text = ""
        # Ensure candidates and content are not None
        if response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text: # Check for text attribute existence
                    assistant_response_text += part.text
        
        if assistant_response_text:
            messages_to_persist.append({
                "role": "assistant", # Storing as 'assistant'
                "content": assistant_response_text
            })
    
    # Persist if there's something to add
    if messages_to_persist:
        result = memory.add(messages=messages_to_persist)
        logger.info(f"Persisted Gemini interaction.")
        
        if result and result.get("data") and result["data"].get("message_ids"):
            message_ids = result["data"]["message_ids"]
            logger.info(f"Persisted message IDs: {message_ids}")
        else:
            logger.info("Failed to persist messages or no message IDs returned.")
    else:
        logger.info("No messages to persist for this interaction.")


async def _persist_interaction_async(
    memory: Memory,
    original_input_messages_for_formatter: List[Dict[str, Any]], 
    response: types.GenerateContentResponse
):
    """Asynchronous version of interaction persistence."""
    # Start with the original user messages for this turn
    messages_to_persist = list(original_input_messages_for_formatter)
    
    # Add assistant response
    if response and response.candidates:
        assistant_response_text = ""
        # Ensure candidates and content are not None
        if response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text: # Check for text attribute existence
                    assistant_response_text += part.text
        
        if assistant_response_text:
            messages_to_persist.append({
                "role": "assistant", # Storing as 'assistant'
                "content": assistant_response_text
            })
    
    # Persist if there's something to add
    if messages_to_persist:
        result = await memory.add(messages=messages_to_persist)
        logger.info(f"Persisted async Gemini interaction.")
        
        if result and result.get("data") and result["data"].get("message_ids"):
            message_ids = result["data"]["message_ids"]
            logger.info(f"Persisted async message IDs: {message_ids}")
        else:
            logger.info("Failed to persist async messages or no message IDs returned.")
    else:
        logger.info("No messages to persist for this async interaction.")


class MemorableModelsService:
    def __init__(self, actual_models_service: Any, memory_instance: Memory, is_async: bool = False):
        self._actual_models_service = actual_models_service
        self._memory_instance = memory_instance
        self._is_async = is_async

    async def generate_content_async(
        self,
        model: str,
        contents: Union[str, List[Union[str, types.Content, PartDict, ContentDict]]],
        **kwargs: Any
    ) -> types.GenerateContentResponse:
        if not self._memory_instance:
            return await self._actual_models_service.generate_content(model=model, contents=contents, **kwargs)

        # Keep track of the original input messages for persistence later
        # This needs to be in the format PromptFormatter understands
        original_input_for_formatter = _extract_query_messages_from_sdk_contents(contents)
        
        # Instrument to get memory-augmented contents
        final_contents_for_sdk = await cast(Coroutine[Any, Any, List[types.Content]], _instrument_generate_content_common(
            original_query_contents=contents,
            memory=self._memory_instance,
            is_async=True
        ))
        
        response = await self._actual_models_service.generate_content(
            model=model,
            contents=final_contents_for_sdk,
            **kwargs
        )
        
        # Persist the original query (or relevant part) and the response (await async version)
        await _persist_interaction_common(
            memory=self._memory_instance,
            original_input_messages_for_formatter=original_input_for_formatter,
            response=response,
            is_async=True
        )
        return response

    def generate_content(
        self,
        model: str,
        contents: Union[str, List[Union[str, types.Content, PartDict, ContentDict]]], # Match SDK's flexibility
        **kwargs: Any
    ) -> types.GenerateContentResponse:
        if not self._memory_instance:
            return self._actual_models_service.generate_content(model=model, contents=contents, **kwargs)

        original_input_for_formatter = _extract_query_messages_from_sdk_contents(contents)

        final_contents_for_sdk = cast(List[types.Content], _instrument_generate_content_common(
            original_query_contents=contents,
            memory=self._memory_instance,
            is_async=False
        ))
        
        response = self._actual_models_service.generate_content(
            model=model,
            contents=final_contents_for_sdk,
            **kwargs
        )
        
        _persist_interaction_common(
            memory=self._memory_instance,
            original_input_messages_for_formatter=original_input_for_formatter,
            response=response,
            is_async=False
        )
        return response

    # Delegate other methods if any are needed (e.g., embed_content, count_tokens)
    def __getattr__(self, name: str) -> Any:
        """Delegates non-overridden methods to the actual models service."""
        return getattr(self._actual_models_service, name)


class MemorableGoogleGenerativeAI:
    """
    A wrapper around the Google GenAI Client (sync) to inject memory capabilities.
    """
    def __init__(self, memory: Memory, client_options: types.ClientOptions | None = None, **kwargs: Any):
        """
        Initializes the memorable client.
        Args:
            memory: The Memory instance to use.
            client_options: Optional client options for genai.Client.
            **kwargs: Additional arguments for genai.Client (e.g., api_key).
        """
        self._memory_instance = memory
        # Pass client_options and other kwargs directly to the genai.Client
        if client_options:
            self._actual_client = genai.Client(client_options=client_options, **kwargs)
        else:
            self._actual_client = genai.Client(**kwargs) # e.g. api_key from env or passed in kwargs

        self.models = MemorableModelsService(self._actual_client.models, self._memory_instance, is_async=False)

    # Delegate other client attributes/methods if necessary
    def __getattr__(self, name: str) -> Any:
        return getattr(self._actual_client, name)


class AsyncMemorableGoogleGenerativeAI:
    """
    A wrapper around the Google GenAI Client (async) to inject memory capabilities.
    Note: The SDK uses `genai.Client` with an async transport or `genai.AsyncClient`
    might exist in newer versions not fully covered by provided docs.
    Assuming `genai.Client` can be made async or `genai.AsyncClient` is the path.
    For this refactor, we will assume `genai.Client` itself handles async if configured,
    or one would pass an async-compatible transport/client.
    The crucial part is that its `models.generate_content` becomes awaitable.
    """
    def __init__(self, memory: Memory, client_options: types.ClientOptions | None = None, **kwargs: Any):
        """
        Initializes the memorable async client.
        Args:
            memory: The Memory instance to use.
            client_options: Optional client options for genai.Client.
            **kwargs: Additional arguments for genai.Client (e.g., api_key).
                      It's assumed these kwargs or client_options can configure the client for async operations.
        """
        self._memory_instance = memory
        # Configure client for async. How this is done depends on the exact SDK version.
        # It might involve passing an async transport in client_options,
        # or the SDK might use `google.generativeai.AsyncClient` if that's the pattern.
        # The key is `self._actual_client.models.generate_content` should be awaitable.
        # For now, we assume genai.Client can be used in an async context if its methods are awaited
        # and it's initialized appropriately (e.g. no explicit async client needed, just await calls)
        # OR, if a distinct AsyncClient is the way, that would be used here.
        # The docs mention "from google.genai import AsyncClient as AsyncGeminiClient" in old code.
        # Let's use genai.Client and rely on its methods being awaitable if used in async context
        # with proper setup (e.g. in an async function).
        # The `MemorableModelsService` checks `is_async` for its behavior.

        if client_options:
             self._actual_client = genai.Client(client_options=client_options, **kwargs) # Potentially configured for async via options
        else:
             self._actual_client = genai.Client(**kwargs)


        # The `models` service wrapper needs to know it's in an async context
        # to call the correct `generate_content_async` method.
        # It should use the .aio attribute of the client for async operations.
        self.models = MemorableModelsService(self._actual_client.aio.models, self._memory_instance, is_async=True)


    # Delegate other client attributes/methods
    def __getattr__(self, name: str) -> Any:
        """Delegates non-overridden methods to the actual async client."""
        # This is tricky for async. If `name` is an async method, `getattr` returns it,
        # and it should be awaitable.
        attr = getattr(self._actual_client, name)
        # If we need to specifically wrap other async methods, that would be done similarly to `models`.
        return attr

# Old classes and wrappers are now removed / replaced by the client-centric approach.
# MemGenerativeModel, AsyncMemGenerativeModel
# _wrap_gemini_generate_content, _wrap_gemini_async_generate_content


# Debug utilities (for benchmarks and debugging)
def get_last_query_response() -> Optional[Dict[str, Any]]:
    """
    Get the last query response for debugging purposes.
    
    This function allows access to the intermediate query response from memory.query_session()
    that occurs within _instrument_generate_content_async/sync. This is useful for benchmarks
    and debugging to inspect what memories were retrieved.
    
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
    context = _debug_prompt_context.get(None)
    logger.info(f"Retrieved prompt context from ContextVar: {context}")
    return context 
