"""Client-side memory implementation."""

from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import AsyncMemFuse


class AsyncMemory:
    """Client-side memory implementation that communicates with the MemFuse server.

    This class is a thin wrapper around the MemFuse server API. It does not store
    any data locally, but instead forwards all requests to the server.

    This class can be used as a context manager to automatically close
    the client session when the context is exited.

    Example:
        ```python
        async with memory:
            # Use memory...
        # Client session is automatically closed
        ```
    """

    def __init__(
        self,
        client: "AsyncMemFuse",
        session_id: str,
        user_id: str,
        agent_id: str,
        user_name: str,
        agent_name: Optional[str] = None,
        session_name: Optional[str] = None,
        max_chat_history: int = 10
    ):
        """Initialize the client memory.

        Args:
            client: AsyncMemFuse instance
            session_id: Session ID
            user_id: User ID
            agent_id: Agent ID
            user_name: User name
            agent_name: Agent name (optional)
            session_name: Session name (optional)
        """
        self.client = client

        # Internal IDs for API calls
        self.session_id = session_id
        self.user_id = user_id
        self.agent_id = agent_id

        # Public properties for names (external interface)
        self.user = user_name
        self.agent = agent_name
        self.session = session_name or session_id

        self.max_chat_history = max_chat_history

    def __repr__(self):
        return (
            f"AsyncMemory(\n"
            f"  session_id={self.session_id}, user_id={self.user_id}, agent_id={self.agent_id}\n"
            f"  user='{self.user}', agent='{self.agent}', session='{self.session}'\n"
            f")"
        )
    
    def __str__(self):
        return self.__repr__()

    async def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: int = 5,
        store_type: Optional[str] = None,
        include_messages: bool = True,
        include_knowledge: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query the memory for relevant information.

        Args:
            query: The query string
            session_id: Optional session ID to tag results with scope.
                        If provided, results will be tagged with scope="in_session" or
                        scope="cross_session" based on the session they belong to.
                        If None or not provided, all results will have scope=null.
            agent_id: Optional agent ID to filter results
            top_k: Number of results to return
            store_type: Deprecated; ignored by server
            include_messages: Deprecated; ignored by server
            include_knowledge: Deprecated; ignored by server
            metadata: Optional metadata to provide additional query context (e.g., {"task": "...", "mode": "..."})

        Returns:
            Dict containing query results
        """
        # Query the memory
        response = await self.client.users.query(
            user_id=self.user_id,
            query=query,
            session_id=session_id,  # Controls scope tagging in results
            agent_id=agent_id,
            top_k=top_k,
            store_type=store_type,
            include_messages=include_messages,
            include_knowledge=include_knowledge,
            metadata=metadata,
        )

        # Return the full response for backward compatibility
        return response

    async def query_session(
        self,
        query: str,
        top_k: int = 5,
        store_type: Optional[str] = None,
        include_messages: bool = True,
        include_knowledge: bool = True,
    ) -> Dict[str, Any]:
        """Query the memory for relevant information within the current session context.
        
        This is a convenience wrapper around query() that automatically uses the 
        current session_id and agent_id values.

        Args:
            query: The query string
            top_k: Number of results to return
            store_type: Type of store to query
            include_messages: Whether to include messages in the query
            include_knowledge: Whether to include knowledge in the query

        Returns:
            Dict containing query results
        """
        return await self.query(
            query=query,
            session_id=self.session_id,
            agent_id=self.agent_id,
            top_k=top_k,
            store_type=store_type,
            include_messages=include_messages,
            include_knowledge=include_knowledge,
        )

    async def add(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add messages to the memory.

        Args:
            messages: List of message dictionaries with role, content, and optional metadata

        Returns:
            Dict containing message IDs
        """
        return await self.client.messages.add(
            session_id=self.session_id,
            messages=messages,
        )
    
    async def _granular_add(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add messages to the memory in a granular way, respecting user-assistant pairing.

        Args:
            messages: List of message dictionaries with role, content, and optional metadata

        Returns:
            Dict containing aggregated message IDs from all chunked additions.
        """
        if not messages:
            return {"message_ids": [], "status": "no messages to add"}

        all_api_call_results = []
        
        if len(messages) <= 2:
            if messages: 
                result = await self.client.messages.add(
                    session_id=self.session_id,
                    messages=messages,
                )
                all_api_call_results.append(result)
        else:
            idx = 0
            sent_chunk_messages_history: List[List[Dict[str, Any]]] = [] 

            while idx < len(messages):
                current_chunk_candidate: List[Dict[str, Any]] = []
                start_of_this_chunk_original_idx = idx 

                first_message_in_chunk = messages[idx]

                if first_message_in_chunk['role'] != 'user':
                    current_chunk_candidate.append(first_message_in_chunk)
                    extension_idx = idx + 1
                    ua_pair_formed_at_tail = False
                    while extension_idx < len(messages):
                        current_chunk_candidate.append(messages[extension_idx])
                        if (len(current_chunk_candidate) >= 2 and
                            current_chunk_candidate[-2]['role'] == 'user' and
                            current_chunk_candidate[-1]['role'] == 'assistant'):
                            idx = extension_idx + 1 
                            ua_pair_formed_at_tail = True
                            break
                        extension_idx += 1
                    
                    if not ua_pair_formed_at_tail:
                        idx = extension_idx 
                else:
                    end_of_chunk_exclusive_idx = min(start_of_this_chunk_original_idx + 2, len(messages))
                    current_chunk_candidate = messages[start_of_this_chunk_original_idx : end_of_chunk_exclusive_idx]
                    idx = end_of_chunk_exclusive_idx 

                is_final_batch_being_processed = (idx == len(messages))

                if is_final_batch_being_processed and current_chunk_candidate:
                    is_candidate_good_ua_pair = (len(current_chunk_candidate) >= 2 and
                                                 current_chunk_candidate[-2]['role'] == 'user' and
                                                 current_chunk_candidate[-1]['role'] == 'assistant')

                    if not is_candidate_good_ua_pair and sent_chunk_messages_history:
                        last_sent_complete_chunk = sent_chunk_messages_history[-1]
                        if (len(current_chunk_candidate) == 1 and
                            current_chunk_candidate[0]['role'] == 'assistant' and
                            last_sent_complete_chunk and
                            last_sent_complete_chunk[-1]['role'] == 'user'):
                            current_chunk_candidate = [last_sent_complete_chunk[-1]] + current_chunk_candidate
                
                if current_chunk_candidate:
                    result = await self.client.messages.add(
                        session_id=self.session_id,
                        messages=current_chunk_candidate,
                    )
                    all_api_call_results.append(result)
                    sent_chunk_messages_history.append(list(current_chunk_candidate)) 

        aggregated_message_ids: List[str] = []
        final_status = "ok" 

        if not all_api_call_results and messages: 
            final_status = "no messages processed or sent by granular add"
        elif not all_api_call_results and not messages: 
             pass 
        
        for i, res_dict in enumerate(all_api_call_results):
            if res_dict and isinstance(res_dict.get("message_ids"), list):
                aggregated_message_ids.extend(res_dict["message_ids"])
            if i == len(all_api_call_results) -1 : 
                if res_dict and "status" in res_dict:
                    final_status = res_dict["status"]
        
        return {"message_ids": aggregated_message_ids, "status": final_status}

    async def list_messages(
        self,
        limit: Optional[int] = 20,
        sort_by: Optional[str] = "timestamp",
        order: Optional[str] = "desc",
        buffer_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """List all messages in the current session's memory.

        Args:
            limit: Maximum number of messages to return. Defaults to 20.
            sort_by: Field to sort messages by (e.g., "timestamp", "id"). Defaults to "timestamp".
            order: Sort order ("asc" or "desc"). Defaults to "desc".

        Returns:
            Dict containing messages
        """
        return await self.client.messages.list(
            session_id=self.session_id, 
            limit=limit, 
            sort_by=sort_by, 
            order=order,
            buffer_only=buffer_only,
        )

    async def read(self, message_ids: List[str]) -> Dict[str, Any]:
        """Read messages from the memory.

        Args:
            message_ids: List of message IDs

        Returns:
            Dict containing messages
        """
        return await self.client.messages.read(
            session_id=self.session_id,
            message_ids=message_ids,
        )

    async def update(
        self, message_ids: List[str], new_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update messages in the memory.

        Args:
            message_ids: List of message IDs
            new_messages: List of new message dictionaries (role, content, optional metadata)

        Returns:
            Dict containing updated message IDs
        """
        return await self.client.messages.update(
            session_id=self.session_id,
            message_ids=message_ids,
            new_messages=new_messages,
        )

    async def delete(self, message_ids: List[str]) -> Dict[str, Any]:
        """Delete messages from the memory.

        Args:
            message_ids: List of message IDs

        Returns:
            Dict containing deleted message IDs
        """
        return await self.client.messages.delete(
            session_id=self.session_id,
            message_ids=message_ids,
        )

    async def add_knowledge(self, knowledge: List[str]) -> Dict[str, Any]:
        """Add knowledge to the memory.

        Args:
            knowledge: List of knowledge strings

        Returns:
            Dict containing knowledge IDs
        """
        return await self.client.knowledge.add(
            user_id=self.user_id,
            knowledge=knowledge,
        )

    async def read_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Read knowledge from the memory.

        Args:
            knowledge_ids: List of knowledge IDs

        Returns:
            Dict containing knowledge items
        """
        return await self.client.knowledge.read(
            user_id=self.user_id,
            knowledge_ids=knowledge_ids,
        )

    async def update_knowledge(
        self, knowledge_ids: List[str], new_knowledge: List[str]
    ) -> Dict[str, Any]:
        """Update knowledge in the memory.

        Args:
            knowledge_ids: List of knowledge IDs
            new_knowledge: List of new knowledge strings

        Returns:
            Dict containing updated knowledge IDs
        """
        return await self.client.knowledge.update(
            user_id=self.user_id,
            knowledge_ids=knowledge_ids,
            new_knowledge=new_knowledge,
        )

    async def delete_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Delete knowledge from the memory.

        Args:
            knowledge_ids: List of knowledge IDs

        Returns:
            Dict containing deleted knowledge IDs
        """
        return await self.client.knowledge.delete(
            user_id=self.user_id,
            knowledge_ids=knowledge_ids,
        )

    # Note: Memory instances don't own the underlying client, so they don't
    # need close() or context manager methods. The client should be closed
    # separately when it's no longer needed.


if TYPE_CHECKING:
    from .client import MemFuse # Import synchronous MemFuse for type hinting


class Memory:
    """Synchronous client-side memory implementation that communicates with the MemFuse server."""

    def __init__(
        self,
        client: "MemFuse", # Use the synchronous MemFuse client
        session_id: str,
        user_id: str,
        agent_id: str,
        user_name: str,
        agent_name: Optional[str] = None,
        session_name: Optional[str] = None,
        max_chat_history: int = 10,
    ):
        """Initialize the synchronous client memory.

        Args:
            client: Synchronous MemFuse client instance
            session_id: Session ID
            user_id: User ID
            agent_id: Agent ID
            user_name: User name
            agent_name: Agent name (optional)
            session_name: Session name (optional)
            max_chat_history: Maximum chat history to keep
        """
        self.client = client
        self.max_chat_history = max_chat_history
        
        # Internal IDs for API calls
        self.session_id = session_id
        self.user_id = user_id
        self.agent_id = agent_id

        # Public properties for names (external interface)
        self.user = user_name
        self.agent = agent_name
        self.session = session_name or session_id

    def __repr__(self):
        return (
            f"Memory(\n"
            f"  user='{self.user}', agent='{self.agent}', session='{self.session}',\n"
            f"  user_id='{self.user_id}', agent_id='{self.agent_id}', session_id='{self.session_id}'\n"
            f")"
        )
    
    def __str__(self):
        return self.__repr__()

    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: int = 5,
        store_type: Optional[str] = None,
        include_messages: bool = True,
        include_knowledge: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query the memory for relevant information.

        Args:
            query: The query string
            session_id: Optional session ID to tag results with scope.
                        If provided, results will be tagged with scope="in_session" or
                        scope="cross_session" based on the session they belong to.
                        If None or not provided, all results will have scope=null.
            agent_id: Optional agent ID to filter results
            top_k: Number of results to return
            store_type: Deprecated; ignored by server
            include_messages: Deprecated; ignored by server
            include_knowledge: Deprecated; ignored by server
            metadata: Optional metadata to provide additional query context (e.g., {"task": "...", "mode": "..."})

        Returns:
            Dict containing query results
        """
        return self.client.users.query_sync(
            user_id=self.user_id,
            query=query,
            session_id=session_id,
            agent_id=agent_id,
            top_k=top_k,
            store_type=store_type,
            include_messages=include_messages,
            include_knowledge=include_knowledge,
            metadata=metadata,
        )

    def query_session(
        self,
        query: str,
        top_k: int = 5,
        store_type: Optional[str] = None,
        include_messages: bool = True,
        include_knowledge: bool = True,
    ) -> Dict[str, Any]:
        """Query the memory for relevant information within the current session context.
        
        This is a convenience wrapper around query() that automatically uses the 
        current session_id and agent_id values.

        Args:
            query: The query string
            top_k: Number of results to return
            store_type: Type of store to query
            include_messages: Whether to include messages in the query
            include_knowledge: Whether to include knowledge in the query

        Returns:
            Dict containing query results
        """
        return self.query(
            query=query,
            session_id=self.session_id,
            agent_id=self.agent_id,
            top_k=top_k,
            store_type=store_type,
            include_messages=include_messages,
            include_knowledge=include_knowledge,
        )

    def add(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add messages to the memory.

        Args:
            messages: List of message dictionaries with role, content, and optional metadata

        Returns:
            Dict containing message IDs
        """
        return self.client.messages.add_sync(
            session_id=self.session_id,
            messages=messages,
        )
    
    def _granular_add(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add messages to the memory in a granular way, respecting user-assistant pairing.

        Args:
            messages: List of message dictionaries with role, content, and optional metadata

        Returns:
            Dict containing aggregated message IDs from all chunked additions.
        """
        if not messages:
            return {"message_ids": [], "status": "no messages to add"}

        all_api_call_results = []
        
        if len(messages) <= 2:
            if messages: 
                result = self.client.messages.add_sync(
                    session_id=self.session_id,
                    messages=messages,
                )
                all_api_call_results.append(result)
        else:
            idx = 0
            sent_chunk_messages_history: List[List[Dict[str, Any]]] = [] 

            while idx < len(messages):
                current_chunk_candidate: List[Dict[str, Any]] = []
                start_of_this_chunk_original_idx = idx 

                first_message_in_chunk = messages[idx]

                if first_message_in_chunk['role'] != 'user':
                    current_chunk_candidate.append(first_message_in_chunk)
                    extension_idx = idx + 1
                    ua_pair_formed_at_tail = False
                    while extension_idx < len(messages):
                        current_chunk_candidate.append(messages[extension_idx])
                        if (len(current_chunk_candidate) >= 2 and
                            current_chunk_candidate[-2]['role'] == 'user' and
                            current_chunk_candidate[-1]['role'] == 'assistant'):
                            idx = extension_idx + 1 
                            ua_pair_formed_at_tail = True
                            break
                        extension_idx += 1
                    
                    if not ua_pair_formed_at_tail:
                        idx = extension_idx 
                else:
                    end_of_chunk_exclusive_idx = min(start_of_this_chunk_original_idx + 2, len(messages))
                    current_chunk_candidate = messages[start_of_this_chunk_original_idx : end_of_chunk_exclusive_idx]
                    idx = end_of_chunk_exclusive_idx 

                is_final_batch_being_processed = (idx == len(messages))

                if is_final_batch_being_processed and current_chunk_candidate:
                    is_candidate_good_ua_pair = (len(current_chunk_candidate) >= 2 and
                                                 current_chunk_candidate[-2]['role'] == 'user' and
                                                 current_chunk_candidate[-1]['role'] == 'assistant')

                    if not is_candidate_good_ua_pair and sent_chunk_messages_history:
                        last_sent_complete_chunk = sent_chunk_messages_history[-1]
                        if (len(current_chunk_candidate) == 1 and
                            current_chunk_candidate[0]['role'] == 'assistant' and
                            last_sent_complete_chunk and
                            last_sent_complete_chunk[-1]['role'] == 'user'):
                            current_chunk_candidate = [last_sent_complete_chunk[-1]] + current_chunk_candidate
                
                if current_chunk_candidate:
                    result = self.client.messages.add_sync(
                        session_id=self.session_id,
                        messages=current_chunk_candidate,
                    )
                    all_api_call_results.append(result)
                    sent_chunk_messages_history.append(list(current_chunk_candidate)) 

        aggregated_message_ids: List[str] = []
        final_status = "ok" 

        if not all_api_call_results and messages: 
            final_status = "no messages processed or sent by granular add"
        elif not all_api_call_results and not messages: 
             pass 
        
        for i, res_dict in enumerate(all_api_call_results):
            if res_dict and isinstance(res_dict.get("message_ids"), list):
                aggregated_message_ids.extend(res_dict["message_ids"])
            if i == len(all_api_call_results) -1 : 
                if res_dict and "status" in res_dict:
                    final_status = res_dict["status"]
        
        return {"message_ids": aggregated_message_ids, "status": final_status}

    def list_messages(
        self,
        limit: Optional[int] = 20,
        sort_by: Optional[str] = "timestamp",
        order: Optional[str] = "desc",
        buffer_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """List all messages in the current session's memory.

        Args:
            limit: Maximum number of messages to return. Defaults to 20.
            sort_by: Field to sort messages by (e.g., "timestamp", "id"). Defaults to "timestamp".
            order: Sort order ("asc" or "desc"). Defaults to "desc".

        Returns:
            Dict containing messages
        """
        return self.client.messages.list_sync(
            session_id=self.session_id, 
            limit=limit, 
            sort_by=sort_by, 
            order=order,
            buffer_only=buffer_only,
        )

    def read(self, message_ids: List[str]) -> Dict[str, Any]:
        """Read messages from the memory.

        Args:
            message_ids: List of message IDs

        Returns:
            Dict containing messages
        """
        return self.client.messages.read_sync(
            session_id=self.session_id,
            message_ids=message_ids,
        )

    def update(
        self, message_ids: List[str], new_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update messages in the memory.

        Args:
            message_ids: List of message IDs
            new_messages: List of new message dictionaries (role, content, optional metadata)

        Returns:
            Dict containing updated message IDs
        """
        return self.client.messages.update_sync(
            session_id=self.session_id,
            message_ids=message_ids,
            new_messages=new_messages,
        )

    def delete(self, message_ids: List[str]) -> Dict[str, Any]:
        """Delete messages from the memory.

        Args:
            message_ids: List of message IDs

        Returns:
            Dict containing deleted message IDs
        """
        return self.client.messages.delete_sync(
            session_id=self.session_id,
            message_ids=message_ids,
        )

    def add_knowledge(self, knowledge: List[str]) -> Dict[str, Any]:
        """Add knowledge to the memory.

        Args:
            knowledge: List of knowledge strings

        Returns:
            Dict containing knowledge IDs
        """
        return self.client.knowledge.add_sync(
            user_id=self.user_id,
            knowledge=knowledge,
        )

    def read_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Read knowledge from the memory.

        Args:
            knowledge_ids: List of knowledge IDs

        Returns:
            Dict containing knowledge items
        """
        return self.client.knowledge.read_sync(
            user_id=self.user_id,
            knowledge_ids=knowledge_ids,
        )

    def update_knowledge(
        self, knowledge_ids: List[str], new_knowledge: List[str]
    ) -> Dict[str, Any]:
        """Update knowledge in the memory.

        Args:
            knowledge_ids: List of knowledge IDs
            new_knowledge: List of new knowledge strings

        Returns:
            Dict containing updated knowledge IDs
        """
        return self.client.knowledge.update_sync(
            user_id=self.user_id,
            knowledge_ids=knowledge_ids,
            new_knowledge=new_knowledge,
        )

    def delete_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Delete knowledge from the memory.

        Args:
            knowledge_ids: List of knowledge IDs

        Returns:
            Dict containing deleted knowledge IDs
        """
        return self.client.knowledge.delete_sync(
            user_id=self.user_id,
            knowledge_ids=knowledge_ids,
        )

    # Note: Memory instances don't own the underlying client, so they don't
    # need close() or context manager methods. The client should be closed
    # separately when it's no longer needed.
    
