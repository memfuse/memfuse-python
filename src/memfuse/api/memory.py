"""Memory API client for MemFuse."""

from typing import Dict, List, Optional, Any
import warnings

from ..models.requests import (
    InitRequest,
    QueryRequest,
    AddRequest,
    ReadRequest,
    UpdateRequest,
    DeleteRequest,
    AddKnowledgeRequest,
    ReadKnowledgeRequest,
    UpdateKnowledgeRequest,
    DeleteKnowledgeRequest,
)


class MemoryApi:
    """Memory API client for MemFuse."""

    def __init__(self, client):
        """Initialize the memory API client.

        Args:
            client: The MemFuse client
        """
        self.client = client

    async def init(
        self,
        user: str,
        agent: Optional[str] = None,
        session: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Initialize a memory instance.

        Args:
            user: User name (required)
            agent: Agent name (optional)
            session: Session name (optional, will be auto-generated if not provided)

        Returns:
            Response data
        """
        request = InitRequest(
            user=user,
            agent=agent,
            session=session,
        )

        return await self.client._request("POST", "/api/v1/memory/init", request.model_dump())

    async def query(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # Deprecated: retained for compatibility; ignored in payload
        store_type: Optional[str] = None,
        include_messages: bool = True,
        include_knowledge: bool = True,
    ) -> Dict[str, Any]:
        """Query the memory.

        Args:
            session_id: Session ID
            query: Query string
            top_k: Number of results to return
            agent_id: Optional agent ID to filter results
            metadata: Optional metadata to provide additional query context (e.g., {"task": "...", "mode": "..."})
            store_type: Deprecated; ignored
            include_messages: Deprecated; ignored
            include_knowledge: Deprecated; ignored

        Returns:
            Response data
        """
        if store_type is not None or not include_messages or not include_knowledge:
            warnings.warn(
                "memory.query: 'store_type', 'include_messages', and 'include_knowledge' are deprecated and ignored in request payload.",
                DeprecationWarning,
                stacklevel=2,
            )

        request = QueryRequest(
            query=query,
            top_k=top_k,
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata,
        )

        return await self.client._request(
            "POST",
            f"/api/v1/memory/query?session_id={session_id}",
            request.model_dump(),
        )

    async def add(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Add messages to the memory.

        Args:
            session_id: Session ID
            messages: List of message dictionaries with role, content, and optional metadata

        Returns:
            Response data
        """
        request = AddRequest(messages=messages)

        return await self.client._request(
            "POST",
            f"/api/v1/memory/add?session_id={session_id}",
            request.model_dump(),
        )

    async def read(
        self,
        session_id: str,
        message_ids: List[str],
    ) -> Dict[str, Any]:
        """Read messages from the memory.

        Args:
            session_id: Session ID
            message_ids: List of message IDs

        Returns:
            Response data
        """
        request = ReadRequest(message_ids=message_ids)

        return await self.client._request(
            "POST",
            f"/api/v1/memory/read?session_id={session_id}",
            request.model_dump(),
        )

    async def update(
        self,
        session_id: str,
        message_ids: List[str],
        new_messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Update messages in the memory.

        Args:
            session_id: Session ID
            message_ids: List of message IDs
            new_messages: List of new message dictionaries (role, content, optional metadata)

        Returns:
            Response data
        """
        request = UpdateRequest(
            message_ids=message_ids,
            new_messages=new_messages,
        )

        return await self.client._request(
            "PUT",
            f"/api/v1/memory/update?session_id={session_id}",
            request.model_dump(),
        )

    async def delete(
        self,
        session_id: str,
        message_ids: List[str],
    ) -> Dict[str, Any]:
        """Delete messages from the memory.

        Args:
            session_id: Session ID
            message_ids: List of message IDs

        Returns:
            Response data
        """
        request = DeleteRequest(message_ids=message_ids)

        return await self.client._request(
            "DELETE",
            f"/api/v1/memory/delete?session_id={session_id}",
            request.model_dump(),
        )

    async def add_knowledge(
        self,
        session_id: str,
        knowledge: List[str],
    ) -> Dict[str, Any]:
        """Add knowledge to the memory.

        Args:
            session_id: Session ID
            knowledge: List of knowledge strings

        Returns:
            Response data
        """
        request = AddKnowledgeRequest(knowledge=knowledge)

        return await self.client._request(
            "POST",
            f"/api/v1/memory/knowledge/add?session_id={session_id}",
            request.model_dump(),
        )

    async def read_knowledge(
        self,
        session_id: str,
        knowledge_ids: List[str],
    ) -> Dict[str, Any]:
        """Read knowledge from the memory.

        Args:
            session_id: Session ID
            knowledge_ids: List of knowledge IDs

        Returns:
            Response data
        """
        request = ReadKnowledgeRequest(knowledge_ids=knowledge_ids)

        return await self.client._request(
            "POST",
            f"/api/v1/memory/knowledge/read?session_id={session_id}",
            request.model_dump(),
        )

    async def update_knowledge(
        self,
        session_id: str,
        knowledge_ids: List[str],
        new_knowledge: List[str],
    ) -> Dict[str, Any]:
        """Update knowledge in the memory.

        Args:
            session_id: Session ID
            knowledge_ids: List of knowledge IDs
            new_knowledge: List of new knowledge strings

        Returns:
            Response data
        """
        request = UpdateKnowledgeRequest(
            knowledge_ids=knowledge_ids,
            new_knowledge=new_knowledge,
        )

        return await self.client._request(
            "PUT",
            f"/api/v1/memory/knowledge/update?session_id={session_id}",
            request.model_dump(),
        )

    async def delete_knowledge(
        self,
        session_id: str,
        knowledge_ids: List[str],
    ) -> Dict[str, Any]:
        """Delete knowledge from the memory.

        Args:
            session_id: Session ID
            knowledge_ids: List of knowledge IDs

        Returns:
            Response data
        """
        request = DeleteKnowledgeRequest(knowledge_ids=knowledge_ids)

        return await self.client._request(
            "DELETE",
            f"/api/v1/memory/knowledge/delete?session_id={session_id}",
            request.model_dump(),
        )
