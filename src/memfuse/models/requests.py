"""Request models for MemFuse client."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class Message(BaseModel):
    """Message model."""
    
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class InitRequest(BaseModel):
    """Request model for initializing memory."""

    user: str
    agent: Optional[str] = None
    session: Optional[str] = None


class QueryRequest(BaseModel):
    """Request model for querying memory."""

    query: str
    top_k: int = 5
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AddRequest(BaseModel):
    """Request model for adding messages."""

    messages: List[Message]


class ReadRequest(BaseModel):
    """Request model for reading messages."""

    message_ids: List[str]


class UpdateRequest(BaseModel):
    """Request model for updating messages."""

    message_ids: List[str]
    new_messages: List[Message]


class DeleteRequest(BaseModel):
    """Request model for deleting messages."""

    message_ids: List[str]


class AddKnowledgeRequest(BaseModel):
    """Request model for adding knowledge."""

    knowledge: List[str]


class ReadKnowledgeRequest(BaseModel):
    """Request model for reading knowledge."""

    knowledge_ids: List[str]


class UpdateKnowledgeRequest(BaseModel):
    """Request model for updating knowledge."""

    knowledge_ids: List[str]
    new_knowledge: List[str]


class DeleteKnowledgeRequest(BaseModel):
    """Request model for deleting knowledge."""

    knowledge_ids: List[str]
