"""Models for MemFuse client."""

from .requests import (
    Message,
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

from .responses import (
    ErrorDetail,
    ApiResponse,
)
