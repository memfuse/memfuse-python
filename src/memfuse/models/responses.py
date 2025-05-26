"""Response models for MemFuse client."""

from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class ErrorDetail(BaseModel):
    """Error detail model."""
    
    field: str
    message: str


class ApiResponse(BaseModel):
    """API response model."""
    
    status: str
    code: int
    data: Optional[Dict[str, Any]] = None
    message: str
    errors: Optional[List[ErrorDetail]] = None
    
    @classmethod
    def success(cls, data: Optional[Dict[str, Any]] = None, message: str = "Success") -> "ApiResponse":
        """Create a success response."""
        return cls(
            status="success",
            code=200,
            data=data,
            message=message,
            errors=None,
        )
    
    @classmethod
    def error(cls, message: str, code: int = 500, errors: Optional[List[ErrorDetail]] = None) -> "ApiResponse":
        """Create an error response."""
        if errors is None:
            errors = [ErrorDetail(field="general", message=message)]
        
        return cls(
            status="error",
            code=code,
            data=None,
            message=message,
            errors=errors,
        )
    

# Re-export common models
__all__ = ["ErrorDetail", "ApiResponse"]
