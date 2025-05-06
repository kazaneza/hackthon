"""
Pydantic models for the Bank of Kigali AI Assistant API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    """Chat message model."""
    
    role: str = Field(..., description="Role of the message sender (user/assistant/system)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    """Chat request model."""
    
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    service_category: str = Field(..., description="Service category for the query")
    user_id: Optional[str] = Field(None, description="User ID for personalization")

class ChatResponse(BaseModel):
    """Chat response model."""
    
    response: str = Field(..., description="AI assistant response")
    conversation_id: str = Field(..., description="Unique conversation ID")
    service_category: str = Field(..., description="Service category that was processed")
    timestamp: str = Field(..., description="Timestamp of the response")
    suggestions: List[str] = Field(..., description="Suggested follow-up questions")

class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Timestamp of the health check")
    version: str = Field(..., description="API version")