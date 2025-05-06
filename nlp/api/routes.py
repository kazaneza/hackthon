"""
API routes for the Bank of Kigali AI Assistant.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from uuid import uuid4
from datetime import datetime
import json

from api.models import ChatRequest, ChatResponse, HealthResponse
from core.ai_service import OpenAIService
from core.message_store import MessageStore
from core.prompts import get_follow_up_suggestions
from config.settings import settings
from utils.logging_config import logger

# Create router
router = APIRouter()

# Initialize services
message_store = MessageStore(
    max_messages=settings.MAX_STORED_MESSAGES,
    expiry_seconds=settings.MESSAGE_EXPIRY_SECONDS
)

# Dependencies
async def get_openai_service():
    """Dependency to get OpenAI service."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    return OpenAIService(api_key=settings.OPENAI_API_KEY)

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    openai_service: OpenAIService = Depends(get_openai_service)
):
    """
    Process a chat request and return an AI-generated response.
    
    Args:
        request: Chat request with messages and service category
        openai_service: OpenAI service dependency
    
    Returns:
        ChatResponse with AI-generated response and suggestions
    """
    try:
        # Use provided user_id or generate an anonymous one
        user_id = request.user_id or f"anonymous-{uuid4()}"
        
        # Get previous messages for context
        previous_messages = message_store.get_messages(user_id)
        
        # Add new user messages to store
        for msg in request.messages:
            if msg.role == "user":  # Only store user messages
                message_store.add_message(user_id, msg)
        
        # Combine previous messages with current request
        # but only if they aren't already included
        current_message_contents = [m.content for m in request.messages]
        context_messages = []
        
        for prev_msg in previous_messages:
            if prev_msg.content not in current_message_contents:
                context_messages.append(prev_msg)
        
        # Create combined messages list with context
        all_messages = context_messages + request.messages
        
        # Generate response from OpenAI
        ai_response = await openai_service.generate_response(
            service_category=request.service_category,
            messages=all_messages
        )
        
        # Store AI response in message store too
        from api.models import ChatMessage
        message_store.add_message(
            user_id, 
            ChatMessage(role="assistant", content=ai_response)
        )
        
        # Generate conversation ID
        conversation_id = str(uuid4())
        
        # Get follow-up suggestions
        suggestions = get_follow_up_suggestions(request.service_category)
        
        # Prepare and return response
        return ChatResponse(
            response=ai_response,
            conversation_id=conversation_id,
            service_category=request.service_category,
            timestamp=datetime.now().isoformat(),
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API is working.
    
    Returns:
        HealthResponse with status information
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version=settings.APP_VERSION
    )

@router.get("/", include_in_schema=False)
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Dictionary with API info
    """
    return {
        "app": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "documentation": "/docs",
        "health": "/health"
    }