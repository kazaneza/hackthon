"""
API routes for the Bank of Kigali AI Assistant with enhanced debugging for memory issues.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import Optional, List
from uuid import uuid4
from datetime import datetime
import json
import os
from dotenv import load_dotenv

from api.models import ChatRequest, ChatResponse, HealthResponse, ChatMessage
from core.ai_service import LangChainService
from core.document_qa import DocumentBasedAIService
from core.message_store_factory import get_best_available_message_store
from core.message_store import create_conversation_summary, extract_user_information
from core.prompts import get_follow_up_suggestions
from config.settings import settings
from utils.logging_config import logger

# Load environment variables
load_dotenv()

# Create router
router = APIRouter()

# Initialize services with improved message store
message_store = get_best_available_message_store()

# Dependencies
async def get_langchain_service():
    """Dependency to get LangChain service."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    return LangChainService(api_key=settings.OPENAI_API_KEY)

async def get_document_ai_service():
    """Dependency to get document-based AI service."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    return DocumentBasedAIService(api_key=settings.OPENAI_API_KEY)

def create_comprehensive_context_message(user_id: str, request_messages: List[ChatMessage]) -> ChatMessage:
    """
    Create a comprehensive system message with all conversation context.
    This is the main fix for the memory issue.
    """
    # Get ALL previous messages from store
    stored_messages = message_store.get_messages(user_id)
    
    # Debug logging
    logger.info(f"Creating context for user {user_id}")
    logger.info(f"Stored messages count: {len(stored_messages)}")
    
    # Combine ALL messages (stored + current request)
    all_messages = stored_messages + [msg for msg in request_messages if msg.role != "system"]
    
    # Extract user information from ALL messages
    user_info = extract_user_information(all_messages)
    
    # Debug logging for user info
    logger.info(f"Extracted user info: {user_info}")
    
    # Get conversation summary
    summary = None
    if hasattr(message_store, 'get_conversation_summary'):
        summary = message_store.get_conversation_summary(user_id)
    
    # Create summary if needed
    if not summary and len(all_messages) >= 4:
        summary = create_conversation_summary(all_messages, max_length=250)
        if summary and hasattr(message_store, 'set_conversation_summary'):
            message_store.set_conversation_summary(user_id, summary)
    
    # Build the context content
    context_parts = []
    
    # START WITH CRITICAL USER INFORMATION
    context_parts.append("=== CRITICAL USER INFORMATION ===")
    if user_info:
        if "name" in user_info:
            context_parts.append(f"CUSTOMER NAME: {user_info['name']}")
            context_parts.append(f"ALWAYS address the customer as '{user_info['name']}' - do NOT say you don't know their name")
        if "account_number" in user_info:
            context_parts.append(f"ACCOUNT NUMBER: {user_info['account_number']}")
        if "customer_id" in user_info:
            context_parts.append(f"CUSTOMER ID: {user_info['customer_id']}")
    else:
        context_parts.append("No user information available yet")
    
    context_parts.append("")  # Empty line for spacing
    
    # Add conversation history
    if summary:
        context_parts.append("=== CONVERSATION SUMMARY ===")
        context_parts.append(summary)
    else:
        context_parts.append("=== RECENT CONVERSATION ===")
        # Show recent messages
        for i, msg in enumerate(all_messages[-8:]):  # Last 8 messages
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                prefix = f"[{i+1}] {msg.role.upper()}: "
                content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                context_parts.append(prefix + content)
    
    context_parts.append("")  # Empty line
    
    # Critical instructions
    context_parts.append("=== CRITICAL INSTRUCTIONS ===")
    context_parts.append("1. You are ALICE, Bank of Kigali's AI assistant")
    context_parts.append("2. ALWAYS use the customer's name when you know it")
    context_parts.append("3. NEVER say you don't have access to their name if it's provided above")
    context_parts.append("4. Maintain conversation continuity by referencing previous interactions")
    context_parts.append("5. Be warm, professional, and personalized")
    
    # Create the system message
    context_content = "\n".join(context_parts)
    
    # Debug logging
    logger.info(f"Context message length: {len(context_content)} characters")
    logger.info(f"Context preview: {context_content[:200]}...")
    
    return ChatMessage(
        role="system",
        content=context_content
    )

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    document_ai_service: DocumentBasedAIService = Depends(get_document_ai_service)
):
    try:
        # Use provided user_id or generate one
        user_id = request.user_id or f"anonymous-{uuid4()}"
        
        # Debug logging
        logger.info(f"Processing chat request for user {user_id}")
        logger.info(f"Service category: {request.service_category}")
        logger.info(f"Incoming messages: {len(request.messages)}")
        
        # Get current messages from store and refresh expiry
        current_stored_messages = message_store.get_messages(user_id)
        logger.info(f"Current stored messages: {len(current_stored_messages)}")
        
        # Store user messages from the request
        for msg in request.messages:
            if msg.role == "user":
                message_store.add_message(user_id, msg)
                logger.info(f"Stored user message: '{msg.content[:50]}...'")
        
        # Get updated messages after storing
        all_stored_messages = message_store.get_messages(user_id)
        logger.info(f"Total stored messages after update: {len(all_stored_messages)}")
        
        # Create comprehensive context message
        context_message = create_comprehensive_context_message(user_id, request.messages)
        
        # Prepare messages for AI
        messages_for_ai = [context_message]
        
        # Add current request messages (excluding system messages)
        for msg in request.messages:
            if msg.role != "system":
                messages_for_ai.append(msg)
        
        # Debug: Log what we're sending to AI
        logger.info(f"Sending {len(messages_for_ai)} messages to AI")
        for i, msg in enumerate(messages_for_ai):
            logger.info(f"Message {i+1}: {msg.role} - {msg.content[:100]}...")
        
        # Generate response
        result = await document_ai_service.generate_response(
            service_category=request.service_category,
            messages=messages_for_ai
        )
        
        ai_response = result["response"]
        
        # Store AI response
        ai_message = ChatMessage(role="assistant", content=ai_response)
        message_store.add_message(user_id, ai_message)
        
        # Update conversation summary
        final_messages = message_store.get_messages(user_id)
        if len(final_messages) >= 4 and hasattr(message_store, 'set_conversation_summary'):
            updated_summary = create_conversation_summary(final_messages, max_length=250)
            if updated_summary:
                message_store.set_conversation_summary(user_id, updated_summary)
                logger.info(f"Updated conversation summary for user {user_id}")
        
        # Generate response
        response = ChatResponse(
            response=ai_response,
            conversation_id=str(uuid4()),
            service_category=request.service_category,
            timestamp=datetime.now().isoformat(),
            suggestions=get_follow_up_suggestions(request.service_category)
        )
        
        # Final debug logging
        logger.info(f"Response generated for user {user_id}")
        logger.info(f"Final stored messages count: {len(final_messages)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Debugging endpoint to check message store state
@router.get("/debug/messages/{user_id}")
async def debug_messages(user_id: str):
    """Debug endpoint to check the current state of message store."""
    try:
        messages = message_store.get_messages(user_id)
        user_info = extract_user_information(messages)
        summary = None
        
        if hasattr(message_store, 'get_conversation_summary'):
            summary = message_store.get_conversation_summary(user_id)
        
        # Format messages for debugging
        formatted_messages = []
        for i, msg in enumerate(messages):
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                formatted_messages.append({
                    "index": i,
                    "role": msg.role,
                    "content_preview": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                    "full_content": msg.content
                })
        
        return {
            "user_id": user_id,
            "total_messages": len(messages),
            "user_info_extracted": user_info,
            "conversation_summary": summary,
            "all_messages": formatted_messages,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

# Rest of your existing routes...
@router.post("/chat/clear/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear the conversation history for a user."""
    try:
        message_store.clear_user_data(user_id)
        logger.info(f"Cleared chat history for user {user_id}")
        
        return {
            "status": "success",
            "message": "Conversation history cleared successfully",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing chat history for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing conversation history: {str(e)}")

@router.get("/chat/context/{user_id}")
async def get_chat_context(user_id: str):
    """Get the current conversation context for a user."""
    try:
        messages = message_store.get_messages(user_id)
        summary = None
        if hasattr(message_store, 'get_conversation_summary'):
            summary = message_store.get_conversation_summary(user_id)
        
        user_info = extract_user_information(messages)
        
        message_info = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                message_info.append({
                    "role": msg.role,
                    "content": msg.content[:100] + ("..." if len(msg.content) > 100 else ""),
                })
        
        return {
            "user_id": user_id,
            "message_count": len(messages),
            "user_info": user_info,
            "conversation_summary": summary,
            "recent_messages": message_info,
            "max_stored_messages": getattr(message_store, 'max_messages', settings.MAX_STORED_MESSAGES),
            "expiry_seconds": getattr(message_store, 'expiry_seconds', settings.MESSAGE_EXPIRY_SECONDS),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting chat context for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting conversation context: {str(e)}")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API is working."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version=settings.APP_VERSION
    )

@router.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "app": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "documentation": "/docs",
        "health": "/health"
    }