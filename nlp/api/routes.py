"""
API routes with enhanced logging for debugging memory issues.
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
from core.enhanced_message_store import EnhancedMessageStore, extract_user_information_from_qa
from core.prompts import get_follow_up_suggestions
from config.settings import settings
from utils.logging_config import logger

# Load environment variables
load_dotenv()

# Create router
router = APIRouter()

# Initialize services with enhanced message store
message_store = EnhancedMessageStore(
    max_pairs=3,  # Keep last 3 Q&A pairs
    expiry_seconds=settings.MESSAGE_EXPIRY_SECONDS
)

# Dependencies (unchanged)
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

def create_context_from_conversations(user_id: str) -> str:
    """
    Create context from recent conversations for the AI.
    
    Args:
        user_id: User identifier
        
    Returns:
        Formatted context string
    """
    # Get user info and recent conversations
    user_info = message_store.get_user_info(user_id)
    recent_conversations = message_store.get_recent_conversations(user_id)
    
    # Enhanced logging for debugging
    logger.info(f"[CONTEXT] Creating context for user {user_id}")
    logger.info(f"[CONTEXT] User info: {user_info}")
    logger.info(f"[CONTEXT] Number of recent conversations: {len(recent_conversations)}")
    for i, conv in enumerate(recent_conversations):
        logger.info(f"[CONTEXT] Conversation {i+1}: Q='{conv.question[:50]}...' A='{conv.answer[:50]}...'")
    
    context_parts = []
    
    # Add user information
    if user_info:
        context_parts.append("=== USER INFORMATION ===")
        if "name" in user_info:
            context_parts.append(f"Customer name: {user_info['name']}")
            logger.info(f"[CONTEXT] Adding name to context: {user_info['name']}")
        if "account_number" in user_info:
            context_parts.append(f"Account number: {user_info['account_number']}")
        if "customer_id" in user_info:
            context_parts.append(f"Customer ID: {user_info['customer_id']}")
        context_parts.append("")
    else:
        logger.warning(f"[CONTEXT] No user info found for user {user_id}")
    
    # Add recent conversation history
    if recent_conversations:
        context_parts.append("=== RECENT CONVERSATION HISTORY ===")
        for i, pair in enumerate(recent_conversations, 1):
            context_parts.append(f"[Q{i}] User: {pair.question}")
            context_parts.append(f"[A{i}] Assistant: {pair.answer}")
            context_parts.append("")
    else:
        logger.warning(f"[CONTEXT] No recent conversations found for user {user_id}")
    
    # Add CRITICAL rules if we have user info
    if user_info and "name" in user_info:
        context_parts.append("=== CRITICAL REMINDERS ===")
        context_parts.append(f"- The customer's name is {user_info['name']}")
        context_parts.append("- Use their name naturally in your responses")
        context_parts.append("- When they ask 'what is my name?', respond confidently with their name")
        context_parts.append("- NEVER claim you don't have access to their information")
        logger.info(f"[CONTEXT] Added critical reminders for name: {user_info['name']}")
    
    final_context = "\n".join(context_parts)
    logger.info(f"[CONTEXT] Final context length: {len(final_context)} characters")
    logger.debug(f"[CONTEXT] Full context:\n{final_context}")
    
    return final_context

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    document_ai_service: DocumentBasedAIService = Depends(get_document_ai_service)
):
    try:
        # Use provided user_id or generate one
        user_id = request.user_id or f"anonymous-{uuid4()}"
        
        logger.info(f"[CHAT] Starting chat request for user {user_id}")
        logger.info(f"[CHAT] Service category: {request.service_category}")
        logger.info(f"[CHAT] Number of messages in request: {len(request.messages)}")
        
        # Get the last user message
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            user_message = "Hello"
        
        logger.info(f"[CHAT] User message: '{user_message}'")
        
        # Check current state before processing
        logger.info(f"[CHAT] Current state for user {user_id}:")
        current_user_info = message_store.get_user_info(user_id)
        current_conversations = message_store.get_recent_conversations(user_id)
        logger.info(f"[CHAT] - Current user info: {current_user_info}")
        logger.info(f"[CHAT] - Current conversations count: {len(current_conversations)}")
        
        # Create context from stored conversations
        conversation_context = create_context_from_conversations(user_id)
        
        # Create system message with context
        messages_for_ai = []
        
        if conversation_context:
            system_message = ChatMessage(
                role="system",
                content=f"""You are ALICE, Bank of Kigali's AI assistant.\n\n{conversation_context}\n\nUse this context to provide personalized, helpful responses. Pay special attention to the user information and previous conversation history.
                
                IMPORTANT: When you have customer information in the context, use it! Never deny having access to information that's clearly provided above."""
            )
            messages_for_ai.append(system_message)
            logger.info(f"[CHAT] Created system message with context length: {len(system_message.content)}")
        else:
            logger.warning(f"[CHAT] No context available for user {user_id}")
        
        # Add the current user message
        messages_for_ai.append(ChatMessage(role="user", content=user_message))
        
        # Log what we're sending to AI
        logger.info(f"[CHAT] Sending {len(messages_for_ai)} messages to AI:")
        for i, msg in enumerate(messages_for_ai):
            logger.info(f"[CHAT] Message {i+1}: {msg.role} - Length: {len(msg.content)}")
            if msg.role == "system":
                logger.debug(f"[CHAT] System message preview: {msg.content[:200]}...")
            else:
                logger.info(f"[CHAT] {msg.role} message: {msg.content}")
        
        # Generate response
        result = await document_ai_service.generate_response(
            service_category=request.service_category,
            messages=messages_for_ai
        )
        
        ai_response = result["response"]
        logger.info(f"[CHAT] AI response: '{ai_response}'")
        
        # Extract any new user information from this Q&A
        detected_user_info = extract_user_information_from_qa(user_message, ai_response)
        logger.info(f"[CHAT] Detected user info from this Q&A: {detected_user_info}")
        
        # Store this Q&A pair
        message_store.add_qa_pair(
            user_id=user_id,
            question=user_message,
            answer=ai_response,
            detected_user_info=detected_user_info
        )
        
        # Check state after storing
        logger.info(f"[CHAT] State after storing Q&A:")
        final_user_info = message_store.get_user_info(user_id)
        final_conversations = message_store.get_recent_conversations(user_id)
        logger.info(f"[CHAT] - Final user info: {final_user_info}")
        logger.info(f"[CHAT] - Final conversations count: {len(final_conversations)}")
        
        # Generate response
        response = ChatResponse(
            response=ai_response,
            conversation_id=str(uuid4()),
            service_category=request.service_category,
            timestamp=datetime.now().isoformat(),
            suggestions=get_follow_up_suggestions(request.service_category)
        )
        
        logger.info(f"[CHAT] Completed chat request for user {user_id}")
        return response
        
    except Exception as e:
        logger.error(f"[CHAT] Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Debugging endpoint to check conversation state
@router.get("/debug/conversations/{user_id}")
async def debug_conversations(user_id: str):
    """Debug endpoint to check stored conversations for a user."""
    try:
        user_info = message_store.get_user_info(user_id)
        conversations = message_store.get_recent_conversations(user_id)
        
        logger.info(f"[DEBUG] Checking conversations for user {user_id}")
        logger.info(f"[DEBUG] User info: {user_info}")
        logger.info(f"[DEBUG] Number of conversations: {len(conversations)}")
        
        formatted_conversations = []
        for i, pair in enumerate(conversations):
            formatted_conversations.append({
                "index": i,
                "question": pair.question,
                "answer": pair.answer,
                "timestamp": pair.timestamp.isoformat(),
                "user_info_extracted": pair.user_info
            })
            logger.info(f"[DEBUG] Conversation {i+1}: Q='{pair.question}' A='{pair.answer}'")
        
        # Generate context to see what AI would receive
        context = create_context_from_conversations(user_id)
        logger.info(f"[DEBUG] Generated context:\n{context}")
        
        debug_response = {
            "user_id": user_id,
            "user_info": user_info,
            "conversation_count": len(conversations),
            "conversations": formatted_conversations,
            "generated_context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"[DEBUG] Debug response prepared")
        return debug_response
        
    except Exception as e:
        logger.error(f"[DEBUG] Error in debug endpoint: {str(e)}", exc_info=True)
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
async def get_user_context(user_id: str):
    """Get the current conversation context for a user."""
    try:
        user_info = message_store.get_user_info(user_id)
        conversations = message_store.get_recent_conversations(user_id)
        context = create_context_from_conversations(user_id)
        
        return {
            "user_id": user_id,
            "user_info": user_info,
            "conversation_count": len(conversations),
            "full_context": context,
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