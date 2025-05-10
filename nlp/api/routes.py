"""
API routes with enhanced context handling and full conversation history.
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

def create_conversation_messages(user_id: str, current_user_message: str) -> List[ChatMessage]:
    """
    Create the full conversation history including all stored Q&A pairs.
    
    Args:
        user_id: User identifier
        current_user_message: The current message from the user
        
    Returns:
        List of ChatMessage objects for the AI
    """
    messages = []
    
    # Get user info and recent conversations
    user_info = message_store.get_user_info(user_id)
    recent_conversations = message_store.get_recent_conversations(user_id)
    
    # Create system message with user context
    system_context = []
    system_context.append("You are ALICE, Bank of Kigali's AI assistant.")
    system_context.append("")
    
    if user_info:
        system_context.append("CUSTOMER INFORMATION:")
        if "name" in user_info:
            system_context.append(f"- Name: {user_info['name']}")
        if "account_number" in user_info:
            system_context.append(f"- Account Number: {user_info['account_number']}")
        if "customer_id" in user_info:
            system_context.append(f"- Customer ID: {user_info['customer_id']}")
        system_context.append("")
    
    # Add critical instructions
    if user_info and "name" in user_info:
        system_context.append("IMPORTANT INSTRUCTIONS:")
        system_context.append(f"- When the customer asks 'What is my name?', respond: 'Your name is {user_info['name']}.'")
        system_context.append("- Use their name naturally in your responses")
        system_context.append("- Never claim you don't have access to information that's shown above")
        system_context.append("")
    
    system_context.append("Be helpful, professional, and personalized when you have the customer's information.")
    
    # Create system message
    system_message = ChatMessage(
        role="system",
        content="\n".join(system_context)
    )
    messages.append(system_message)
    
    # Add all previous conversations as actual chat messages
    for pair in recent_conversations:
        # Add user message
        messages.append(ChatMessage(
            role="user",
            content=pair.question
        ))
        # Add assistant message
        messages.append(ChatMessage(
            role="assistant",
            content=pair.answer
        ))
    
    # Add the current user message
    messages.append(ChatMessage(
        role="user",
        content=current_user_message
    ))
    
    logger.info(f"[CHAT] Created {len(messages)} messages for AI (1 system + {len(recent_conversations)*2} history + 1 current)")
    
    return messages

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
        
        # Create full conversation history including all stored messages
        messages_for_ai = create_conversation_messages(user_id, user_message)
        
        # Log what we're sending to AI
        logger.info(f"[CHAT] Sending {len(messages_for_ai)} messages to AI:")
        for i, msg in enumerate(messages_for_ai):
            logger.info(f"[CHAT] Message {i+1}: {msg.role} - Length: {len(msg.content)}")
            if msg.role == "system":
                logger.info(f"[CHAT] FULL SYSTEM MESSAGE:")
                logger.info("=" * 80)
                logger.info(msg.content)
                logger.info("=" * 80)
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
        
        # Show what messages would be sent to AI
        test_message = "Test message"
        ai_messages = create_conversation_messages(user_id, test_message)
        
        debug_response = {
            "user_id": user_id,
            "user_info": user_info,
            "conversation_count": len(conversations),
            "conversations": formatted_conversations,
            "ai_messages_preview": [
                {
                    "role": msg.role,
                    "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                }
                for msg in ai_messages
            ],
            "full_message_count": len(ai_messages),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"[DEBUG] Debug response prepared")
        return debug_response
        
    except Exception as e:
        logger.error(f"[DEBUG] Error in debug endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

# Clear chat history endpoint
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

# Get user context endpoint
@router.get("/chat/context/{user_id}")
async def get_user_context(user_id: str):
    """Get the current conversation context for a user."""
    try:
        user_info = message_store.get_user_info(user_id)
        conversations = message_store.get_recent_conversations(user_id)
        
        # Create example messages to show what AI would receive
        example_messages = create_conversation_messages(user_id, "What banking services do you offer?")
        
        return {
            "user_id": user_id,
            "user_info": user_info,
            "conversation_count": len(conversations),
            "stored_conversations": [
                {
                    "question": pair.question,
                    "answer": pair.answer,
                    "timestamp": pair.timestamp.isoformat()
                }
                for pair in conversations
            ],
            "ai_messages_preview": [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in example_messages
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting chat context for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting conversation context: {str(e)}")

# Health check endpoint
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API is working."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version=settings.APP_VERSION
    )

# Root endpoint
@router.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "app": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "documentation": "/docs",
        "health": "/health"
    }