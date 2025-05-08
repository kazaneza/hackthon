"""
API routes for the Bank of Kigali AI Assistant with improved conversation memory.
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

def create_context_message(user_id: str) -> Optional[ChatMessage]:
    """
    Create a system message with conversation context.
    
    Args:
        user_id: User ID to get context for
        
    Returns:
        System message with context, or None if no context available
    """
    # Get previous messages
    previous_messages = message_store.get_messages(user_id)
    if not previous_messages:
        return None
    
    # Try to get conversation summary
    summary = None
    if hasattr(message_store, 'get_conversation_summary'):
        summary = message_store.get_conversation_summary(user_id)
    
    # If no summary exists, create one
    if not summary and len(previous_messages) >= 2:
        summary = create_conversation_summary(
            previous_messages,
            max_length=getattr(settings, 'CONVERSATION_SUMMARY_LENGTH', 200)
        )
        if summary and hasattr(message_store, 'set_conversation_summary'):
            message_store.set_conversation_summary(user_id, summary)
    
    # Extract user information
    user_info = extract_user_information(previous_messages)
    
    # Create context content
    context_parts = []
    
    # Add user information if available
    if user_info:
        user_info_text = "User information: "
        if "name" in user_info:
            user_info_text += f"Name is {user_info['name']}. "
        context_parts.append(user_info_text)
    
    # Add conversation summary if available
    if summary:
        context_parts.append(f"Conversation summary: {summary}")
    else:
        # Create basic context from recent messages
        context = "Recent conversation: "
        for i, msg in enumerate(previous_messages[-3:]):  # Last 3 messages
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                prefix = "User: " if msg.role == "user" else "Assistant: "
                context += f"{prefix}{msg.content[:50]}{'...' if len(msg.content) > 50 else ''} "
        context_parts.append(context)
    
    # Create system message with context
    return ChatMessage(
        role="system",
        content="\n".join(context_parts)
    )

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    document_ai_service: DocumentBasedAIService = Depends(get_document_ai_service)
):
    try:
        # Use provided user_id or generate an anonymous one
        user_id = request.user_id or f"anonymous-{uuid4()}"
        
        # Create context message with conversation history and user info
        context_message = create_context_message(user_id)
        
        # Add new user messages to store
        for msg in request.messages:
            if msg.role == "user":  # Only store user messages
                message_store.add_message(user_id, msg)
        
        # Create combined messages list with context
        all_messages = []
        
        # Add context message if available
        if context_message:
            all_messages.append(context_message)
        
        # Add any system messages from the current request
        for msg in request.messages:
            if hasattr(msg, 'role') and msg.role == "system":
                all_messages.append(msg)
        
        # Add user and assistant messages from the current request
        for msg in request.messages:
            if hasattr(msg, 'role') and msg.role in ["user", "assistant"]:
                all_messages.append(msg)
        
        # Generate response using document-based AI service
        result = await document_ai_service.generate_response(
            service_category=request.service_category,
            messages=all_messages
        )
        
        # Extract response - don't include sources in client-facing response
        ai_response = result["response"]
        
        # Store AI response in message store
        ai_message = ChatMessage(role="assistant", content=ai_response)
        message_store.add_message(user_id, ai_message)
        
        # Update conversation summary if needed
        if hasattr(message_store, 'set_conversation_summary'):
            all_stored_messages = message_store.get_messages(user_id)
            updated_summary = create_conversation_summary(
                all_stored_messages,
                max_length=getattr(settings, 'CONVERSATION_SUMMARY_LENGTH', 200)
            )
            if updated_summary:
                message_store.set_conversation_summary(user_id, updated_summary)
        
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

@router.post("/chat/clear/{user_id}")
async def clear_chat_history(user_id: str):
    """
    Clear the conversation history for a user.
    
    Args:
        user_id: User ID to clear history for
        
    Returns:
        Dictionary with success message
    """
    try:
        # Handle different message store implementations
        if hasattr(message_store, 'clear_user_data'):
            message_store.clear_user_data(user_id)
        else:
            # Fallback for basic implementation
            if hasattr(message_store, 'messages') and user_id in message_store.messages:
                del message_store.messages[user_id]
            
            # Clear conversation summary if it exists
            if hasattr(message_store, 'conversation_summaries') and hasattr(message_store.conversation_summaries, 'get') and user_id in message_store.conversation_summaries:
                del message_store.conversation_summaries[user_id]
        
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
    """
    Get the current conversation context for a user.
    Useful for debugging and development purposes.
    
    Args:
        user_id: User ID to get context for
        
    Returns:
        Dictionary with conversation context info
    """
    try:
        messages = message_store.get_messages(user_id)
        
        # Get conversation summary if available
        summary = None
        if hasattr(message_store, 'get_conversation_summary'):
            summary = message_store.get_conversation_summary(user_id)
        
        # Extract user information
        user_info = extract_user_information(messages)
        
        # Format messages for display
        message_info = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                message_info.append({
                    "role": msg.role,
                    "content": msg.content[:100] + ("..." if len(msg.content) > 100 else "")
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

# Copy these from your original routes.py file:

@router.get("/documents/status")
async def document_status(
    document_ai_service: DocumentBasedAIService = Depends(get_document_ai_service)
):
    """
    Get document processing status.
    
    Returns:
        Dictionary with document statistics
    """
    try:
        # Get document processor from service
        doc_processor = document_ai_service.document_processor
        
        # Check if vector store exists
        vector_store_exists = doc_processor.load_vector_store()
        
        # Count documents in each category
        document_counts = {}
        total_docs = 0
        
        for category in doc_processor.categories:
            category_path = os.path.join(doc_processor.documents_base_path, category)
            
            if os.path.exists(category_path):
                # Count PDF files in category
                pdf_files = [f for f in os.listdir(category_path) if f.lower().endswith('.pdf')]
                document_counts[category] = len(pdf_files)
                total_docs += len(pdf_files)
        
        return {
            "vector_store_initialized": vector_store_exists,
            "document_counts": document_counts,
            "total_documents": total_docs,
            "categories": doc_processor.categories
        }
        
    except Exception as e:
        logger.error(f"Error in document status endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting document status: {str(e)}")

@router.post("/documents/reindex")
async def reindex_documents(
    document_ai_service: DocumentBasedAIService = Depends(get_document_ai_service)
):
    """
    Force reindexing of all documents.
    
    Returns:
        Dictionary with reindexing status
    """
    try:
        # Get document processor from service
        doc_processor = document_ai_service.document_processor
        
        # Get current vector store path
        vector_store_path = os.path.join(doc_processor.documents_base_path, "chroma_db")
        
        # Delete vector store if it exists
        if os.path.exists(vector_store_path):
            import shutil
            shutil.rmtree(vector_store_path)
            logger.info(f"Deleted existing vector store at {vector_store_path}")
        
        # Load and process documents
        documents = doc_processor.load_documents()
        document_chunks = doc_processor.process_documents(documents)
        
        # Create vector store
        doc_processor.create_vector_store(document_chunks)
        
        return {
            "status": "success",
            "message": f"Reindexed {len(documents)} documents into {len(document_chunks)} chunks",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in document reindex endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reindexing documents: {str(e)}")

@router.get("/documents/search")
async def search_documents(
    query: str,
    limit: Optional[int] = 5,
    document_ai_service: DocumentBasedAIService = Depends(get_document_ai_service)
):
    """
    Search documents with a query string.
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        Dictionary with search results
    """
    try:
        # Get document processor from service
        doc_processor = document_ai_service.document_processor
        
        # Check if vector store exists
        if not doc_processor.vector_store:
            doc_processor.load_vector_store()
            
            if not doc_processor.vector_store:
                raise HTTPException(
                    status_code=404, 
                    detail="Vector store not initialized. Please upload documents first."
                )
        
        # Get retriever
        retriever = doc_processor.get_retriever()
        
        # Update search parameters
        retriever.search_kwargs["k"] = limit
        
        # Perform search
        docs = retriever.get_relevant_documents(query)
        
        # Format results
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "category": doc.metadata.get("category", "unknown"),
                "product": doc.metadata.get("product_type", "unknown"),
                "file": doc.metadata.get("filename", "unknown"),
                "page": doc.metadata.get("page", 0) + 1  # Adjust 0-index to 1-index
            })
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in document search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")