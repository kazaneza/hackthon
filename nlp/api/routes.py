"""
API routes for the Bank of Kigali AI Assistant.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import Optional, List
from uuid import uuid4
from datetime import datetime
import json
import os
from dotenv import load_dotenv

from api.models import ChatRequest, ChatResponse, HealthResponse
from core.ai_service import LangChainService
from core.document_qa import DocumentBasedAIService  # New import
from core.message_store import MessageStore
from core.prompts import get_follow_up_suggestions
from config.settings import settings
from utils.logging_config import logger

# Load environment variables
load_dotenv()

# Create router
router = APIRouter()

# Initialize services
message_store = MessageStore(
    max_messages=settings.MAX_STORED_MESSAGES,
    expiry_seconds=settings.MESSAGE_EXPIRY_SECONDS
)

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

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    document_ai_service: DocumentBasedAIService = Depends(get_document_ai_service)
):
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
            # Ensure the role is valid
            if hasattr(prev_msg, 'role') and prev_msg.role in ['system', 'assistant', 'user', 'function', 'tool', 'developer']:
                if prev_msg.content not in current_message_contents:
                    context_messages.append(prev_msg)
        
        # Create combined messages list with context
        all_messages = context_messages + request.messages
        
        # Generate response using document-based AI service
        result = await document_ai_service.generate_response(
            service_category=request.service_category,
            messages=all_messages
        )
        
        # Extract response and sources
        ai_response = result["response"]
        sources = result.get("sources", [])
        
        # Add document sources to response if available
        if sources:
            # Format sources information
            sources_text = "\n\nThis information comes from the following documents:\n"
            for i, source in enumerate(sources):
                sources_text += f"{i+1}. {source['product']} ({source['category'].upper()}) - {source['file']} (Page {source['page']})\n"
            
            # Append sources to response
            ai_response += sources_text
        
        # Store AI response in message store
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