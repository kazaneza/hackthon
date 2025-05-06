"""
Bank of Kigali AI Assistant
---
A FastAPI application that leverages OpenAI to answer banking queries related to:
- Queue management
- Feedback collection
- Personalized banking
- Upselling
- Executive services
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import os
import openai
from datetime import datetime
import logging
import json
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bank of Kigali AI Assistant API",
    description="API for Bank of Kigali AI Assistant using OpenAI",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key from .env file or environment variables
try:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not found in environment variables or .env file")
except Exception as e:
    logger.error(f"Error loading OpenAI API key: {e}")

# Models for request/response
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant/system)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    service_category: str = Field(..., description="Service category for the query")
    user_id: Optional[str] = Field(None, description="User ID for personalization")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI assistant response")
    conversation_id: str = Field(..., description="Unique conversation ID")
    service_category: str = Field(..., description="Service category that was processed")
    timestamp: str = Field(..., description="Timestamp of the response")
    suggestions: Optional[List[str]] = Field(None, description="Suggested follow-up questions")

# System prompts by service category
SYSTEM_PROMPTS = {
    "queue_management": """You are the Bank of Kigali's AI assistant specializing in queue management.
Help customers understand wait times, schedule appointments, and navigate branch services.
Provide concise, accurate information about the Bank of Kigali's queuing system, how to use the ticket system, 
and alternative channels to avoid queues. Be courteous and professional.""",
    
    "feedback_collection": """You are the Bank of Kigali's AI assistant specializing in collecting customer feedback.
Your role is to gather detailed feedback on banking experiences, helping customers express their concerns or compliments.
Ask relevant follow-up questions to get specific details. Be empathetic and thank customers for their feedback.""",
    
    "personalized_banking": """You are the Bank of Kigali's AI assistant specializing in personalized banking services.
Assist customers with account-related inquiries, provide information about balances, transactions, and account features.
Be helpful and informative while maintaining a focus on security and privacy. Do not ask for sensitive information.""",
    
    "upselling": """You are the Bank of Kigali's AI assistant specializing in recommending additional banking products.
Offer relevant upselling suggestions based on customer needs. Highlight Bank of Kigali's credit cards, loans, 
investment opportunities, and special packages. Be persuasive but not pushy.""",
    
    "executive_services": """You are the Bank of Kigali's AI assistant specializing in executive banking services.
Provide high-level analytics, wealth management advice, and premium banking services information.
Be detailed, professional, and cater to high-value clients with sophisticated financial needs.""",
    
    "general": """You are the Bank of Kigali's AI assistant. 
Provide helpful, accurate information about Bank of Kigali services and answer customer queries professionally.
When unsure, acknowledge limitations and suggest speaking with a human banker. 
Maintain a warm, professional tone that represents the Bank of Kigali brand."""
}

# Configuration for OpenAI
# Configuration for OpenAI
class OpenAIConfig:
    model = "gpt-4o-mini"  # Changed from "gpt-4-turbo" to "gpt-4o-mini"
    temperature = 0.7
    max_tokens = 500

# Dependency to get OpenAI client
def get_openai_client():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return client

# Helper function to prepare messages for OpenAI
def prepare_messages(service_category: str, user_messages: List[ChatMessage]) -> List[Dict[str, str]]:
    system_prompt = SYSTEM_PROMPTS.get(service_category, SYSTEM_PROMPTS["general"])
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend([{"role": m.role, "content": m.content} for m in user_messages])
    
    return messages

# Function to generate follow-up suggestions
def generate_suggestions(service_category: str, last_message: str) -> List[str]:
    suggestions = {
        "queue_management": [
            "What's the current wait time at the main branch?",
            "How can I schedule an appointment?",
            "What services can I access without visiting a branch?"
        ],
        "feedback_collection": [
            "I want to submit feedback about my recent branch visit",
            "How do you use customer feedback to improve services?",
            "Where can I see responses to my previous feedback?"
        ],
        "personalized_banking": [
            "Can you explain my recent account activity?",
            "What banking features are available on my account?",
            "How can I set up automatic payments?"
        ],
        "upselling": [
            "What loan products would suit my needs?",
            "Are there any special offers on credit cards right now?",
            "Tell me about your investment opportunities"
        ],
        "executive_services": [
            "Can you provide a deposit trend analysis?",
            "What wealth management services do you offer?",
            "How can I access your premium banking package?"
        ],
        "general": [
            "What are your operating hours?",
            "Where are your ATMs located?",
            "How can I open a new account?"
        ]
    }
    
    return suggestions.get(service_category, suggestions["general"])

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    openai_client: Any = Depends(get_openai_client)
):
    try:
        # Prepare messages for OpenAI
        messages = prepare_messages(request.service_category, request.messages)
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model=OpenAIConfig.model,
            messages=messages,
            temperature=OpenAIConfig.temperature,
            max_tokens=OpenAIConfig.max_tokens
        )
        
        # Extract response
        ai_response = response.choices[0].message.content
        
        # Generate conversation ID if not provided
        conversation_id = str(uuid.uuid4())
        
        # Generate follow-up suggestions
        suggestions = generate_suggestions(request.service_category, ai_response)
        
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

# Route for health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Documentation endpoint
@app.get("/", include_in_schema=False)
async def root():
    return {
        "app": "Bank of Kigali AI Assistant API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred"}
        )

# Sample implementation for future database connection
class DatabaseManager:
    """
    Placeholder for future database implementation
    This will be used to store conversations, user data, and analytics
    """
    
    def __init__(self):
        self.connection = None
    
    async def connect(self):
        # Will be implemented when database specifics are known
        logger.info("Database connection would be established here")
        pass
    
    async def store_conversation(self, user_id, conversation_id, messages):
        # Will store conversation history
        logger.info(f"Storing conversation {conversation_id} for user {user_id}")
        pass
    
    async def get_user_data(self, user_id):
        # Will retrieve user profile data for personalization
        logger.info(f"Getting data for user {user_id}")
        return {}

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8888))
    
    print(f"Starting Bank of Kigali AI Assistant API")
    print(f"API documentation available at: http://localhost:{port}/docs")
    print(f"API endpoint available at: http://localhost:{port}/chat")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )