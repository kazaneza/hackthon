"""
Enhanced configuration settings for the Bank of Kigali AI Assistant application.
"""

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # App info
    APP_TITLE: str = "Bank of Kigali ALICE AI Assistant"
    APP_DESCRIPTION: str = "Bank of Kigali AI Assistant"
    APP_VERSION: str = "1.0.0"
    
    # LLM settings
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4o-mini"  # Compatible with LangChain
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 500
    
    # LangChain settings
    LANGCHAIN_TRACING: bool = os.environ.get("LANGCHAIN_TRACING", "false").lower() == "true"
    LANGCHAIN_PROJECT: str = os.environ.get("LANGCHAIN_PROJECT", "bank-of-kigali-assistant")
    
    # Message store settings
    MAX_STORED_MESSAGES: int = 10  # Increased from 5 to 10
    MESSAGE_EXPIRY_SECONDS: int = 1800  # Increased from 120s (2min) to 1800s (30min)
    
    # Conversation memory settings
    ENABLE_CONVERSATION_SUMMARY: bool = True
    CONVERSATION_SUMMARY_LENGTH: int = 200
    
    # Redis settings (for future use)
    REDIS_ENABLED: bool = os.environ.get("REDIS_ENABLED", "false").lower() == "true"
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = int(os.environ.get("PORT", 8888))
    RELOAD: bool = False 
    LOG_LEVEL: str = "info"
    
    # CORS settings
    CORS_ORIGINS: list = ["*"]  # In production, specify exact origins
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["*"]
    CORS_ALLOW_HEADERS: list = ["*"]

# Create settings instance
settings = Settings()