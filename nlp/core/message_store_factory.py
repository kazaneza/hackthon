"""
Factory for creating message stores with different backends for the Bank of Kigali AI Assistant.
"""

from typing import Any
from config.settings import settings
from utils.logging_config import logger

# Use direct import for the MessageStore class
from core.message_store import MessageStore

def get_best_available_message_store():
    """
    Get the best available message store based on configuration.
    
    Currently, this returns the standard MessageStore with improved settings.
    Future implementations could add support for Redis or database backends.
    
    Returns:
        Message store instance
    """
    # Check for Redis support if enabled in settings
    redis_enabled = getattr(settings, 'REDIS_ENABLED', False)
    
    if redis_enabled:
        try:
            # Try to import Redis
            import redis
            
            # Check if we've implemented RedisMessageStore
            try:
                from core.redis_message_store import RedisMessageStore
                
                # Create Redis client and test connection
                redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
                redis_client = redis.from_url(redis_url)
                redis_client.ping()  # Will raise error if connection fails
                
                logger.info("Using Redis-backed message store for better persistence")
                return RedisMessageStore(
                    max_messages=settings.MAX_STORED_MESSAGES,
                    expiry_seconds=settings.MESSAGE_EXPIRY_SECONDS,
                    redis_url=redis_url
                )
            except (ImportError, AttributeError):
                logger.warning("Redis is enabled but RedisMessageStore is not implemented")
            except Exception as e:
                logger.warning(f"Redis connection failed: {str(e)}")
        except ImportError:
            logger.warning("Redis package not installed, falling back to in-memory store")
    
    # Use standard in-memory message store
    logger.info(f"Using in-memory message store with {settings.MAX_STORED_MESSAGES} "
                f"messages and {settings.MESSAGE_EXPIRY_SECONDS}s expiry")
    return MessageStore(
        max_messages=settings.MAX_STORED_MESSAGES,
        expiry_seconds=settings.MESSAGE_EXPIRY_SECONDS
    )

def create_message_store():
    """
    Legacy method for backward compatibility.
    
    Returns:
        Message store instance
    """
    return get_best_available_message_store()