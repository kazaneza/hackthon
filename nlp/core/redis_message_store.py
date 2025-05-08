"""
Redis-backed message store for persistent conversation memory across server restarts.
"""

import json
import redis
from datetime import datetime
from typing import Dict, List, Any, Optional
from config.settings import settings
from utils.logging_config import logger

class RedisMessageStore:
    """
    Redis-backed store for user messages with persistent storage.
    Maintains conversational context across server restarts and multiple instances.
    """
    
    def __init__(
        self, 
        max_messages: int = settings.MAX_STORED_MESSAGES,
        expiry_seconds: int = settings.MESSAGE_EXPIRY_SECONDS,
        redis_url: str = settings.REDIS_URL
    ):
        """
        Initialize the Redis message store.
        
        Args:
            max_messages: Maximum number of messages to store per user
            expiry_seconds: Time in seconds after which messages expire
            redis_url: Redis connection URL
        """
        self.redis = redis.from_url(redis_url)
        self.max_messages = max_messages
        self.expiry_seconds = expiry_seconds
        logger.info(f"Redis message store initialized with {max_messages} max messages and {expiry_seconds}s expiry")
    
    def _get_messages_key(self, user_id: str) -> str:
        """Get Redis key for user messages."""
        return f"bk:messages:{user_id}"
    
    def _get_summary_key(self, user_id: str) -> str:
        """Get Redis key for user conversation summary."""
        return f"bk:summary:{user_id}"
    
    def add_message(self, user_id: str, message: Any) -> None:
        """
        Add a message to the store for a user.
        
        Args:
            user_id: Unique identifier for the user
            message: Message object to store
        """
        key = self._get_messages_key(user_id)
        
        # Get current messages
        current_data = self.redis.get(key)
        
        if current_data:
            messages = json.loads(current_data)
        else:
            messages = []
        
        # Serialize the message
        if hasattr(message, 'role') and hasattr(message, 'content'):
            # Handle ChatMessage objects
            message_data = {
                "role": message.role,
                "content": message.content,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Handle other message types
            message_data = {
                "data": str(message),
                "timestamp": datetime.now().isoformat()
            }
        
        # Add new message
        messages.append(message_data)
        
        # Keep only the last max_messages
        if len(messages) > self.max_messages:
            messages = messages[-self.max_messages:]
        
        # Store in Redis with expiry
        self.redis.setex(
            key,
            self.expiry_seconds,
            json.dumps(messages)
        )
    
    def get_messages(self, user_id: str) -> List[Any]:
        """
        Get all messages for a user and refresh their expiry time.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            List of message objects
        """
        key = self._get_messages_key(user_id)
        data = self.redis.get(key)
        
        if data:
            # Reset expiry
            self.redis.expire(key, self.expiry_seconds)
            
            # Deserialize
            messages_data = json.loads(data)
            
            # Convert to message objects
            messages = []
            
            for msg_data in messages_data:
                if "role" in msg_data and "content" in msg_data:
                    # Import here to avoid circular import
                    from api.models import ChatMessage
                    messages.append(ChatMessage(
                        role=msg_data["role"],
                        content=msg_data["content"]
                    ))
                else:
                    # For non-ChatMessage objects, just use the string representation
                    messages.append(msg_data.get("data", ""))
            
            return messages
        
        return []
    
    def set_conversation_summary(self, user_id: str, summary: str) -> None:
        """
        Store a summary of the conversation for a user.
        
        Args:
            user_id: Unique identifier for the user
            summary: Summary of the conversation
        """
        key = self._get_summary_key(user_id)
        
        # Store in Redis with longer expiry (2x message expiry)
        self.redis.setex(
            key,
            self.expiry_seconds * 2,
            summary
        )
    
    def get_conversation_summary(self, user_id: str) -> Optional[str]:
        """
        Get the conversation summary for a user if available.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Conversation summary if available, None otherwise
        """
        key = self._get_summary_key(user_id)
        data = self.redis.get(key)
        
        if data:
            # Reset expiry
            self.redis.expire(key, self.expiry_seconds * 2)
            
            # Return as string
            return data.decode('utf-8')
        
        return None
    
    def clear_user_data(self, user_id: str) -> None:
        """
        Clear all data for a user.
        
        Args:
            user_id: Unique identifier for the user
        """
        # Delete messages and summary
        self.redis.delete(self._get_messages_key(user_id))
        self.redis.delete(self._get_summary_key(user_id))
    
    def get_active_users(self) -> List[str]:
        """
        Get a list of all active users with stored messages.
        
        Returns:
            List of user IDs
        """
        # Get all keys matching the message pattern
        keys = self.redis.keys("bk:messages:*")
        
        # Extract user IDs from keys
        user_ids = [key.decode('utf-8').split(':')[2] for key in keys]
        
        return user_ids