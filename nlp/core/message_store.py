"""
Context-aware message store with automatic expiration for the Bank of Kigali AI Assistant.
"""

from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Tuple, Any
from utils.logging_config import logger

class MessageStore:
    """
    In-memory store for user messages with automatic expiration.
    Maintains conversational context across multiple messages.
    """
    
    def __init__(self, max_messages: int = 5, expiry_seconds: int = 120):
        """
        Initialize the message store.
        
        Args:
            max_messages: Maximum number of messages to store per user
            expiry_seconds: Time in seconds after which messages expire
        """
        self.messages: Dict[str, List[Tuple[datetime, Any]]] = {}
        self.max_messages = max_messages
        self.expiry_seconds = expiry_seconds
        self.lock = threading.RLock()
        
        # Start background thread for cleanup
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"Message store initialized with {max_messages} max messages and {expiry_seconds}s expiry")
    
    def add_message(self, user_id: str, message: Any) -> None:
        """
        Add a message to the store for a user.
        
        Args:
            user_id: Unique identifier for the user
            message: Message object to store
        """
        with self.lock:
            timestamp = datetime.now()
            if user_id not in self.messages:
                self.messages[user_id] = []
            
            # Add new message
            self.messages[user_id].append((timestamp, message))
            
            # Keep only the last max_messages
            if len(self.messages[user_id]) > self.max_messages:
                self.messages[user_id] = self.messages[user_id][-self.max_messages:]
    
    def get_messages(self, user_id: str) -> List[Any]:
        """
        Get all messages for a user and refresh their expiry time.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            List of message objects
        """
        with self.lock:
            # Update timestamp for all messages to extend their life
            if user_id in self.messages:
                now = datetime.now()
                self.messages[user_id] = [(now, msg) for _, msg in self.messages[user_id]]
                return [msg for _, msg in self.messages[user_id]]
            return []
    
    def _cleanup_expired(self) -> None:
        """Background thread that periodically removes expired messages."""
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                self._remove_expired()
            except Exception as e:
                logger.error(f"Error in message cleanup thread: {e}")
    
    def _remove_expired(self) -> None:
        """Remove all expired messages from the store."""
        with self.lock:
            now = datetime.now()
            expiry_time = now - timedelta(seconds=self.expiry_seconds)
            
            # Users to remove (completely)
            users_to_remove = []
            expired_count = 0
            
            for user_id, user_messages in self.messages.items():
                # Filter out expired messages
                valid_messages = [(ts, msg) for ts, msg in user_messages if ts >= expiry_time]
                expired_count += len(user_messages) - len(valid_messages)
                
                if not valid_messages:
                    users_to_remove.append(user_id)
                else:
                    self.messages[user_id] = valid_messages
            
            # Remove empty user entries
            for user_id in users_to_remove:
                del self.messages[user_id]
            
            if expired_count > 0 or users_to_remove:
                logger.debug(f"Removed {expired_count} expired messages and {len(users_to_remove)} users")