"""
Enhanced context-aware message store with improved retention for the Bank of Kigali AI Assistant.
"""

from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Tuple, Any, Optional
from utils.logging_config import logger

class MessageStore:
    """
    In-memory store for user messages with automatic expiration.
    Maintains conversational context across multiple messages with improved retention.
    """
    
    def __init__(self, max_messages: int = 10, expiry_seconds: int = 1800):
        """
        Initialize the message store.
        
        Args:
            max_messages: Maximum number of messages to store per user (default: 10)
            expiry_seconds: Time in seconds after which messages expire (default: 30 min)
        """
        self.messages: Dict[str, List[Tuple[datetime, Any]]] = {}
        self.max_messages = max_messages
        self.expiry_seconds = expiry_seconds
        self.lock = threading.RLock()
        
        # Store conversation summaries for even longer context
        self.conversation_summaries: Dict[str, Tuple[datetime, str]] = {}
        
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
    
    def set_conversation_summary(self, user_id: str, summary: str) -> None:
        """
        Store a summary of the conversation for a user.
        This provides an additional layer of context retention.
        
        Args:
            user_id: Unique identifier for the user
            summary: Summary of the conversation
        """
        with self.lock:
            self.conversation_summaries[user_id] = (datetime.now(), summary)
    
    def get_conversation_summary(self, user_id: str) -> Optional[str]:
        """
        Get the conversation summary for a user if available.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Conversation summary if available, None otherwise
        """
        with self.lock:
            if user_id in self.conversation_summaries:
                # Update timestamp to extend life
                summary_data = self.conversation_summaries[user_id]
                self.conversation_summaries[user_id] = (datetime.now(), summary_data[1])
                return summary_data[1]
            return None
    
    def clear_user_data(self, user_id: str) -> None:
        """
        Clear all data for a user.
        
        Args:
            user_id: Unique identifier for the user
        """
        with self.lock:
            # Remove messages
            if user_id in self.messages:
                del self.messages[user_id]
            
            # Remove summary
            if user_id in self.conversation_summaries:
                del self.conversation_summaries[user_id]
    
    def _cleanup_expired(self) -> None:
        """Background thread that periodically removes expired messages."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._remove_expired()
            except Exception as e:
                logger.error(f"Error in message cleanup thread: {e}")
    
    def _remove_expired(self) -> None:
        """Remove all expired messages and summaries from the store."""
        with self.lock:
            now = datetime.now()
            expiry_time = now - timedelta(seconds=self.expiry_seconds)
            summary_expiry_time = now - timedelta(seconds=self.expiry_seconds * 2)  # Summaries last twice as long
            
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
            
            # Clean up expired summaries
            summaries_to_remove = []
            for user_id, (timestamp, _) in self.conversation_summaries.items():
                if timestamp < summary_expiry_time:
                    summaries_to_remove.append(user_id)
            
            for user_id in summaries_to_remove:
                del self.conversation_summaries[user_id]
            
            if expired_count > 0 or users_to_remove or summaries_to_remove:
                logger.debug(
                    f"Removed {expired_count} expired messages, {len(users_to_remove)} users, "
                    f"and {len(summaries_to_remove)} expired summaries"
                )

# Helper functions for conversation summaries
def create_conversation_summary(messages: List[Any], max_length: int = 200) -> str:
    """
    Create a simple summary of a conversation from recent messages.
    
    Args:
        messages: List of message objects
        max_length: Maximum length of the summary
        
    Returns:
        Conversation summary
    """
    if not messages:
        return ""
    
    # Extract user and assistant messages
    conversation_text = []
    for msg in messages:
        if hasattr(msg, 'role') and hasattr(msg, 'content'):
            prefix = "User: " if msg.role == "user" else "Assistant: "
            conversation_text.append(prefix + msg.content)
    
    # Create summary (simple implementation - truncate to max_length)
    full_text = " | ".join(conversation_text[-6:])  # Last 6 exchanges
    if len(full_text) > max_length:
        return full_text[:max_length-3] + "..."
    return full_text

def extract_user_information(messages: List[Any]) -> Dict[str, str]:
    """
    Extract potential user information from messages including name, 
    account number, and customer ID.
    
    Args:
        messages: List of message objects
        
    Returns:
        Dictionary with user information
    """
    user_info = {}
    name_patterns = [
        "my name is", "i am", "call me", "i'm", "this is", "name's"
    ]
    account_patterns = [
        "account number", "my account", "account #", "account no", "account id"
    ]
    customer_id_patterns = [
        "customer id", "customer number", "client id", "client number", "my id", "my customer"
    ]
    
    for msg in messages:
        if hasattr(msg, 'role') and hasattr(msg, 'content') and msg.role == "user":
            content = msg.content.lower()
            
            # Look for name
            if "name" not in user_info:
                for pattern in name_patterns:
                    if pattern in content:
                        # Extract potential name after pattern
                        name_start = content.find(pattern) + len(pattern)
                        
                        # Find name ending - look for common sentence endings
                        name_end = len(content)  # default to end of content
                        
                        # Common endings to look for
                        potential_endings = [
                            ".", ",", "and ", " and ", "but ", " but ", "so ", " so ", 
                            "?", "!", ";", ":", " the ", " i ", " my ", " or ", "\n",
                            " he ", " she ", " his ", " her ", " will ", " would ",
                            " how ", " what ", " when ", " where ", " why "
                        ]
                        
                        # Find the earliest ending
                        for ending in potential_endings:
                            ending_pos = content.find(ending, name_start)
                            if ending_pos != -1 and ending_pos < name_end:
                                name_end = ending_pos
                        
                        # Get the name part
                        potential_name = content[name_start:name_end].strip()
                        
                        # Remove any trailing punctuation
                        while potential_name and potential_name[-1] in ",.!?;:":
                            potential_name = potential_name[:-1].strip()
                        
                        # Validate the extracted name
                        if (potential_name and 
                            len(potential_name.split()) <= 5 and  # Not too many words (increased from 4 to 5)
                            len(potential_name) < 60 and  # Not too long
                            potential_name != pattern and  # Not just the pattern itself
                            any(char.isalpha() for char in potential_name)):  # Has at least one letter
                            
                            # Clean up and title case
                            name_words = potential_name.split()
                            clean_name = ' '.join(word.capitalize() for word in name_words if len(word) > 0)
                            user_info["name"] = clean_name
                            break
            
            # Look for account numbers
            if "account_number" not in user_info:
                for pattern in account_patterns:
                    if pattern in content:
                        # Look for digits after pattern
                        pattern_index = content.find(pattern) + len(pattern)
                        # Find first digit
                        digit_start = -1
                        for i in range(pattern_index, len(content)):
                            if content[i].isdigit():
                                digit_start = i
                                break
                        
                        if digit_start != -1:
                            # Find where digits end
                            digit_end = digit_start
                            while digit_end < len(content) and (content[digit_end].isdigit() or content[digit_end] == '-'):
                                digit_end += 1
                            
                            account_number = content[digit_start:digit_end].strip()
                            if account_number and len(account_number) >= 4:  # Simple validation
                                user_info["account_number"] = account_number
                                break
            
            # Look for customer ID
            if "customer_id" not in user_info:
                for pattern in customer_id_patterns:
                    if pattern in content:
                        # Look for alphanumeric ID after pattern
                        pattern_index = content.find(pattern) + len(pattern)
                        # Find first alphanumeric
                        id_start = -1
                        for i in range(pattern_index, len(content)):
                            if content[i].isalnum():
                                id_start = i
                                break
                        
                        if id_start != -1:
                            # Find where ID ends
                            id_end = id_start
                            while id_end < len(content) and (content[id_end].isalnum() or content[id_end] == '-'):
                                id_end += 1
                            
                            customer_id = content[id_start:id_end].strip()
                            if customer_id and len(customer_id) >= 4:  # Simple validation
                                user_info["customer_id"] = customer_id
                                break
    
    return user_info