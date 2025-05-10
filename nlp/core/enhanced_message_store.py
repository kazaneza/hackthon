"""
Enhanced Message Store with conversation pair history for the Bank of Kigali AI Assistant.
This implements a cleaner approach focusing on Q&A pairs with limited retention.
File: nlp/core/enhanced_message_store.py
"""

from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Tuple, Any, Optional
from utils.logging_config import logger

class ConversationPair:
    """Represents a single question-answer pair in a conversation."""
    
    def __init__(self, question: str, answer: str, timestamp: datetime):
        self.question = question
        self.answer = answer
        self.timestamp = timestamp
        self.user_info = {}  # Store user info extracted from this conversation
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
            "user_info": self.user_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationPair':
        pair = cls(
            question=data["question"],
            answer=data["answer"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        pair.user_info = data.get("user_info", {})
        return pair

class EnhancedMessageStore:
    """
    Improved message store that tracks conversation pairs (Q&A) with limited history.
    This approach is simpler and more efficient than storing individual messages.
    """
    
    def __init__(self, max_pairs: int = 3, expiry_seconds: int = 1800):
        """
        Initialize the enhanced message store.
        
        Args:
            max_pairs: Maximum number of Q&A pairs to store per user (default: 3)
            expiry_seconds: Time in seconds after which conversation expires (default: 30 min)
        """
        self.conversations: Dict[str, List[ConversationPair]] = {}
        self.user_info: Dict[str, Dict[str, str]] = {}  # Persistent user information
        self.max_pairs = max_pairs
        self.expiry_seconds = expiry_seconds
        self.lock = threading.RLock()
        
        # Start background thread for cleanup
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"Enhanced message store initialized with {max_pairs} max conversation pairs and {expiry_seconds}s expiry")
    
    def add_qa_pair(self, user_id: str, question: str, answer: str, detected_user_info: Dict[str, str] = None) -> None:
        """
        Add a question-answer pair to the store.
        
        Args:
            user_id: Unique identifier for the user
            question: The user's question
            answer: The assistant's answer
            detected_user_info: Any user information detected in this conversation
        """
        with self.lock:
            # Create new conversation pair
            pair = ConversationPair(question, answer, datetime.now())
            
            # Store user info if provided
            if detected_user_info:
                pair.user_info = detected_user_info
                # Also update persistent user info
                if user_id not in self.user_info:
                    self.user_info[user_id] = {}
                self.user_info[user_id].update(detected_user_info)
            
            # Add to conversation history
            if user_id not in self.conversations:
                self.conversations[user_id] = []
            
            self.conversations[user_id].append(pair)
            
            # Keep only the last max_pairs conversations
            if len(self.conversations[user_id]) > self.max_pairs:
                self.conversations[user_id] = self.conversations[user_id][-self.max_pairs:]
            
            logger.debug(f"Added Q&A pair for user {user_id}. Total pairs: {len(self.conversations[user_id])}")
    
    def get_recent_conversations(self, user_id: str) -> List[ConversationPair]:
        """
        Get recent conversation pairs for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            List of recent ConversationPair objects
        """
        with self.lock:
            if user_id in self.conversations:
                # Refresh timestamp
                now = datetime.now()
                for pair in self.conversations[user_id]:
                    pair.timestamp = now
                
                return self.conversations[user_id]
            return []
    
    def get_user_info(self, user_id: str) -> Dict[str, str]:
        """
        Get persistent user information.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary with user information
        """
        with self.lock:
            return self.user_info.get(user_id, {})
    
    def clear_user_data(self, user_id: str) -> None:
        """
        Clear all data for a user.
        
        Args:
            user_id: Unique identifier for the user
        """
        with self.lock:
            if user_id in self.conversations:
                del self.conversations[user_id]
            if user_id in self.user_info:
                del self.user_info[user_id]
            logger.info(f"Cleared all data for user {user_id}")
    
    def _cleanup_expired(self) -> None:
        """Background thread that periodically removes expired conversations."""
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes
                self._remove_expired()
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
    
    def _remove_expired(self) -> None:
        """Remove expired conversations from the store."""
        with self.lock:
            now = datetime.now()
            expiry_time = now - timedelta(seconds=self.expiry_seconds)
            
            users_to_remove = []
            expired_count = 0
            
            for user_id, pairs in self.conversations.items():
                # Check if any pairs are still valid
                valid_pairs = [pair for pair in pairs if pair.timestamp >= expiry_time]
                
                if not valid_pairs:
                    users_to_remove.append(user_id)
                    expired_count += len(pairs)
                else:
                    self.conversations[user_id] = valid_pairs
                    expired_count += len(pairs) - len(valid_pairs)
            
            # Remove completely expired users
            for user_id in users_to_remove:
                del self.conversations[user_id]
            
            if expired_count > 0 or users_to_remove:
                logger.debug(f"Removed {expired_count} expired conversation pairs for {len(users_to_remove)} users")

def extract_user_information_from_qa(question: str, answer: str = "") -> Dict[str, str]:
    """
    Extract user information from a Q&A pair.
    
    Args:
        question: The user's question
        answer: The assistant's answer (optional)
        
    Returns:
        Dictionary with detected user information
    """
    user_info = {}
    
    # Process question for user info
    content = question.lower()
    
    # Name patterns
    name_patterns = [
        "my name is", "i am", "call me", "i'm", "this is", "name's"
    ]
    
    for pattern in name_patterns:
        if pattern in content:
            name_start = content.find(pattern) + len(pattern)
            # Find end of name
            name_end = len(content)
            for ending in [".", ",", "and", "but", "?", "!", "the", "\n"]:
                pos = content.find(" " + ending, name_start)
                if pos != -1 and pos < name_end:
                    name_end = pos
            
            potential_name = content[name_start:name_end].strip()
            
            if potential_name and len(potential_name.split()) <= 4:
                # Clean and capitalize
                name_words = potential_name.replace("  ", " ").split()
                clean_name = ' '.join(word.capitalize() for word in name_words if word)
                if clean_name:
                    user_info["name"] = clean_name
                    break
    
    # Account number patterns
    account_patterns = [
        "account number", "my account", "account #", "account no", "account id"
    ]
    
    for pattern in account_patterns:
        if pattern in content:
            # Look for numbers after pattern
            pattern_start = content.find(pattern) + len(pattern)
            # Extract digits and hyphens
            import re
            match = re.search(r'[\d-]+', content[pattern_start:pattern_start+50])
            if match and len(match.group()) >= 4:
                user_info["account_number"] = match.group()
                break
    
    return user_info