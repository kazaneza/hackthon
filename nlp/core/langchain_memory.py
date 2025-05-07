"""
LangChain memory integration for the Bank of Kigali AI Assistant.
This provides an alternative to the custom MessageStore using LangChain's built-in memory.
"""

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from typing import Dict, Optional
from datetime import datetime
import json
from uuid import uuid4
from config.settings import settings
from utils.logging_config import logger

class LangChainMemoryManager:
    """
    Manages conversation memory using LangChain's memory components.
    Can be used as an alternative to the custom MessageStore.
    """
    
    def __init__(self, expiry_seconds: int = settings.MESSAGE_EXPIRY_SECONDS):
        """
        Initialize the LangChain memory manager.
        
        Args:
            expiry_seconds: Time in seconds after which memories expire
        """
        self.memories: Dict[str, ConversationBufferMemory] = {}
        self.expiry_seconds = expiry_seconds
        logger.info(f"LangChain memory manager initialized with {expiry_seconds}s expiry")
    
    def get_memory(self, conversation_id: str) -> ConversationBufferMemory:
        """
        Get or create a memory object for a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            ConversationBufferMemory for the conversation
        """
        if conversation_id not in self.memories:
            # Create a new memory instance
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            self.memories[conversation_id] = memory
        
        return self.memories[conversation_id]
    
    def add_user_message(self, conversation_id: str, message: str) -> None:
        """
        Add a user message to memory.
        
        Args:
            conversation_id: Unique identifier for the conversation
            message: User message content
        """
        memory = self.get_memory(conversation_id)
        memory.chat_memory.add_user_message(message)
        logger.debug(f"Added user message to memory for conversation {conversation_id}")
    
    def add_ai_message(self, conversation_id: str, message: str) -> None:
        """
        Add an AI message to memory.
        
        Args:
            conversation_id: Unique identifier for the conversation
            message: AI message content
        """
        memory = self.get_memory(conversation_id)
        memory.chat_memory.add_ai_message(message)
        logger.debug(f"Added AI message to memory for conversation {conversation_id}")
    
    def get_conversation_history(self, conversation_id: str) -> Dict:
        """
        Get the conversation history for Chain/Agent usage.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            Dictionary with the conversation history in the memory_key
        """
        memory = self.get_memory(conversation_id)
        return memory.load_memory_variables({})
    
    def clear_memory(self, conversation_id: str) -> None:
        """
        Clear the memory for a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
        """
        if conversation_id in self.memories:
            self.memories[conversation_id].clear()
            logger.debug(f"Cleared memory for conversation {conversation_id}")
    
    def remove_conversation(self, conversation_id: str) -> None:
        """
        Remove a conversation from the memory manager.
        
        Args:
            conversation_id: Unique identifier for the conversation
        """
        if conversation_id in self.memories:
            del self.memories[conversation_id]
            logger.debug(f"Removed conversation {conversation_id} from memory manager")