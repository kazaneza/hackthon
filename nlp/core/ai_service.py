"""
OpenAI integration service for the Bank of Kigali AI Assistant.
"""

from typing import List, Dict, Any
import openai
from config.settings import settings
from core.prompts import get_system_prompt
from utils.logging_config import logger

class OpenAIService:
    """Service for interacting with OpenAI API."""
    
    def __init__(self, api_key: str = settings.OPENAI_API_KEY):
        """
        Initialize the OpenAI service.
        
        Args:
            api_key: OpenAI API key
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = settings.OPENAI_MODEL
        self.temperature = settings.OPENAI_TEMPERATURE
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        
        logger.info(f"OpenAI service initialized with model {self.model}")
    
    def prepare_messages(self, service_category: str, messages: List[Any]) -> List[Dict[str, str]]:
        """
        Prepare messages for the OpenAI API.
        
        Args:
            service_category: Service category for the prompt
            messages: List of message objects
            
        Returns:
            Formatted messages for OpenAI API
        """
        system_prompt = get_system_prompt(service_category)
        
        formatted_messages = [{"role": "system", "content": system_prompt}]
        
        # Ensure each message has a valid role before adding
        valid_roles = ['system', 'assistant', 'user', 'function', 'tool', 'developer']
        
        for m in messages:
            if hasattr(m, 'role') and hasattr(m, 'content'):
                role = m.role if m.role in valid_roles else "user"  # Default to user if invalid
                formatted_messages.append({"role": role, "content": m.content})
        
        return formatted_messages
    
    async def generate_response(self, service_category: str, messages: List[Any]) -> str:
        """
        Generate a response using the OpenAI API.
        
        Args:
            service_category: Service category for the prompt
            messages: List of message objects
            
        Returns:
            Response from OpenAI
        """
        try:
            formatted_messages = self.prepare_messages(service_category, messages)
            
            logger.debug(f"Sending request to OpenAI with {len(messages)} messages")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            ai_response = response.choices[0].message.content
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            raise