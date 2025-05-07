"""
LangChain integration service for the Bank of Kigali AI Assistant.
"""

from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from config.settings import settings
from core.prompts import get_system_prompt
from utils.logging_config import logger

class LangChainService:
    """Service for interacting with LLMs via LangChain."""
    
    def __init__(self, api_key: str = settings.OPENAI_API_KEY):
        """
        Initialize the LangChain service.
        
        Args:
            api_key: OpenAI API key (or other provider key)
        """
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS
        )
        logger.info(f"LangChain service initialized with model {settings.OPENAI_MODEL}")
    
    def prepare_messages(self, service_category: str, messages: List[Any]) -> List[Dict]:
        """
        Prepare messages for LangChain.
        
        Args:
            service_category: Service category for the prompt
            messages: List of message objects
            
        Returns:
            Formatted message history for LangChain
        """
        system_prompt = get_system_prompt(service_category)
        
        # Start with the system prompt template
        messages_for_template = [
            SystemMessagePromptTemplate.from_template(system_prompt)
        ]
        
        # Extract conversation history
        human_messages = []
        for i, m in enumerate(messages):
            if hasattr(m, 'role') and hasattr(m, 'content'):
                if m.role == "user":
                    human_messages.append(m.content)
        
        # If there are human messages, add them to the template
        if human_messages:
            human_message = " ".join(human_messages)
            messages_for_template.append(
                HumanMessagePromptTemplate.from_template("{input}")
            )
            return messages_for_template, {"input": human_message}
        
        # Default empty input if no human messages found
        messages_for_template.append(
            HumanMessagePromptTemplate.from_template("{input}")
        )
        return messages_for_template, {"input": "Hello"}
    
    async def generate_response(self, service_category: str, messages: List[Any]) -> str:
        """
        Generate a response using LangChain.
        
        Args:
            service_category: Service category for the prompt
            messages: List of message objects
            
        Returns:
            Response from the LLM
        """
        try:
            message_templates, input_values = self.prepare_messages(service_category, messages)
            
            # Create a chat prompt template
            chat_prompt = ChatPromptTemplate.from_messages(message_templates)
            
            # Create an LLM chain
            chain = LLMChain(llm=self.llm, prompt=chat_prompt)
            
            logger.debug(f"Sending request to LangChain with {len(messages)} messages")
            
            # Run the chain
            response = chain.run(**input_values)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response with LangChain: {e}")
            raise