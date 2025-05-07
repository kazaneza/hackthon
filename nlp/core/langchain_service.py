"""
Advanced LangChain implementation for the Bank of Kigali AI Assistant.
This version uses chains, tools, and agents for more powerful capabilities.
"""

from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.manager import CallbackManager
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from config.settings import settings
from core.prompts import get_system_prompt
from utils.logging_config import logger

class AdvancedLangChainService:
    """Advanced service using LangChain chains, tools, and agents."""
    
    def __init__(self, api_key: str = settings.OPENAI_API_KEY):
        """
        Initialize the advanced LangChain service.
        
        Args:
            api_key: OpenAI API key (or other provider key)
        """
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS
        )
        
        # Initialize banking tools
        self.tools = self._create_banking_tools()
        
        logger.info(f"Advanced LangChain service initialized with model {settings.OPENAI_MODEL}")
    
    def _create_banking_tools(self) -> List[Tool]:
        """
        Create banking-specific tools for LangChain.
        
        Returns:
            List of LangChain tools
        """
        # Example banking tools - these would be connected to real bank APIs in production
        account_info_tool = Tool(
            name="account_information",
            description="Get information about bank accounts like balance, transactions, etc.",
            func=lambda query: "This is a simulation. In production, this would return real account data."
        )
        
        branch_locator_tool = Tool(
            name="branch_locator",
            description="Find nearby Bank of Kigali branches and ATMs",
            func=lambda query: "This is a simulation. In production, this would return real branch locations."
        )
        
        rates_tool = Tool(
            name="exchange_rates",
            description="Get current exchange rates and interest rates",
            func=lambda query: "This is a simulation. In production, this would return real exchange rates."
        )
        
        product_info_tool = Tool(
            name="product_information",
            description="Get detailed information about Bank of Kigali products and services",
            func=lambda query: "This is a simulation. In production, this would return real product information."
        )
        
        return [account_info_tool, branch_locator_tool, rates_tool, product_info_tool]
    
    def _convert_to_langchain_messages(self, messages: List[Any]) -> List[BaseMessage]:
        """
        Convert API message format to LangChain message format.
        
        Args:
            messages: List of message objects from the API
            
        Returns:
            List of LangChain message objects
        """
        langchain_messages = []
        
        for m in messages:
            if hasattr(m, 'role') and hasattr(m, 'content'):
                if m.role == "system":
                    langchain_messages.append(SystemMessage(content=m.content))
                elif m.role == "user":
                    langchain_messages.append(HumanMessage(content=m.content))
                elif m.role == "assistant":
                    langchain_messages.append(AIMessage(content=m.content))
                # Skip other roles for simplicity
        
        return langchain_messages
    
    def create_conversation_chain(self, service_category: str, conversation_id: str) -> ConversationChain:
        """
        Create a conversation chain for the specified service category.
        
        Args:
            service_category: Service category for the system prompt
            conversation_id: Unique identifier for the conversation
            
        Returns:
            LangChain ConversationChain
        """
        system_prompt = get_system_prompt(service_category)
        
        # Create prompt template with system message and history
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        # Create memory
        memory = ConversationBufferMemory(return_messages=True, memory_key="history")
        
        # Create the chain
        chain = ConversationChain(
            llm=self.llm, 
            prompt=prompt,
            memory=memory,
            verbose=True
        )
        
        return chain
    
    def create_agent(self, service_category: str) -> AgentExecutor:
        """
        Create a LangChain agent with tools for the specified service category.
        
        Args:
            service_category: Service category for the system prompt
            
        Returns:
            LangChain AgentExecutor
        """
        system_prompt = get_system_prompt(service_category)
        
        # Create prompt template for the agent
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                f"{system_prompt}\n\nYou have access to the following tools to help customers:\n"
                f"{{tools}}\n\nUse these tools when appropriate to provide accurate information."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    async def generate_response(self, service_category: str, messages: List[Any]) -> str:
        """
        Generate a response using LangChain components.
        
        Args:
            service_category: Service category for the prompt
            messages: List of message objects
            
        Returns:
            Response from LangChain
        """
        try:
            # For executive services and personalized banking, use the agent with tools
            if service_category in ["executive_services", "personalized_banking"]:
                # Extract the last user message
                last_user_message = None
                for m in reversed(messages):
                    if hasattr(m, 'role') and m.role == "user":
                        last_user_message = m.content
                        break
                
                if not last_user_message:
                    last_user_message = "Hello"
                
                # Create and run the agent
                agent = self.create_agent(service_category)
                agent_response = agent.invoke({
                    "input": last_user_message,
                    "chat_history": self._convert_to_langchain_messages(messages[:-1])
                })
                
                return agent_response["output"]
            
            # For other service categories, use a simple conversation chain
            else:
                # Create a unique conversation ID (in a real app, this would be persistent)
                conversation_id = "temp-" + service_category
                
                # Create the conversation chain
                chain = self.create_conversation_chain(service_category, conversation_id)
                
                # Extract the last user message
                last_user_message = None
                for m in reversed(messages):
                    if hasattr(m, 'role') and m.role == "user":
                        last_user_message = m.content
                        break
                
                if not last_user_message:
                    last_user_message = "Hello"
                
                # Add previous messages to memory
                for m in messages[:-1]:
                    if hasattr(m, 'role') and hasattr(m, 'content'):
                        if m.role == "user":
                            chain.memory.chat_memory.add_user_message(m.content)
                        elif m.role == "assistant":
                            chain.memory.chat_memory.add_ai_message(m.content)
                
                # Run the chain
                response = chain.predict(input=last_user_message)
                
                return response
            
        except Exception as e:
            logger.error(f"Error generating response with LangChain: {e}")
            raise