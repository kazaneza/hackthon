"""
Enhanced Document-Based AI Service with improved personalization and context handling.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Import existing components
from config.settings import settings
from utils.logging_config import logger
from core.prompts import get_system_prompt
from api.models import ChatMessage


class ProductQAService:
    """Enhanced Question answering service for Bank of Kigali product information."""
    
    def __init__(self, document_processor):
        """Initialize the QA service."""
        self.document_processor = document_processor
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.OPENAI_MODEL,
            temperature=0.7,
            max_tokens=settings.OPENAI_MAX_TOKENS
        )
        
        logger.info("Product QA service initialized")
    
    def create_qa_chain(self, user_info=None, conversation_context=None) -> RetrievalQA:
        """Create a QA chain for product queries with enhanced context."""
        retriever = self.document_processor.get_retriever()
        
        system_template = """You are ALICE, Bank of Kigali's AI assistant specializing in product information.

Use the following pieces of context to answer the customer's question about Bank of Kigali products and services.
Always be helpful, professional, and personalized when possible.
IMPORTANT: When you have customer information (name, account, etc.) from the conversation context, use it naturally in your responses. Do not claim you don't have access to information that is clearly provided in the context."""

        # Add conversation context if available
        if conversation_context:
            system_template += f"\n\nCONVERSATION CONTEXT:\n{conversation_context}\n"
        
        # Add personalization
        if user_info:
            if "name" in user_info:
                system_template += f"\nThe customer's name is {user_info['name']}. Address them by name naturally throughout your response.\n"
            
            if "account_number" in user_info:
                system_template += f"Account number: {user_info['account_number']}\n"
            
            if "customer_id" in user_info:
                system_template += f"Customer ID: {user_info['customer_id']}\n"
        
        system_template += """
Important guidelines:
1. Reference the conversation context to maintain continuity
2. Use the customer's name naturally when responding
3. Connect your answer to their previous questions when relevant
4. Be specific about product features and requirements
5. Suggest related products when appropriate
6. Always be warm and professional

If you don't have specific information, be honest but offer to help find alternatives or connect them with a specialist.

{context}
"""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
        return qa_chain
    
    async def answer_product_question(self, question: str, user_info=None, conversation_context=None) -> Dict[str, Any]:
        """Answer a product-related question with enhanced personalization."""
        qa_chain = self.create_qa_chain(user_info, conversation_context)
        
        try:
            result = qa_chain({"query": question})
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            # Ensure personalization is applied
            if user_info and "name" in user_info and user_info["name"]:
                name = user_info["name"]
                # If name isn't used in the response, add it appropriately
                if name.lower() not in answer.lower():
                    # Add greeting with name if response doesn't start with one
                    if not any(answer.startswith(greeting) for greeting in ["Hello", "Hi", "Good", "Welcome"]):
                        answer = f"Hello {name}, " + answer
                    else:
                        # Insert name into existing greeting
                        for greeting in ["Hello", "Hi"]:
                            if answer.startswith(greeting):
                                answer = answer.replace(greeting, f"{greeting} {name}", 1)
                                break
            
            # Format source information
            sources = []
            for doc in source_docs:
                sources.append({
                    "category": doc.metadata.get("category", "unknown"),
                    "product": doc.metadata.get("product_type", "unknown"),
                    "file": doc.metadata.get("filename", "unknown"),
                    "page": doc.metadata.get("page", 0) + 1
                })
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error answering product question: {e}")
            default_response = "I'm having trouble finding that information right now. Let me know what specific products you're interested in, and I'll do my best to help."
            
            if user_info and "name" in user_info and user_info["name"]:
                default_response = f"I'm sorry, {user_info['name']}. {default_response}"
            
            return {
                "answer": default_response,
                "sources": []
            }


class DocumentBasedAIService:
    """Enhanced AI service that integrates document-based QA with better context handling."""
    
    def __init__(self, api_key: str = settings.OPENAI_API_KEY):
        """Initialize the document-based AI service."""
        # Initialize document processor
        self.document_processor = DocumentProcessor()
        self.document_processor.setup()
        
        # Initialize product QA service
        self.product_qa = ProductQAService(self.document_processor)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS
        )
        
        logger.info("Document-based AI service initialized")
    
    def extract_user_context(self, messages: List[Any]) -> Dict[str, Any]:
        """Extract both user information and conversation context from messages."""
        user_info = {}
        conversation_context = ""
        recent_messages = []
        
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                if msg.role == "system":
                    # Extract user info from system messages
                    content = msg.content
                    if "KNOWN USER INFORMATION:" in content:
                        # Extract name
                        if "- Name: " in content:
                            name_start = content.find("- Name: ") + len("- Name: ")
                            name_end = content.find("\n", name_start)
                            if name_end != -1:
                                user_info["name"] = content[name_start:name_end].strip()
                        
                        # Extract account number
                        if "- Account Number: " in content:
                            acc_start = content.find("- Account Number: ") + len("- Account Number: ")
                            acc_end = content.find("\n", acc_start)
                            if acc_end != -1:
                                user_info["account_number"] = content[acc_start:acc_end].strip()
                        
                        # Extract customer ID
                        if "- Customer ID: " in content:
                            id_start = content.find("- Customer ID: ") + len("- Customer ID: ")
                            id_end = content.find("\n", id_start)
                            if id_end != -1:
                                user_info["customer_id"] = content[id_start:id_end].strip()
                    
                    # Extract conversation summary/context
                    if "CONVERSATION SUMMARY:" in content:
                        summary_start = content.find("CONVERSATION SUMMARY:") + len("CONVERSATION SUMMARY:")
                        summary_end = content.find("\n", summary_start)
                        if summary_end != -1:
                            conversation_context = content[summary_start:summary_end].strip()
                    elif "RECENT CONVERSATION:" in content:
                        context_start = content.find("RECENT CONVERSATION:")
                        context_end = content.find("\nPERSONALIZATION RULES:", context_start)
                        if context_end != -1:
                            conversation_context = content[context_start:context_end].strip()
                
                # Collect recent user/assistant messages
                if msg.role in ["user", "assistant"]:
                    recent_messages.append(msg)
        
        # Create a brief conversation context if not extracted from system message
        if not conversation_context and recent_messages:
            context_lines = []
            for msg in recent_messages[-4:]:  # Last 4 messages
                prefix = "User" if msg.role == "user" else "Assistant"
                content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
                context_lines.append(f"{prefix}: {content}")
            conversation_context = "\n".join(context_lines)
        
        return {
            "user_info": user_info,
            "conversation_context": conversation_context
        }
    
    def is_product_question(self, message: str) -> bool:
        """Determine if a message is asking about products."""
        product_keywords = [
            "product", "service", "account", "loan", "credit", "card", "mortgage",
            "interest", "rate", "fee", "charge", "term", "deposit", "savings",
            "checking", "investment", "insurance", "sme", "semi", "corporate", "retail",
            "institutional", "agribusiness", "agri", "business", "banking", "offer",
            "application", "apply", "eligibility", "requirement", "qualify", "benefit",
            "feature", "package", "plan", "program", "promotion", "special", "discount",
            "financing", "fund", "money", "payment", "transaction", "transfer", "borrow"
        ]
        
        question_indicators = [
            "how", "what", "where", "when", "who", "which", "why", "can", "do", "does",
            "tell me about", "explain", "describe", "information on", "details about",
            "i want", "i need", "show me", "find", "looking for"
        ]
        
        message_lower = message.lower()
        
        # Check for direct product mentions
        has_product_keyword = any(keyword in message_lower for keyword in product_keywords)
        
        # Check for questions about products
        has_question_about_product = any(
            indicator in message_lower and any(
                keyword in message_lower[message_lower.find(indicator):] 
                for keyword in product_keywords
            )
            for indicator in question_indicators
        )
        
        return has_product_keyword or has_question_about_product
    
    async def generate_response(self, service_category: str, messages: List[Any]) -> Dict[str, Any]:
        """Generate a response using document-based QA or LangChain with full context."""
        try:
            # Extract user context
            context_data = self.extract_user_context(messages)
            user_info = context_data["user_info"]
            conversation_context = context_data["conversation_context"]
            
            # Get the last user message
            last_user_message = None
            for m in reversed(messages):
                if hasattr(m, 'role') and m.role == "user":
                    last_user_message = m.content
                    break
            
            if not last_user_message:
                last_user_message = "Hello"
            
            # Check if it's a product question
            if self.is_product_question(last_user_message):
                # Use product QA with full context
                result = await self.product_qa.answer_product_question(
                    last_user_message, 
                    user_info=user_info,
                    conversation_context=conversation_context
                )
                
                return {
                    "response": result["answer"],
                    "sources": result["sources"]
                }
            else:
                # Use existing LangChain service with enhanced context
                from core.ai_service import LangChainService
                langchain_service = LangChainService()
                
                # Create an enhanced message list with context
                enhanced_messages = []
                
                # Add a system message with context
                if user_info or conversation_context:
                    context_parts = []
                    
                    if user_info and "name" in user_info:
                        context_parts.append(f"Customer name: {user_info['name']}")
                    
                    if conversation_context:
                        context_parts.append(f"Recent conversation: {conversation_context}")
                    
                    enhanced_system_msg = ChatMessage(
                        role="system",
                        content=f"Context: {' | '.join(context_parts)}\nBe personalized and reference context appropriately."
                    )
                    enhanced_messages.append(enhanced_system_msg)
                
                # Add original messages
                enhanced_messages.extend(messages)
                
                ai_response = await langchain_service.generate_response(
                    service_category=service_category,
                    messages=enhanced_messages
                )
                
                # Ensure personalization in non-product responses
                if user_info and "name" in user_info and user_info["name"]:
                    name = user_info["name"]
                    if name.lower() not in ai_response.lower():
                        # Add name appropriately
                        if not any(ai_response.startswith(greeting) for greeting in ["Hello", "Hi", "Good"]):
                            ai_response = f"Hi {name}, " + ai_response
                        else:
                            for greeting in ["Hello", "Hi"]:
                                if ai_response.startswith(greeting):
                                    ai_response = ai_response.replace(greeting, f"{greeting} {name}", 1)
                                    break
                
                return {
                    "response": ai_response,
                    "sources": []
                }
                
        except Exception as e:
            logger.error(f"Error generating response with document-based AI service: {e}")
            raise


# Import the DocumentProcessor class from your existing code
class DocumentProcessor:
    """Processes PDF documents, creates embeddings, and builds a searchable vector store."""
    
    def __init__(self, documents_base_path: str = "data/products"):
        """Initialize the document processor."""
        self.documents_base_path = documents_base_path
        self.categories = ["sme", "retail", "corporate", "institutional", "agribusiness"]
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        self.vector_store = None
        
        logger.info(f"Document processor initialized with base path: {documents_base_path}")
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load all PDF documents from the product categories."""
        all_documents = []
        
        for category in self.categories:
            category_path = os.path.join(self.documents_base_path, category)
            
            if not os.path.exists(category_path):
                logger.warning(f"Category path does not exist: {category_path}")
                continue
            
            logger.info(f"Loading documents from {category_path}")
            
            loader = DirectoryLoader(
                category_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            
            try:
                category_docs = loader.load()
                
                for doc in category_docs:
                    filename = os.path.basename(doc.metadata.get("source", "unknown"))
                    doc.metadata.update({
                        "category": category,
                        "filename": filename,
                        "product_type": self._extract_product_type(filename)
                    })
                
                all_documents.extend(category_docs)
                logger.info(f"Loaded {len(category_docs)} documents from {category}")
                
            except Exception as e:
                logger.error(f"Error loading documents from {category}: {e}")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def _extract_product_type(self, filename: str) -> str:
        """Extract product type from filename."""
        name = os.path.splitext(filename)[0]
        return name.replace("_", " ").title()
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process documents by splitting into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        document_chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(document_chunks)} document chunks from {len(documents)} documents")
        return document_chunks
    
    def create_vector_store(self, document_chunks: List[Dict[str, Any]]) -> None:
        """Create a vector store from document chunks."""
        self.vector_store = Chroma.from_documents(
            documents=document_chunks,
            embedding=self.embeddings,
            persist_directory=os.path.join(self.documents_base_path, "chroma_db")
        )
        
        try:
            self.vector_store.persist()
        except:
            logger.info("Vector store auto-persisted")
        
        logger.info(f"Created vector store with {len(document_chunks)} document chunks")
    
    def load_vector_store(self) -> bool:
        """Load an existing vector store if available."""
        persist_directory = os.path.join(self.documents_base_path, "chroma_db")
        
        if os.path.exists(persist_directory):
            try:
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vector store from {persist_directory}")
                return True
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
        
        logger.warning("No existing vector store found")
        return False
    
    def setup(self) -> None:
        """Set up the document processor by loading or creating a vector store."""
        if not self.load_vector_store():
            documents = self.load_documents()
            document_chunks = self.process_documents(documents)
            self.create_vector_store(document_chunks)
    
    def get_retriever(self):
        """Get a retriever for the vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call setup() first.")
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 7}
        )