"""
Bank of Kigali Document-Based Product QA System
---
A document retrieval and question answering system for product PDFs:
- Processes PDF files from different banking categories (SME, Retail, Corporate, etc.)
- Creates embeddings for semantic search
- Retrieves relevant document sections for customer queries
- Integrates with the existing LangChain-based AI assistant
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Import existing components
from config.settings import settings
from utils.logging_config import logger
from core.prompts import get_system_prompt


class DocumentProcessor:
    """
    Processes PDF documents, creates embeddings, and builds a searchable vector store.
    """
    
    def __init__(self, documents_base_path: str = "data/products"):
        """
        Initialize the document processor.
        
        Args:
            documents_base_path: Base directory containing product PDF documents
        """
        self.documents_base_path = documents_base_path
        self.categories = ["sme", "retail", "corporate", "institutional", "agribusiness"]
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        self.vector_store = None
        
        logger.info(f"Document processor initialized with base path: {documents_base_path}")
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load all PDF documents from the product categories.
        
        Returns:
            List of document objects with text and metadata
        """
        all_documents = []
        
        for category in self.categories:
            category_path = os.path.join(self.documents_base_path, category)
            
            if not os.path.exists(category_path):
                logger.warning(f"Category path does not exist: {category_path}")
                continue
            
            logger.info(f"Loading documents from {category_path}")
            
            # Use DirectoryLoader to load all PDFs in the category directory
            loader = DirectoryLoader(
                category_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            
            try:
                category_docs = loader.load()
                
                # Add category metadata to each document
                for doc in category_docs:
                    # Extract filename from source
                    filename = os.path.basename(doc.metadata.get("source", "unknown"))
                    
                    # Update metadata
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
        """
        Extract product type from filename.
        Simple extraction by removing extension and replacing underscores.
        
        Args:
            filename: Document filename
            
        Returns:
            Product type
        """
        # Remove extension and replace underscores with spaces
        name = os.path.splitext(filename)[0]
        return name.replace("_", " ").title()
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process documents by splitting into chunks.
        
        Args:
            documents: List of document objects
            
        Returns:
            List of document chunks
        """
        # Create text splitter for chunking documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split documents into chunks
        document_chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(document_chunks)} document chunks from {len(documents)} documents")
        return document_chunks
    
    def create_vector_store(self, document_chunks: List[Dict[str, Any]]) -> None:
        """
        Create a vector store from document chunks.
        
        Args:
            document_chunks: List of document chunks
        """
        # Create Chroma vector store with OpenAI embeddings
        self.vector_store = Chroma.from_documents(
            documents=document_chunks,
            embedding=self.embeddings,
            persist_directory=os.path.join(self.documents_base_path, "chroma_db")
        )
        
        # Persist the vector store
        self.vector_store.persist()
        
        logger.info(f"Created vector store with {len(document_chunks)} document chunks")
    
    def load_vector_store(self) -> bool:
        """
        Load an existing vector store if available.
        
        Returns:
            True if loaded successfully, False otherwise
        """
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
        """
        Set up the document processor by loading or creating a vector store.
        """
        # Try to load existing vector store
        if not self.load_vector_store():
            # Load and process documents
            documents = self.load_documents()
            document_chunks = self.process_documents(documents)
            
            # Create vector store
            self.create_vector_store(document_chunks)
    
    def get_retriever(self):
        """
        Get a retriever for the vector store.
        
        Returns:
            Retriever for querying the vector store
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call setup() first.")
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )


class ProductQAService:
    """
    Question answering service for Bank of Kigali product information.
    """
    
    def __init__(self, document_processor: DocumentProcessor):
        """
        Initialize the QA service.
        
        Args:
            document_processor: Document processor with vector store
        """
        self.document_processor = document_processor
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS
        )
        
        logger.info("Product QA service initialized")
    
    def create_qa_chain(self) -> RetrievalQA:
        """
        Create a QA chain for product queries.
        
        Returns:
            RetrievalQA chain
        """
        # Get retriever from document processor
        retriever = self.document_processor.get_retriever()
        
        # Create system template with context for product information
        system_template = """You are the Bank of Kigali's AI assistant specializing in product information.
        
Use the following pieces of context to answer the customer's question about Bank of Kigali products and services.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always include the product category and name in your response.

{context}
"""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ])
        
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return qa_chain
    
    async def answer_product_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a product-related question using the document-based QA system.
        
        Args:
            question: Customer's product question
            
        Returns:
            Dictionary with answer and source documents
        """
        # Create QA chain
        qa_chain = self.create_qa_chain()
        
        try:
            # Run the chain
            result = qa_chain({"query": question})
            
            # Extract answer and source documents
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            # Format source information
            sources = []
            for i, doc in enumerate(source_docs):
                sources.append({
                    "category": doc.metadata.get("category", "unknown"),
                    "product": doc.metadata.get("product_type", "unknown"),
                    "file": doc.metadata.get("filename", "unknown"),
                    "page": doc.metadata.get("page", 0) + 1  # Adjust 0-index to 1-index for user readability
                })
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error answering product question: {e}")
            return {
                "answer": "I encountered an error while trying to find information about this product. Please try again or contact a bank representative.",
                "sources": []
            }


class DocumentBasedAIService:
    """
    Enhanced AI service that integrates document-based QA with the existing LangChain service.
    """
    
    def __init__(self, api_key: str = settings.OPENAI_API_KEY):
        """
        Initialize the document-based AI service.
        
        Args:
            api_key: OpenAI API key
        """
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
    
    def is_product_question(self, message: str) -> bool:
        """
        Determine if a message is asking about products.
        
        Args:
            message: User message
            
        Returns:
            True if the message is asking about products, False otherwise
        """
        # Keywords that might indicate a product question
        product_keywords = [
            "product", "service", "account", "loan", "credit", "card", "mortgage",
            "interest", "rate", "fee", "charge", "term", "deposit", "savings",
            "checking", "investment", "insurance", "sme", "corporate", "retail",
            "institutional", "agribusiness", "agri", "business", "banking"
        ]
        
        # Check if message contains product keywords
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in product_keywords)
    
    async def generate_response(self, service_category: str, messages: List[Any]) -> Dict[str, Any]:
        """
        Generate a response using document-based QA or LangChain.
        
        Args:
            service_category: Service category for the prompt
            messages: List of message objects
            
        Returns:
            Response with answer and optional sources
        """
        try:
            # Extract the last user message
            last_user_message = None
            for m in reversed(messages):
                if hasattr(m, 'role') and m.role == "user":
                    last_user_message = m.content
                    break
            
            if not last_user_message:
                last_user_message = "Hello"
            
            # Check if it's a product question
            if self.is_product_question(last_user_message):
                # Use product QA
                result = await self.product_qa.answer_product_question(last_user_message)
                
                return {
                    "response": result["answer"],
                    "sources": result["sources"]
                }
            else:
                # Use existing LangChain service (imported from your current code)
                from core.ai_service import LangChainService
                langchain_service = LangChainService()
                
                ai_response = await langchain_service.generate_response(
                    service_category=service_category,
                    messages=messages
                )
                
                return {
                    "response": ai_response,
                    "sources": []  # No document sources for non-product questions
                }
                
        except Exception as e:
            logger.error(f"Error generating response with document-based AI service: {e}")
            raise


# Modified route implementation for PDF-based product QA
def update_chat_route():
    """
    This function shows how to modify the existing chat route to include document-based QA.
    Copy this into api/routes.py and adapt as needed.
    """
    
    # From api.routes.py (modified version)
    @router.post("/chat", response_model=ChatResponse)
    async def chat(
        request: ChatRequest,
        document_ai_service: DocumentBasedAIService = Depends(get_document_ai_service)  # New dependency
    ):
        try:
            # Use provided user_id or generate an anonymous one
            user_id = request.user_id or f"anonymous-{uuid4()}"
            
            # Get previous messages for context (same as original)
            previous_messages = message_store.get_messages(user_id)
            
            # Add new user messages to store (same as original)
            for msg in request.messages:
                if msg.role == "user":  # Only store user messages
                    message_store.add_message(user_id, msg)
            
            # Combine previous messages with current request (same as original)
            current_message_contents = [m.content for m in request.messages]
            context_messages = []
            
            for prev_msg in previous_messages:
                if hasattr(prev_msg, 'role') and prev_msg.role in ['system', 'assistant', 'user', 'function', 'tool', 'developer']:
                    if prev_msg.content not in current_message_contents:
                        context_messages.append(prev_msg)
            
            # Create combined messages list with context
            all_messages = context_messages + request.messages
            
            # Generate response using document-based AI service
            result = await document_ai_service.generate_response(
                service_category=request.service_category,
                messages=all_messages
            )
            
            # Extract response and sources
            ai_response = result["response"]
            sources = result["sources"]
            
            # Add document sources to response if available
            if sources:
                # Format sources information
                sources_text = "\n\nThis information comes from the following documents:\n"
                for i, source in enumerate(sources):
                    sources_text += f"{i+1}. {source['product']} ({source['category'].upper()}) - {source['file']} (Page {source['page']})\n"
                
                # Append sources to response
                ai_response += sources_text
            
            # Store AI response in message store (same as original)
            from api.models import ChatMessage
            message_store.add_message(
                user_id, 
                ChatMessage(role="assistant", content=ai_response)
            )
            
            # Rest is the same as original
            conversation_id = str(uuid4())
            suggestions = get_follow_up_suggestions(request.service_category)
            
            return ChatResponse(
                response=ai_response,
                conversation_id=conversation_id,
                service_category=request.service_category,
                timestamp=datetime.now().isoformat(),
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error in chat endpoint: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# New dependency for document-based AI service
async def get_document_ai_service():
    """Dependency to get document-based AI service."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    return DocumentBasedAIService(api_key=settings.OPENAI_API_KEY)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_product_qa():
        # Initialize document processor
        doc_processor = DocumentProcessor()
        doc_processor.setup()
        
        # Initialize product QA service
        product_qa = ProductQAService(doc_processor)
        
        # Test product question
        question = "What are the requirements for an SME loan?"
        result = await product_qa.answer_product_question(question)
        
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print("Sources:")
        for source in result["sources"]:
            print(f"  - {source['product']} ({source['category']}) - {source['file']} (Page {source['page']})")
    
    # Run test
    asyncio.run(test_product_qa())