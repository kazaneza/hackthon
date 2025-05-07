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
            chunk_size=1500,        # Increased for better context
            chunk_overlap=300,      # Increased for better overlap
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
        # Note: This is no longer needed with newer versions of Chroma but kept for compatibility
        try:
            self.vector_store.persist()
        except:
            logger.info("Vector store auto-persisted")
        
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
            search_kwargs={"k": 7}  # Increased from 5 for better context
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
            temperature=0.7,  # Default temperature for balanced creativity
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
Even if the information is partial or doesn't exactly match the question, try to provide the most relevant details from the context.
If the customer asks about a specific product that might be a typo or slight variation (like SEMI vs SME), consider providing information about the closest matching product.

Be conversational and helpful, focusing on what would be most useful to the customer. Avoid mentioning the specific documents you're using in your answer.

Here are some guidelines for your responses:
1. Start with a direct answer to the question if possible
2. Provide relevant details about the product features, benefits, or requirements
3. If appropriate, suggest related products or services that might be of interest
4. End with a helpful suggestion or offer to provide more information on specific aspects

If you genuinely don't have any relevant information, politely say something like "I don't have specific information about that product. Would you like me to tell you about our other similar offerings instead?"

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
            
            # Format source information (for internal use/logging only)
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
                "answer": "I'm sorry, I couldn't find specific information about that. Would you like to know about our other banking products and services instead?",
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
            "checking", "investment", "insurance", "sme", "semi", "corporate", "retail",
            "institutional", "agribusiness", "agri", "business", "banking", "offer",
            "application", "apply", "eligibility", "requirement", "qualify", "benefit",
            "feature", "package", "plan", "program", "promotion", "special", "discount",
            "financing", "fund", "money", "payment", "transaction", "transfer", "borrow"
        ]
        
        # Question indicators
        question_indicators = [
            "how", "what", "where", "when", "who", "which", "why", "can", "do", "does",
            "tell me about", "explain", "describe", "information on", "details about"
        ]
        
        # Check if message contains product keywords or question indicators about products
        message_lower = message.lower()
        
        # Check for product keywords
        has_product_keyword = any(keyword in message_lower for keyword in product_keywords)
        
        # Check for question indicators followed by product keywords
        has_question_about_product = any(
            indicator in message_lower and any(
                keyword in message_lower[message_lower.find(indicator):] 
                for keyword in product_keywords
            )
            for indicator in question_indicators
        )
        
        return has_product_keyword or has_question_about_product
    
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


# For standalone testing
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