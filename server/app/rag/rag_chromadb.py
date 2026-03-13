import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# PDF and text processing
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# CrewAI RAG imports
from crewai.rag.config.utils import set_rag_config, get_rag_client, clear_rag_config
from crewai.rag.chromadb.config import ChromaDBConfig

# Embeddings
from langchain_community.embeddings import OllamaEmbeddings

# CrewAI
from crewai import Agent, Task, Crew
from langchain_core.tools import BaseTool

# Ollama
from langchain_community.llms import Ollama

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF document processing and text extraction"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def process_text(self, text: str, source: str = "unknown") -> List[Document]:
        """Split text into chunks and create Document objects"""
        chunks = self.text_splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_id": i,
                    "chunk_size": len(chunk)
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF file and return Document objects"""
        text = self.extract_text_from_pdf(pdf_path)
        if text:
            return self.process_text(text, source=pdf_path)
        return []


class ConversationMemory:
    """Manages conversation history for different sessions"""
    
    def __init__(self):
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        return self.conversations.get(session_id, [])
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    def get_context_string(self, session_id: str) -> str:
        """Get conversation history as formatted string"""
        history = self.get_conversation_history(session_id)
        context = ""
        for msg in history:
            context += f"{msg['role'].title()}: {msg['content']}\n"
        return context


class DocumentSearchTool(BaseTool):
    """CrewAI tool for document search using ChromaDB"""
    
    name: str = "document_search"
    description: str = "Search through uploaded documents for relevant information based on user queries"
    rag_client: Any = None
    
    def __init__(self, rag_client, **kwargs):
        super().__init__(**kwargs)
        self.rag_client = rag_client
    
    def _run(self, query: str) -> str:
        """Execute document search"""
        try:
            # Use CrewAI's built-in RAG client to search
            results = self.rag_client.search(query, n_results=3)
            
            if not results or not results.get('documents'):
                return "No relevant information found in the document database."
            
            context = "Relevant information from documents:\n\n"
            
            # Process results from ChromaDB
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results.get('metadatas') else []
            
            for i, doc in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}
                source = metadata.get('source', 'Unknown source')
                context += f"From {source}:\n{doc}\n\n"
            
            return context
            
        except Exception as e:
            logger.error(f"Error in document search: {e}")
            return f"Error searching documents: {str(e)}"


class RAGChatbot:
    """Main RAG Chatbot class using CrewAI's ChromaDB integration"""
    
    def __init__(
        self,
        ollama_model: str = "llama3.2:1b",
        embedding_model: str = "nomic-embed-text",
        collection_name: str = "documents"
    ):
        # Initialize components
        self.llm = Ollama(model=ollama_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Setup CrewAI ChromaDB configuration
        self._setup_chromadb(collection_name)
        
        # Initialize other components
        self.pdf_processor = PDFProcessor()
        self.memory = ConversationMemory()
        
        # Initialize CrewAI components
        self.document_tool = DocumentSearchTool(self.rag_client)
        self._setup_crew()
    
    def _setup_chromadb(self, collection_name: str):
        """Setup ChromaDB using CrewAI's RAG configuration"""
        try:
            # Clear any existing RAG config
            clear_rag_config()
            
            # Set up ChromaDB configuration
            config = ChromaDBConfig(
                collection_name=collection_name,
                persist_directory="./chromadb_data"
            )
            set_rag_config(config)
            
            # Get the RAG client
            self.rag_client = get_rag_client()
            
            logger.info(f"ChromaDB initialized with collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Error setting up ChromaDB: {e}")
            raise
    
    def _setup_crew(self):
        """Setup CrewAI agent and crew"""
        # CrewAI expects model format like "ollama/model_name"
        crewai_model = f"ollama/{self.llm.model}"
        
        self.agent = Agent(
            role="Document Assistant",
            goal="Help users find information from uploaded documents and maintain conversational context",
            backstory="""You are an intelligent document assistant that can search through 
                        uploaded PDF documents to answer questions. You maintain conversation 
                        context and provide accurate, helpful responses based on the available 
                        document content.""",
            tools=[self.document_tool],
            llm=crewai_model,
            verbose=True
        )
    
    def add_document_from_path(self, pdf_path: str) -> bool:
        """Add a PDF document to the knowledge base"""
        try:
            # Process PDF
            documents = self.pdf_processor.process_pdf(pdf_path)
            
            if not documents:
                logger.warning(f"No content extracted from {pdf_path}")
                return False
            
            # Add documents to ChromaDB via CrewAI
            for doc in documents:
                self.rag_client.add(
                    texts=[doc.page_content],
                    metadatas=[doc.metadata],
                    ids=[str(uuid.uuid4())]
                )
            
            logger.info(f"Added {len(documents)} chunks from {pdf_path} to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {pdf_path}: {e}")
            return False
    
    def add_document_from_text(self, text: str, source: str = "text_input") -> bool:
        """Add text content to the knowledge base"""
        try:
            # Process text
            documents = self.pdf_processor.process_text(text, source=source)
            
            if not documents:
                logger.warning("No content to add")
                return False
            
            # Add documents to ChromaDB via CrewAI
            for doc in documents:
                self.rag_client.add(
                    texts=[doc.page_content],
                    metadatas=[doc.metadata],
                    ids=[str(uuid.uuid4())]
                )
            
            logger.info(f"Added {len(documents)} chunks from text to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Error adding text: {e}")
            return False
    
    def chat(self, message: str, session_id: str) -> str:
        """Main chat function"""
        try:
            # Add user message to conversation memory
            self.memory.add_message(session_id, "user", message)
            
            # Get conversation context
            conversation_context = self.memory.get_context_string(session_id)
            
            # Create task for the agent
            task_description = f"""
            User Query: {message}
            
            Conversation Context:
            {conversation_context}
            
            Please help the user by:
            1. Using the document_search tool to find relevant information if the query relates to document content
            2. Maintaining conversation context and referencing previous messages when relevant
            3. Providing a helpful, accurate response based on available information
            4. If no relevant documents are found, provide a general helpful response
            """
            
            task = Task(
                description=task_description,
                agent=self.agent,
                expected_output="A helpful response that addresses the user's query using available document information and conversation context"
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                verbose=False
            )
            
            # Execute the task
            result = crew.kickoff()
            response = str(result)
            
            # Add assistant response to conversation memory
            self.memory.add_message(session_id, "assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_response = "I'm sorry, I encountered an error while processing your request. Please try again."
            self.memory.add_message(session_id, "assistant", error_response)
            return error_response
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        return self.memory.get_conversation_history(session_id)
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        self.memory.clear_conversation(session_id)
    
    def search_documents(self, query: str) -> str:
        """Direct document search without conversation context"""
        return self.document_tool._run(query)
