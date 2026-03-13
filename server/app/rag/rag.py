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
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF and return document chunks"""
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return []
        
        documents = self.text_splitter.create_documents(
            [text],
            metadatas=[{"source": pdf_path, "type": "pdf"}]
        )
        return documents


class ConversationMemory:
    """Manages conversation history and memory"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict]] = {}
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent messages
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        return self.conversations.get(session_id, [])
    
    def get_context_string(self, session_id: str) -> str:
        """Get conversation history as formatted string"""
        history = self.get_conversation_history(session_id)
        context = ""
        for msg in history:
            context += f"{msg['role'].title()}: {msg['content']}\n"
        return context


class VectorSearchTool(BaseTool):
    """CrewAI tool for vector database search"""
    
    name: str = "vector_search"
    description: str = "Search through the document vector database for relevant information"
    
    def __init__(self, vector_store: QdrantVectorStore, embeddings: OllamaEmbeddings):
        super().__init__()
        self._vector_store = vector_store
        self._embeddings = embeddings
    
    def _run(self, query: str) -> str:
        """Execute vector search"""
        try:
            query_embedding = self._embeddings.embed_query(query)
            results = self._vector_store.search(query_embedding, limit=3)
            
            if not results:
                return "No relevant information found in the document database."
            
            context = "Relevant information from documents:\n\n"
            for i, result in enumerate(results, 1):
                context += f"Document {i} (Score: {result['score']:.3f}):\n"
                context += f"{result['content']}\n\n"
            
            return context
        except Exception as e:
            return f"Error searching documents: {str(e)}"


class RAGChatbot:
    """Main RAG chatbot class integrating all components"""
    
    def __init__(
        self,
        ollama_model: str = "llama3.2:1b",
        embedding_model: str = "nomic-embed-text",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333
    ):
        # Initialize components
        self.llm = Ollama(model=ollama_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store = QdrantVectorStore(host=qdrant_host, port=qdrant_port)
        self.pdf_processor = PDFProcessor()
        self.memory = ConversationMemory()
        
        # Initialize CrewAI components
        self.vector_tool = VectorSearchTool(self.vector_store, self.embeddings)
        self._setup_crew()
    
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
            tools=[self.vector_tool],
            llm=crewai_model,
            verbose=True
        )
        
        self.crew = Crew(
            agents=[self.agent],
            verbose=True
        )
    
    def add_pdf(self, pdf_path: str) -> bool:
        """Add a PDF to the vector database"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            documents = self.pdf_processor.process_pdf(pdf_path)
            
            if not documents:
                logger.error("No content extracted from PDF")
                return False
            
            # Generate embeddings
            texts = [doc.page_content for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)
            
            # Add to vector store
            self.vector_store.add_documents(documents, embeddings)
            logger.info(f"Successfully added PDF with {len(documents)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error adding PDF: {e}")
            return False
    
    def chat(self, message: str, session_id: str = "default") -> str:
        """Chat with the bot, maintaining conversation memory"""
        try:
            # Add user message to memory
            self.memory.add_message(session_id, "user", message)
            
            # Get conversation context
            context = self.memory.get_context_string(session_id)
            
            # Create enhanced prompt with context
            enhanced_message = f"""
            Conversation History:
            {context}
            
            Current Question: {message}
            
            Please provide a helpful response based on the available documents and conversation context.
            If you need to search for information, use the vector_search tool.
            """
            
            # Create task
            task = Task(
                description=enhanced_message,
                agent=self.agent,
                expected_output="A helpful and contextual response to the user's question"
            )
            
            # Execute task
            result = self.crew.kickoff(tasks=[task])
            response = str(result)
            
            # Add response to memory
            self.memory.add_message(session_id, "assistant", response)
            
            return response
            
        except Exception as e:
            error_msg = f"Error during chat: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_conversation_history(self, session_id: str = "default") -> List[Dict]:
        """Get conversation history for a session"""
        return self.memory.get_conversation_history(session_id)
    
    def clear_conversation(self, session_id: str = "default"):
        """Clear conversation history for a session"""
        if session_id in self.memory.conversations:
            del self.memory.conversations[session_id]


# Example usage
if __name__ == "__main__":
    # Initialize the RAG chatbot
    chatbot = RAGChatbot(
        ollama_model="llama3.2:1b",  # Change to your preferred model
        embedding_model="nomic-embed-text"
    )
    
    # Add a PDF to the vector database
    pdf_path = "path/to/your/document.pdf"
    if chatbot.add_pdf(pdf_path):
        print("PDF successfully added to the database!")
    
    # Start chatting
    session_id = "user_session_1"
    
    # Example conversation
    response1 = chatbot.chat("What is this document about?", session_id)
    print(f"Bot: {response1}")
    
    response2 = chatbot.chat("Can you give me more details about the main topic?", session_id)
    print(f"Bot: {response2}")
    
    # View conversation history
    history = chatbot.get_conversation_history(session_id)
    print("\nConversation History:")
    for msg in history:
        print(f"{msg['role'].title()}: {msg['content']}")