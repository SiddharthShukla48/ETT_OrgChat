import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json

# Set dummy OpenAI API key for CrewAI tools (even when using Ollama)
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-ollama")

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai_tools import (
    CSVSearchTool,
    PDFSearchTool,
    JSONSearchTool,
    RagTool
)

# ChromaDB RAG imports 
from crewai.rag.config.utils import set_rag_config, get_rag_client, clear_rag_config
from crewai.rag.chromadb.config import ChromaDBConfig

# LLM imports
from langchain_community.llms import Ollama

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentRAGSystem:
    """Multi-agent RAG system with specialized agents for different file types"""
    
    def __init__(
        self,
        ollama_model: str = "llama3.2:1b",
        rag_context_path: str = "./RAG_context",
        collection_name: str = "multi_agent_rag"
    ):
        self.ollama_model = f"ollama/{ollama_model}"
        self.rag_context_path = rag_context_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB for general RAG
        self._setup_chromadb()
        
        # Initialize specialized tools
        self._setup_tools()
        
        # Initialize agents
        self._setup_agents()
        
        # Conversation memory
        self.conversations: Dict[str, List[Dict]] = {}
        
        logger.info("Multi-agent RAG system initialized successfully!")
    
    def _setup_chromadb(self):
        """Setup ChromaDB using CrewAI's RAG configuration"""
        try:
            clear_rag_config()
            config = ChromaDBConfig(
                collection_name=self.collection_name,
                persist_directory="./chromadb_data"
            )
            set_rag_config(config)
            self.rag_client = get_rag_client()
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error setting up ChromaDB: {e}")
            raise
    
    def _setup_tools(self):
        """Setup specialized tools for different file types"""
        try:
            # CSV Search Tool for projects data
            self.csv_tool = CSVSearchTool(
                csv=os.path.join(self.rag_context_path, "projects.csv"),
                config=dict(
                    llm=dict(
                        provider="ollama",
                        config=dict(
                            model="llama3.2:1b",
                        ),
                    ),
                    embedder=dict(
                        provider="ollama",
                        config=dict(
                            model="nomic-embed-text",
                        ),
                    ),
                )
            )
            
            # PDF Search Tool for policy documents
            self.pdf_tool = PDFSearchTool(
                pdf=os.path.join(self.rag_context_path, "sample_policy_and_procedures_manual (1).pdf"),
                config=dict(
                    llm=dict(
                        provider="ollama",
                        config=dict(
                            model="llama3.2:1b",
                        ),
                    ),
                    embedder=dict(
                        provider="ollama",
                        config=dict(
                            model="nomic-embed-text",
                        ),
                    ),
                )
            )
            
            # JSON Search Tool for organizational data
            self.json_tool = JSONSearchTool(
                json_path=os.path.join(self.rag_context_path, "rag_context_organizational_data.json"),
                config=dict(
                    llm=dict(
                        provider="ollama",
                        config=dict(
                            model="llama3.2:1b",
                        ),
                    ),
                    embedder=dict(
                        provider="ollama",
                        config=dict(
                            model="nomic-embed-text",
                        ),
                    ),
                )
            )
            
            # General RAG Tool using ChromaDB
            self.rag_tool = RagTool(
                config=dict(
                    llm=dict(
                        provider="ollama",
                        config=dict(
                            model="llama3.2:1b",
                        ),
                    ),
                    embedder=dict(
                        provider="ollama",
                        config=dict(
                            model="nomic-embed-text",
                        ),
                    ),
                )
            )
            
            logger.info("All specialized tools initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error setting up tools: {e}")
            raise
    
    def _setup_agents(self):
        """Setup specialized agents for different data types"""
        
        # Projects & Employee Data Specialist (CSV)
        self.projects_agent = Agent(
            role="Projects & Employee Data Specialist",
            goal="Search and analyze project assignments, employee information, and departmental data",
            backstory="""You are an expert in analyzing project data and employee assignments. 
                        You have access to comprehensive project databases and can provide detailed 
                        information about employee roles, project timelines, and departmental structures.""",
            tools=[self.csv_tool],
            llm=self.ollama_model,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
        
        # Policy & Procedures Specialist (PDF)
        self.policy_agent = Agent(
            role="Policy & Procedures Specialist",
            goal="Search and interpret company policies, procedures, and regulatory documents",
            backstory="""You are a compliance and policy expert who specializes in company 
                        procedures and regulatory requirements. You can quickly locate and 
                        interpret policy information from official documents.""",
            tools=[self.pdf_tool],
            llm=self.ollama_model,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
        
        # Organizational Data Analyst (JSON)
        self.org_agent = Agent(
            role="Organizational Data Analyst",
            goal="Analyze organizational structure, employee details, and company information",
            backstory="""You are an organizational analyst with deep knowledge of company 
                        structure, employee hierarchies, and organizational data. You can 
                        provide insights into company demographics and organizational patterns.""",
            tools=[self.json_tool],
            llm=self.ollama_model,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
        
        # Knowledge Synthesis Manager
        self.synthesis_agent = Agent(
            role="Knowledge Synthesis Manager",
            goal="Coordinate information from multiple sources and provide comprehensive responses",
            backstory="""You are a senior knowledge manager who excels at synthesizing 
                        information from multiple sources. You coordinate with various specialists 
                        to provide complete, accurate, and well-structured responses to complex queries.""",
            tools=[self.rag_tool],
            llm=self.ollama_model,
            verbose=True,
            allow_delegation=True,
            max_iter=5
        )
        
        logger.info("All agents initialized successfully!")
    
    def _add_to_conversation(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def _get_conversation_context(self, session_id: str) -> str:
        """Get recent conversation context"""
        if session_id not in self.conversations:
            return ""
        
        recent_messages = self.conversations[session_id][-6:]  # Last 6 messages
        context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in recent_messages
        ])
        return f"Recent conversation:\n{context}\n" if context else ""
    
    def analyze_query_type(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine which agents should be involved"""
        query_lower = query.lower()
        
        # Keywords for different data types
        project_keywords = ['project', 'employee', 'role', 'department', 'assignment', 'team', 'lead', 'contributor']
        policy_keywords = ['policy', 'procedure', 'regulation', 'compliance', 'rule', 'guideline', 'manual']
        org_keywords = ['organization', 'company', 'salary', 'hire', 'manager', 'structure', 'employee details']
        
        # Determine relevance scores
        project_score = sum(1 for keyword in project_keywords if keyword in query_lower)
        policy_score = sum(1 for keyword in policy_keywords if keyword in query_lower)
        org_score = sum(1 for keyword in org_keywords if keyword in query_lower)
        
        return {
            'requires_projects': project_score > 0,
            'requires_policy': policy_score > 0,
            'requires_org': org_score > 0,
            'is_general': project_score == 0 and policy_score == 0 and org_score == 0,
            'scores': {
                'projects': project_score,
                'policy': policy_score,
                'organization': org_score
            }
        }
    
    def chat(self, message: str, session_id: str = "default") -> str:
        """Main chat function with multi-agent coordination"""
        try:
            # Add user message to conversation
            self._add_to_conversation(session_id, "user", message)
            
            # Get conversation context
            context = self._get_conversation_context(session_id)
            
            # Analyze query to determine agent involvement
            query_analysis = self.analyze_query_type(message)
            
            # Create tasks based on query analysis
            tasks = []
            agents_involved = []
            
            if query_analysis['requires_projects']:
                project_task = Task(
                    description=f"""
                    Analyze the user's query about projects and employee data: "{message}"
                    
                    {context}
                    
                    Search the projects database for relevant information about:
                    - Employee assignments and roles
                    - Project details and timelines  
                    - Department information
                    - Team structures
                    
                    Provide specific data and insights based on your search results.
                    """,
                    agent=self.projects_agent,
                    expected_output="Detailed information from projects and employee database"
                )
                tasks.append(project_task)
                agents_involved.append(self.projects_agent)
            
            if query_analysis['requires_policy']:
                policy_task = Task(
                    description=f"""
                    Search company policies and procedures for information related to: "{message}"
                    
                    {context}
                    
                    Look for relevant policies, procedures, guidelines, and regulatory information.
                    Provide accurate citations and specific policy details.
                    """,
                    agent=self.policy_agent,
                    expected_output="Relevant policy and procedure information"
                )
                tasks.append(policy_task)
                agents_involved.append(self.policy_agent)
            
            if query_analysis['requires_org']:
                org_task = Task(
                    description=f"""
                    Analyze organizational data for information related to: "{message}"
                    
                    {context}
                    
                    Search for:
                    - Employee details and hierarchies
                    - Organizational structure
                    - Company information
                    - Management relationships
                    """,
                    agent=self.org_agent,
                    expected_output="Organizational and employee information"
                )
                tasks.append(org_task)
                agents_involved.append(self.org_agent)
            
            # Always include synthesis task
            synthesis_task = Task(
                description=f"""
                User Query: "{message}"
                
                {context}
                
                Based on the information gathered by specialist agents, provide a comprehensive 
                and well-structured response that:
                1. Directly addresses the user's question
                2. Integrates information from multiple sources if applicable
                3. Is clear, accurate, and helpful
                4. Maintains conversational context
                
                If this is a general query not covered by specialists, provide a helpful general response.
                """,
                agent=self.synthesis_agent,
                expected_output="A comprehensive, well-structured response to the user's query",
                context=tasks if tasks else None
            )
            tasks.append(synthesis_task)
            agents_involved.append(self.synthesis_agent)
            
            # Create and execute crew
            crew = Crew(
                agents=agents_involved,
                tasks=tasks,
                process=Process.sequential,
                verbose=False
            )
            
            # Execute the crew
            result = crew.kickoff()
            response = str(result)
            
            # Add response to conversation
            self._add_to_conversation(session_id, "assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in multi-agent chat: {e}")
            error_response = "I apologize, but I encountered an error while processing your request. Please try again."
            self._add_to_conversation(session_id, "assistant", error_response)
            return error_response
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        return self.conversations.get(session_id, [])
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    def add_document_to_rag(self, text: str, metadata: Dict = None) -> bool:
        """Add additional documents to the general RAG system"""
        try:
            doc_data = {
                "content": text,
                "metadata": metadata or {}
            }
            
            self.rag_client.add_documents(
                documents=[doc_data],
                collection_name=self.collection_name
            )
            return True
        except Exception as e:
            logger.error(f"Error adding document to RAG: {e}")
            return False
