import os
import uuid
import json
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# LLM imports
from langchain_groq import ChatGroq
from app.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleFileSearchTool:
    """Simple tool to search files without complex RAG dependencies"""
    
    def __init__(self, file_path: str, file_type: str):
        self.file_path = file_path
        self.file_type = file_type
        self.content = self._load_content()
    
    def _load_content(self):
        """Load and parse file content"""
        try:
            if self.file_type == "json":
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            elif self.file_type == "csv":
                with open(self.file_path, 'r', newline='') as f:
                    return list(csv.DictReader(f))
            elif self.file_type == "pdf":
                # For now, return a placeholder - PDF processing would need PyPDF2
                return "PDF content processing not implemented in simple version"
            else:
                with open(self.file_path, 'r') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error loading {self.file_type} file: {e}")
            return None
    
    def search(self, query: str) -> str:
        """Search content based on query"""
        query_lower = query.lower()
        
        if self.file_type == "json" and self.content:
            # Search JSON structure
            return self._search_json(query_lower)
        elif self.file_type == "csv" and self.content is not None:
            # Search CSV data
            return self._search_csv(query_lower)
        elif self.file_type == "pdf":
            return f"PDF search for '{query}' - PDF processing needs to be implemented"
        else:
            return f"No results found for '{query}'"
    
    def _search_json(self, query: str) -> str:
        """Search JSON data"""
        results = []
        
        # Search in policies
        if "policy" in query or "policies" in query:
            if "policies" in self.content:
                for policy in self.content["policies"]:
                    policy_text = f"Policy: {policy.get('title', 'N/A')} - {policy.get('description', 'N/A')}"
                    if any(term in policy_text.lower() for term in query.split()):
                        results.append(policy_text)
        
        # Search in company info
        if "company" in query or "organization" in query:
            if "organization_info" in self.content:
                org_info = self.content["organization_info"]
                results.append(f"Company: {org_info.get('company_name', 'N/A')} - {org_info.get('mission', 'N/A')}")
        
        # Search in employees
        if "employee" in query or "employees" in query:
            if "employees" in self.content:
                employee_count = len(self.content["employees"])
                results.append(f"Total employees: {employee_count}")
                # Add sample employee info
                for emp in self.content["employees"][:3]:  # First 3 employees
                    results.append(f"Employee: {emp.get('first_name', '')} {emp.get('last_name', '')} - {emp.get('role', 'N/A')} in {emp.get('department', 'N/A')}")
        
        return "\n".join(results) if results else f"No specific information found for '{query}' in organizational data"
    
    def _search_csv(self, query: str) -> str:
        """Search CSV data"""
        results = []
        rows = self.content or []

        if not rows:
            return f"No project data found for '{query}'"

        columns = set()
        for row in rows:
            columns.update(row.keys())
        
        # Check for department-specific queries
        if "department" in query:
            dept_keywords = ['engineering', 'finance', 'marketing', 'sales', 'operations', 'it support', 'legal']
            found_dept = None
            for dept in dept_keywords:
                if dept in query:
                    found_dept = dept
                    break
            
            if found_dept and 'department' in columns:
                dept_data = [
                    row for row in rows
                    if found_dept in (row.get('department') or '').lower()
                ]
                if dept_data:
                    unique_employees = len({row.get('employee_name') for row in dept_data if row.get('employee_name')}) if 'employee_name' in columns else 0
                    results.append(f"Department Analysis: {unique_employees} unique employees work in the {found_dept.title()} department")
                    
                    # Add sample employee info from this department
                    for row in dept_data[:5]:
                        results.append(f"Employee {row.get('employee_name', 'N/A')} (ID: {row.get('employee_id', 'N/A')}) working on {row.get('project_name', 'N/A')} as {row.get('role_in_project', 'N/A')}")
                    
                    return "\n".join(results)
        
        # General search for projects and employees
        if "project" in query or "employee" in query:
            # Get basic stats
            total_projects = len({row.get('project_id') for row in rows if row.get('project_id')}) if 'project_id' in columns else 0
            total_employees = len({row.get('employee_id') for row in rows if row.get('employee_id')}) if 'employee_id' in columns else 0
            
            results.append(f"Project Database: {total_employees} employees working on {total_projects} projects")
            
            # Add sample data
            for row in rows[:3]:
                results.append(f"Employee {row.get('employee_name', 'N/A')} working on {row.get('project_name', 'N/A')} as {row.get('role_in_project', 'N/A')}")
        
        return "\n".join(results) if results else f"No project data found for '{query}'"


class SimplifiedMultiAgentRAGSystem:
    """Simplified multi-agent RAG system with direct file processing"""
    
    def __init__(
        self,
        groq_model: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        rag_context_path: str = "./RAG_context"
    ):
        self.groq_model = groq_model or settings.groq_model
        self.rag_context_path = rag_context_path

        key = groq_api_key or settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is required to use Groq models")
        
        # Initialize LLM
        self.llm = ChatGroq(model=self.groq_model, api_key=key, temperature=0)
        
        # Initialize file tools
        self._setup_file_tools()
        
        # Conversation memory
        self.conversations: Dict[str, List[Dict]] = {}
        
        logger.info("Simplified multi-agent RAG system initialized successfully!")
    
    def _setup_file_tools(self):
        """Setup simple file processing tools"""
        try:
            # JSON tool for organizational data
            self.json_tool = SimpleFileSearchTool(
                os.path.join(self.rag_context_path, "rag_context_organizational_data.json"),
                "json"
            )
            
            # CSV tool for projects data
            self.csv_tool = SimpleFileSearchTool(
                os.path.join(self.rag_context_path, "projects.csv"),
                "csv"
            )
            
            # PDF tool placeholder
            self.pdf_tool = SimpleFileSearchTool(
                os.path.join(self.rag_context_path, "sample_policy_and_procedures_manual (1).pdf"),
                "pdf"
            )
            
            logger.info("File processing tools initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error setting up file tools: {e}")
            raise
    
    def analyze_query_type(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine which agents should handle it"""
        query_lower = query.lower()
        
        analysis = {
            'requires_projects': False,
            'requires_policy': False,
            'requires_org': False,
            'is_general': False,
            'scores': {
                'projects': 0,
                'policy': 0,
                'organization': 0
            }
        }
        
        # Projects keywords
        project_keywords = ['project', 'assignment', 'team', 'employee', 'role', 'department', 'working on']
        for keyword in project_keywords:
            if keyword in query_lower:
                analysis['scores']['projects'] += 1
        
        # Policy keywords  
        policy_keywords = ['policy', 'procedure', 'rule', 'guideline', 'handbook', 'regulation', 'hiring', 'leave', 'vacation']
        for keyword in policy_keywords:
            if keyword in query_lower:
                analysis['scores']['policy'] += 1
        
        # Organizational keywords
        org_keywords = ['organization', 'company', 'structure', 'hierarchy', 'management', 'organizational']
        for keyword in org_keywords:
            if keyword in query_lower:
                analysis['scores']['organization'] += 1
        
        # Determine requirements
        analysis['requires_projects'] = analysis['scores']['projects'] > 0
        analysis['requires_policy'] = analysis['scores']['policy'] > 0  
        analysis['requires_org'] = analysis['scores']['organization'] > 0
        analysis['is_general'] = sum(analysis['scores'].values()) == 0
        
        return analysis
    
    def chat(self, message: str, session_id: Optional[str] = None) -> str:
        """Process a chat message using the multi-agent system"""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Get conversation history
            conversation_history = self.get_conversation_history(session_id)
            
            # Analyze query type
            query_analysis = self.analyze_query_type(message)
            
            # Search relevant data sources
            search_results = {}
            
            if query_analysis['requires_projects']:
                search_results['projects'] = self.csv_tool.search(message)
            
            if query_analysis['requires_policy']:
                search_results['policy'] = self.pdf_tool.search(message)
            
            if query_analysis['requires_org']:
                search_results['organization'] = self.json_tool.search(message)
            
            # Combine search results for LLM processing
            context_parts = []
            
            if search_results.get('organization'):
                context_parts.append(f"Organizational Data:\n{search_results['organization']}")
            
            if search_results.get('projects'):
                context_parts.append(f"Project Data:\n{search_results['projects']}")
            
            if search_results.get('policy'):
                context_parts.append(f"Policy Data:\n{search_results['policy']}")
            
            if not context_parts:
                final_response = "I don't have specific information about that topic in our current knowledge base. Please contact HR directly for more detailed information."
            else:
                # Use LLM to synthesize a natural response
                context = "\n\n".join(context_parts)
                
                prompt = f"""
Based on the following company data, provide a clear and specific answer to this question: {message}

Available Company Data:
{context}

Instructions:
- Give a direct, helpful answer based only on the provided data
- Include specific numbers, names, dates, and details when available
- If the question asks for counts or statistics, provide exact numbers
- Be conversational and helpful, not just a data dump
- If the data doesn't fully answer the question, acknowledge what information is available

Answer:"""

                try:
                    llm_response = self.llm.invoke(prompt)
                    final_response = getattr(llm_response, "content", str(llm_response))
                except Exception as llm_error:
                    logger.error(f"LLM processing error: {llm_error}")
                    final_response = "\n\n".join(context_parts)  # Fallback to raw data
            
            # Store conversation
            self._store_conversation(session_id, message, final_response)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return f"I apologize, but I encountered an error processing your request. Please try again or contact support."
    
    def _store_conversation(self, session_id: str, user_message: str, assistant_response: str):
        """Store conversation in memory"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].extend([
            {"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": assistant_response, "timestamp": datetime.now().isoformat()}
        ])
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        return self.conversations.get(session_id, [])
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
