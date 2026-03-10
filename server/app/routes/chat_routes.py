from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import uuid
from ..rag.simplified_multi_agent_rag import SimplifiedMultiAgentRAGSystem
from ..database.database import get_db_connection
from ..models.chat_models import ChatRequest, ChatResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Initialize the simplified multi-agent RAG system
rag_system = SimplifiedMultiAgentRAGSystem()

class MultiAgentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class MultiAgentChatResponse(BaseModel):
    response: str
    session_id: str
    agent_used: Optional[str] = None
    query_analysis: Optional[dict] = None

@router.post("/multi-agent", response_model=MultiAgentChatResponse)
async def multi_agent_chat(request: MultiAgentChatRequest):
    """
    Multi-agent RAG chat endpoint that routes queries to specialized agents
    for PDF, CSV, and JSON data sources using CrewAI.
    """
    try:
        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = str(uuid.uuid4())
        
        logger.info(f"Processing multi-agent query: {request.message}")
        
        # Use the multi-agent RAG system
        response = rag_system.chat(request.message, request.session_id)
        
        # Get query analysis for debugging/transparency
        query_analysis = rag_system.analyze_query_type(request.message)
        
        # Determine which agent was primarily used based on analysis
        agent_used = None
        if query_analysis.get('requires_projects'):
            agent_used = "Projects & Employee Data Specialist"
        elif query_analysis.get('requires_policy'):
            agent_used = "Policy & Procedures Specialist"
        elif query_analysis.get('requires_org'):
            agent_used = "Organizational Data Analyst"
        else:
            agent_used = "Knowledge Synthesis Manager"
        
        return MultiAgentChatResponse(
            response=response,
            session_id=request.session_id,
            agent_used=agent_used,
            query_analysis=query_analysis
        )
        
    except Exception as e:
        logger.error(f"Error in multi-agent chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multi-agent RAG error: {str(e)}")

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, db=Depends(get_db_connection)):
    """
    Standard chat endpoint (keeping for backward compatibility)
    """
    try:
        # For now, redirect to multi-agent system
        multi_request = MultiAgentChatRequest(
            message=request.message,
            session_id=request.session_id
        )
        
        result = await multi_agent_chat(multi_request)
        
        return ChatResponse(
            response=result.response,
            session_id=result.session_id
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """
    Get conversation history for a session
    """
    try:
        history = rag_system.get_conversation_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"History retrieval error: {str(e)}")

@router.delete("/sessions/{session_id}")
async def clear_chat_session(session_id: str):
    """
    Clear conversation history for a session
    """
    try:
        rag_system.clear_conversation(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Session clear error: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for the multi-agent RAG system
    """
    try:
        # Simple test query to verify system is working
        test_response = rag_system.chat("Hello", "health_check")
        return {
            "status": "healthy",
            "multi_agent_rag": "operational",
            "test_response_length": len(test_response)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy", 
            "error": str(e)
        }
