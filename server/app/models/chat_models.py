from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    
class MultiAgentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class MultiAgentChatResponse(BaseModel):
    response: str
    session_id: str
    agent_used: Optional[str] = None
    query_analysis: Optional[dict] = None
