"""
HR Bot V2 - FastAPI Application

REST API for the Enterprise Assistant.

Endpoints:
- POST /chat - Process a chat message
- POST /chat/stream - Stream chat response
- GET /health - Health check
- GET /sessions/{session_id}/history - Get conversation history
- DELETE /sessions/{session_id} - Clear session
"""

import logging
import uuid
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hr_bot.config.settings import get_settings
from hr_bot.deep_agents import get_manager_agent, ManagerAgent

logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User's message/question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    include_history: bool = Field(True, description="Include conversation history in context")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Session ID")
    success: bool = Field(True, description="Whether the request was successful")
    agents_consulted: List[str] = Field(default_factory=list, description="Specialist agents that were consulted")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("healthy", description="Service status")
    version: str = Field("2.0.0", description="API version")
    model: str = Field(..., description="Manager model name")


class ConversationHistory(BaseModel):
    """Conversation history response."""
    session_id: str
    messages: List[Dict[str, str]]
    message_count: int


# ============================================================================
# Session Manager
# ============================================================================

class SessionManager:
    """Manage multiple user sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, ManagerAgent] = {}
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> tuple[str, ManagerAgent]:
        """Get existing session or create new one."""
        if session_id and session_id in self.sessions:
            return session_id, self.sessions[session_id]
        
        # Create new session
        new_session_id = session_id or str(uuid.uuid4())
        self.sessions[new_session_id] = ManagerAgent()
        return new_session_id, self.sessions[new_session_id]
    
    def get_session(self, session_id: str) -> Optional[ManagerAgent]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def clear_all(self) -> None:
        """Clear all sessions."""
        self.sessions.clear()


# Global session manager
session_manager = SessionManager()


# ============================================================================
# Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        logger.info("HR Bot V2 API starting up...")
        yield
        logger.info("HR Bot V2 API shutting down...")
        session_manager.clear_all()
    
    app = FastAPI(
        title="HR Bot V2 API",
        description="Enterprise Assistant API with HR and Finance specialists",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ========================================================================
    # Routes
    # ========================================================================
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            model=settings.OPENAI_MODEL
        )
    
    @app.post("/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(request: ChatRequest):
        """
        Process a chat message.
        
        Routes the message to appropriate specialist (HR or Finance)
        and returns the response.
        """
        try:
            session_id, manager = session_manager.get_or_create_session(request.session_id)
            
            result = await manager.ainvoke(
                request.message,
                session_id=session_id,
                include_history=request.include_history
            )
            
            return ChatResponse(
                response=result["output"],
                session_id=session_id,
                success=result.get("success", True),
                agents_consulted=result.get("metadata", {}).get("agents_consulted", []),
                metadata={
                    "delegations": result.get("delegations", []),
                    "model": result.get("metadata", {}).get("model")
                }
            )
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/chat/sync", response_model=ChatResponse, tags=["Chat"])
    def chat_sync(request: ChatRequest):
        """
        Process a chat message synchronously.
        
        Use this endpoint when async is not needed.
        """
        try:
            session_id, manager = session_manager.get_or_create_session(request.session_id)
            
            result = manager.invoke(
                request.message,
                session_id=session_id,
                include_history=request.include_history
            )
            
            return ChatResponse(
                response=result["output"],
                session_id=session_id,
                success=result.get("success", True),
                agents_consulted=result.get("metadata", {}).get("agents_consulted", []),
                metadata={
                    "delegations": result.get("delegations", []),
                    "model": result.get("metadata", {}).get("model")
                }
            )
            
        except Exception as e:
            logger.error(f"Sync chat error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/sessions/{session_id}/history", response_model=ConversationHistory, tags=["Sessions"])
    async def get_history(session_id: str):
        """Get conversation history for a session."""
        manager = session_manager.get_session(session_id)
        
        if not manager:
            raise HTTPException(status_code=404, detail="Session not found")
        
        history = manager.get_conversation_history()
        
        return ConversationHistory(
            session_id=session_id,
            messages=history,
            message_count=len(history)
        )
    
    @app.delete("/sessions/{session_id}", tags=["Sessions"])
    async def delete_session(session_id: str):
        """Delete a session and its conversation history."""
        if session_manager.delete_session(session_id):
            return {"status": "deleted", "session_id": session_id}
        raise HTTPException(status_code=404, detail="Session not found")
    
    @app.post("/sessions/{session_id}/reset", tags=["Sessions"])
    async def reset_session(session_id: str):
        """Reset conversation history for a session (keep session)."""
        manager = session_manager.get_session(session_id)
        
        if not manager:
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager.reset_conversation()
        return {"status": "reset", "session_id": session_id}
    
    return app


# ============================================================================
# Run directly
# ============================================================================

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
