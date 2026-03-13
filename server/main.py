import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import Base, engine
from app.routes.auth import router as auth_router
from app.routes.users import router as users_router

# Ensure SQLAlchemy models are imported before create_all
from app.models import user as _user_model  # noqa: F401

logger = logging.getLogger(__name__)

app = FastAPI(
    title="OrgChat API",
    version="1.0.0",
    description="Backend API for OrgChat with RBAC and Groq-powered chat",
)

# Allow local frontend/dev tools.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")


app.include_router(auth_router)
app.include_router(users_router)

# Chat route initializes the RAG system at import time and requires Groq config.
try:
    from app.routes.chat_routes import router as chat_router

    app.include_router(chat_router)
except Exception as exc:
    logger.warning("Chat routes disabled during startup: %s", exc)


@app.get("/")
def root() -> dict:
    return {
        "message": "OrgChat API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
