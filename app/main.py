
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
import httpx
import logging
import asyncio

from app.core.config import settings
from app.core.security import verify_jwt
from app.core.error_handler import setup_error_handlers
from app.db.supabase import close_supabase_connection
from app.services.task_queue import TaskQueueService

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle resources.
    - Initializes async task queue for background jobs
    - Creates persistent HTTP client for embeddings
    - Recovers in-progress jobs from database on startup
    - Gracefully shuts down all resources
    """
    # Startup: Initialize async task queue
    try:
        await TaskQueueService.initialize()
        logger.info("Async task queue initialized and worker started")
    except Exception as e:
        logger.error(f"Failed to initialize task queue: {e}")
        raise
    
    # Startup: Create persistent HTTP client for embeddings
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
    )
    logger.info("HTTP client initialized")
    logger.info(f"Database pooling configured: pool_size={settings.DB_POOL_SIZE}, timeout={settings.DB_POOL_TIMEOUT}s")
    logger.info(f"Async processing: batch_size={settings.BATCH_SIZE}, max_retries={settings.MAX_RETRIES}")
    
    yield
    
    # Shutdown: Close all resources gracefully
    await app.state.http_client.aclose()
    logger.info("HTTP client closed")
    
    await TaskQueueService.shutdown()
    logger.info("Async task queue shutdown complete")
    
    close_supabase_connection()
    logger.info("Database connection pool closed")

app = FastAPI(title="PromptArche", lifespan=lifespan)

# Setup global error handlers
setup_error_handlers(app)

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")

from app.routers import auth, ingestion, dashboard, pages, health
app.include_router(auth.router)
app.include_router(ingestion.router)
app.include_router(dashboard.router)
app.include_router(pages.router)
app.include_router(health.router, prefix="/api")
