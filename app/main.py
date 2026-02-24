
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging
import asyncio

from app.core.csrf import CSRFProtectionMiddleware

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
    # Startup: Validate environment secrets
    try:
        settings.validate_secrets()
        logger.info("Environment secrets validation passed")
    except Exception as e:
        logger.critical(f"Environment validation failed: {e}")
        raise
    
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
    
    # Startup: Database Pre-flight Check
    from app.services.health_check import HealthCheckService
    from app.db.supabase import get_supabase
    
    # Default to True so we don't block if check fails catastrophically/timeouts
    # But ideally strictly False until proven True. Let's go with False for safety.
    app.state.db_ready = False 
    app.state.db_missing_tables = []
    
    try:
        supabase = get_supabase()
        health_service = HealthCheckService(supabase_client=supabase)
        schema_status = await health_service.check_schema_readiness()
        
        app.state.db_ready = schema_status["ready"]
        app.state.db_missing_tables = schema_status.get("missing_tables", [])
        
        if not app.state.db_ready:
            logger.critical(f"DATABASE NOT READY. Missing tables: {app.state.db_missing_tables}")
            logger.critical("Run 'app/db/schema.sql' and 'app/db/ingestion_jobs.sql' in Supabase SQL Editor.")
            raise RuntimeError("Database schema not properly configured. Application cannot start.")
        else:
            logger.info("Database schema verification passed.")
            
    except Exception as e:
        logger.error(f"Failed to perform DB pre-flight check: {e}")
        raise RuntimeError("Database health check failed. Application cannot start.")
    
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

# Add HTTPS redirect middleware if configured
if settings.FORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

# Add CSRF protection middleware
app.add_middleware(CSRFProtectionMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup global error handlers
setup_error_handlers(app)

# Context Processor to inject DB status into all templates
@app.middleware("http")
async def add_global_context(request: Request, call_next):
    # This is a bit of a hack since FastAPI doesn't have standard context processors like Flask/Django
    # We rely on the routes manipulating the templates.TemplateResponse to include request.app.state
    # But we can't easily inject into Jinja env globals dynamically per request in middleware without
    # touching the templates object.
    
    # Better approach: Update Jinja2Templates env globals on startup or update per request
    # Since state is global, we can just access it in the router or attach to request state
    request.state.db_ready = getattr(app.state, "db_ready", False)
    request.state.db_missing_tables = getattr(app.state, "db_missing_tables", [])
    
    response = await call_next(request)
    return response

# Inject the variable into Jinja environment globals so it's available everywhere
# We do this dynamically in a dependency or just update the templates env once if static.
# Since it might change (if we implemented a re-check), accessing state is better.
# For simplicity, we'll patch the template response or adding it to the router's common data.
# HOWEVER, the cleanest way in FastAPI + Jinja is checking the request.app.state in the template? 
# No, Jinja can't access app directly.
# Let's add a context processor helper.

def get_db_status(request: Request):
    return {
        "db_ready": getattr(request.app.state, "db_ready", False),
        "db_missing_tables": getattr(request.app.state, "db_missing_tables", [])
    }

# Update all template responses is tedious. 
# Plan B: We already pass 'request' to all templates. 
# We can just extend the base template to check `request.app.state.db_ready`.
# FastAPI allows accessing `request.app` inside Jinja if request is passed.

from app.routers import auth, ingestion, dashboard, pages, health, pipeline
app.include_router(auth.router)
app.include_router(ingestion.router)
app.include_router(dashboard.router)
app.include_router(pages.router)
app.include_router(health.router, prefix="/api")
app.include_router(pipeline.router)
