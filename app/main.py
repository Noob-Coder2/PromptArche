
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
import httpx
import logging

from app.core.config import settings
from app.core.security import verify_jwt

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle resources.
    Creates persistent HTTP client on startup, closes on shutdown.
    """
    # Startup: Create persistent HTTP client for embeddings
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
    )
    logger.info("HTTP client initialized")
    yield
    # Shutdown: Close HTTP client
    await app.state.http_client.aclose()
    logger.info("HTTP client closed")

app = FastAPI(title="PromptArche", lifespan=lifespan)

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")

from app.routers import auth, ingestion, dashboard, pages
app.include_router(auth.router)
app.include_router(ingestion.router)
app.include_router(dashboard.router)
app.include_router(pages.router)
