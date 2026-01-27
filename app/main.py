
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

from app.core.config import settings
from app.core.security import verify_jwt

import logging

app = FastAPI(title="PromprtArche")

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "supabase_url": settings.SUPABASE_URL,
        "supabase_key": settings.SUPABASE_KEY
    })

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "supabase_url": settings.SUPABASE_URL,
        "supabase_key": settings.SUPABASE_KEY
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    # Determine auth via cookie for page loads
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login")
    
    # Verify token (optional deep verification here or trust signature)
    # For now, let's assume if valid signature/not expired it's okay.
    # We can reuse verify_jwt logic if we wrap it to handle strings.
    
    return templates.TemplateResponse("base.html", {"request": request}) # Placeholder

from app.routers import web
app.include_router(web.router)
