
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

from app.routers import web
app.include_router(web.router)
