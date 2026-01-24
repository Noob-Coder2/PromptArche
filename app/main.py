from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="PromprtArche")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return {"message": "Welcome to PromprtArche"}
