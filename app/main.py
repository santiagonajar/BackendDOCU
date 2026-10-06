from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(title="DocuClaro AI - API Hoja de Alta")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"status": "ok", "message": "Servidor DocuClaro AI listo"}