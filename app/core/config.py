import os
from dotenv import load_dotenv

# Carga las variables definidas en el archivo .env
load_dotenv()

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "DocuClaro AI")
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

settings = Settings()