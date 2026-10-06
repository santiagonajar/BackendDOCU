import traceback
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.document import ResumenNotaEgreso
from app.schemas.simplified import ResumenSimplificado
from app.core.config import settings
from app.services.llm_service import LLMService

router = APIRouter()
llm_service = LLMService()

MAX_PAGES = 10  # igual que en el frontend


# ------------------------------------------------------------------
# Endpoint 1: Verificación de Variables de Entorno (.env)
# OJO: bórralo o protégelo antes de publicar la app.
# ------------------------------------------------------------------
@router.get("/test-config")
def test_config():
    key_preview = settings.GEMINI_API_KEY[:8] + "..." if settings.GEMINI_API_KEY else "No encontrada"

    models_list = []
    try:
        for m in llm_service.client.models.list():
            models_list.append(m.name)
    except Exception as e:
        models_list = [f"Error al listar: {str(e)}"]

    return {
        "project_name": settings.PROJECT_NAME,
        "api_key_loaded": bool(settings.GEMINI_API_KEY),
        "key_preview": key_preview,
        "supported_models": models_list,
    }


# ------------------------------------------------------------------
# Endpoint 2: Procesa TODAS las páginas de una misma nota en una sola petición
# El campo del formulario se llama "files" (puede repetirse varias veces).
# ------------------------------------------------------------------
@router.post("/process-document", response_model=ResumenNotaEgreso)
async def process_document(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Debes enviar al menos una imagen.")
    if len(files) > MAX_PAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo {MAX_PAGES} páginas por nota.",
        )

    for f in files:
        if not (f.content_type or "").startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Todos los archivos deben ser imágenes en formato JPG o PNG.",
            )

    try:
        # Se conserva el orden en que llegaron las páginas
        pages = [(await f.read(), f.content_type) for f in files]

        return await llm_service.analyze_discharge_note(pages)

    except HTTPException:
        raise

    except Exception as e:
        print("\n" + "=" * 60)
        print("EXCEPCIÓN NO CONTROLADA EN /process-document:")
        traceback.print_exc()
        print("=" * 60 + "\n")

        raise HTTPException(
            status_code=500,
            detail=f"Error interno al procesar las imágenes con la IA: {str(e)}",
        )


# ------------------------------------------------------------------
# Endpoint 3: "No, explícamelo más fácil"
# Recibe el resumen estructurado (JSON) y devuelve una explicación nivel principiante.
# No vuelve a enviar las imágenes: solo el texto ya resumido.
# ------------------------------------------------------------------
@router.post("/simplify-further", response_model=ResumenSimplificado)
async def simplify_further(resumen: ResumenNotaEgreso):
    try:
        # Compatible con Pydantic v1 y v2
        data = resumen.model_dump() if hasattr(resumen, "model_dump") else resumen.dict()

        return await llm_service.simplify_summary(data)

    except HTTPException:
        raise

    except Exception as e:
        print("\n" + "=" * 60)
        print("EXCEPCIÓN NO CONTROLADA EN /simplify-further:")
        traceback.print_exc()
        print("=" * 60 + "\n")

        raise HTTPException(
            status_code=500,
            detail=f"Error interno al simplificar el resumen: {str(e)}",
        )