import asyncio
import json
from typing import List, Tuple

from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError, ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from fastapi import HTTPException
from app.core.config import settings

MODEL_NAME = "models/gemini-3.6-flash"

# (bytes de la imagen, mime_type)
PageData = Tuple[bytes, str]

# ─────────────────────────────────────────────────────────────
# PROMPT 1: LEER Y RESUMIR NOTA DE EGRESO
# ─────────────────────────────────────────────────────────────
PROMPT = """
Eres un asistente médico experto que analiza y resume Hojas de Alta / Notas de Egreso médicas (del IMSS o instituciones de salud) para pacientes y sus familias.
Las imágenes que recibes son las páginas, EN ORDEN, de UNA sola nota de egreso.

REGLA DE VALIDACIÓN:
- "es_documento_valido": Pon `true` para cualquier documento médico, hoja de alta, resumen clínico, receta o foto de hoja impresa con texto.
- ÚNICAMENTE pon `false` si la imagen es algo completamente ajeno al ámbito médico o sin texto alguno (por ejemplo: un carro, una mascota, un paisaje, un objeto cualquiera, una pantalla en negro o una selfie).

ESTILO DE REDACCIÓN:
- Español sencillo, claro y empático. Sin tecnicismos ni jerga médica compleja.
- Instrucciones directas orientadas a la acción (usando verbos claros: Tome, Acuda, Asista, Evite, Guarde).
- Respuestas breves y fáciles de entender para un adulto mayor.
- OMITE por completo resultados detallados de laboratorio, escalas clínicas y valores técnicos paraclínicos.

FORMATO DE SALIDA:
Responde EXCLUSIVAMENTE con un JSON válido con esta estructura exacta:

{
  "es_documento_valido": true,
  "paciente_nombre": "Nombre del paciente en mayúsculas (o 'Paciente' si no es visible)",
  "fecha_egreso": "Fecha de alta o fecha visible en la hoja",
  "diagnostico_simplificado": "Explicación breve (máximo 2 frases) en palabras sencillas de la razón de hospitalización y su estado",
  "medicamentos": [
    {
      "nombre": "Nombre del medicamento y presentación",
      "dosis_instrucciones": "Dosis y horarios en una sola línea",
      "observaciones": "Indicación especial o null"
    }
  ],
  "siguientes_pasos": [
    {
      "lugar_o_servicio": "Servicio médico o consultorio al que debe acudir",
      "accion": "Acción clara a realizar"
    }
  ],
  "cuidados": [
    "Cuidado o recomendación general en casa"
  ],
  "senales_de_alarma": [
    "Síntoma clave por el cual debería regresar a Urgencias"
  ]
}
"""

SIMPLIFY_PROMPT = """
Eres un asistente que ayuda a adultos mayores a entender su hoja de alta.
Recibes el RESUMEN (en JSON) de una hoja de alta. Vuelve a explicarlo en nivel PRINCIPIANTE.

Responde EXCLUSIVAMENTE con un JSON válido con este esquema:
{
  "que_tuve": "...",
  "medicamentos": [ { "nombre": "...", "como_tomarlo": "..." } ],
  "que_hacer": [ "..." ],
  "cuidados": [ "..." ],
  "senales_de_alarma": [ "..." ]
}
"""


class LLMService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    @retry(
        retry=retry_if_exception_type(ServerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
    )
    def _call_gemini_api(self, contents: list, temperature: float = 0.2):
        return self.client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=temperature,
            ),
        )

    async def _generate_json(self, contents: list, temperature: float = 0.2) -> dict:
        try:
            response = await asyncio.to_thread(self._call_gemini_api, contents, temperature)

            if not response.text:
                raise HTTPException(status_code=500, detail="La API de IA no devolvió contenido.")

            return json.loads(response.text)

        except HTTPException:
            raise
        except ClientError as e:
            if getattr(e, "code", None) == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Se alcanzó el límite de uso de la inteligencia artificial. Inténtalo más tarde.",
                )
            raise HTTPException(
                status_code=400,
                detail=f"Error en la solicitud a Gemini API: {str(e)}",
            )
        except (ServerError, APIError) as e:
            raise HTTPException(
                status_code=503,
                detail="El servicio de Inteligencia Artificial está saturado. Inténtalo de nuevo en un minuto.",
            )
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail="Error al procesar la respuesta en formato JSON.",
            )

    async def analyze_discharge_note(self, pages: List[PageData]) -> dict:
        total = len(pages)

        contents: list = [PROMPT]
        for i, (data, mime) in enumerate(pages, start=1):
            contents.append(f"Página {i} de {total}:")
            contents.append(types.Part.from_bytes(data=data, mime_type=mime or "image/jpeg"))
        contents.append("Fin de la nota. Devuelve el JSON del resumen de TODA la nota.")

        result = await self._generate_json(contents, temperature=0.2)

        # Solo si la IA explícitamente detectó un objeto/foto random no médico marcamos error 400
        if result.get("es_documento_valido") is False:
            raise HTTPException(
                status_code=400,
                detail="La foto no se puede analizar porque no corresponde a una nota de alta. Por favor, toma una foto clara de tu documento.",
            )

        return result

    async def simplify_summary(self, resumen: dict) -> dict:
        contents: list = [
            SIMPLIFY_PROMPT,
            "RESUMEN ORIGINAL (JSON):",
            json.dumps(resumen, ensure_ascii=False),
            "Devuelve el JSON de la explicación simplificada.",
        ]
        return await self._generate_json(contents, temperature=0.3)