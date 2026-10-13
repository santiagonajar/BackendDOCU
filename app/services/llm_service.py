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
Eres un asistente que verifica y explica Notas de Egreso / Hojas de Alta del IMSS a adultos mayores y a sus cuidadores.
Las imágenes que recibes son las páginas, EN ORDEN, de UNA SOLA nota de egreso de UN solo paciente.

PASO OBLIGATORIO DE VALIDACIÓN (CRITERIO AMPLIO):
Analiza detenidamente las imágenes. 
- Debes establecer "es_documento_valido": false ÚNICAMENTE si la imagen es una foto totalmente ajena al ámbito médico o sin texto (por ejemplo: un paisaje, un árbol, una mascota, un vehículo, objetos de la calle, comida o una selfie).
- Si la imagen contiene texto impreso, formatos hospitalarios, recetas, notas médicas, tablas de medicamentos o datos de un paciente (aunque esté borrosa, incompleta, tomada con sombra o sea una hoja de alta real del IMSS), DEBES establecer "es_documento_valido": true. Sé muy flexible: ante la duda de si es un documento de salud, acéptalo como true.

ESTILO (si es válido):
- Español sencillo, frases cortas, sin tecnicismos. Habla directo al paciente.
- Respuestas MUY breves y orientadas a la acción.
- Cada acción empieza con un verbo en imperativo (Tome, Acuda, Saque, Evite, Regrese...).

OMITE por completo laboratorios, paraclínicos y escalas clínicas.

REGLAS POR CAMPO:
- es_documento_valido: true o false según la validación anterior.
- paciente_nombre: Nombre completo del paciente o "No aparece en el documento" si no es válido.
- fecha_egreso: Fecha de alta o "No aparece en el documento".
- diagnostico_simplificado: Si no es válido, escribe exactamente: "La foto no se puede analizar porque no corresponde a una nota de alta del IMSS."
- medicamentos: Lista vacía [] si no es válido.
- siguientes_pasos: Lista vacía [] si no es válido.
- cuidados: Lista vacía [] si no es válido.
- senales_de_alarma: Lista vacía [] si no es válido.

Responde EXCLUSIVAMENTE con un JSON válido que cumpla este esquema:

{
  "es_documento_valido": true,
  "paciente_nombre": "Nombre completo del paciente en mayúsculas",
  "fecha_egreso": "Fecha de alta (Ej. 05 de Agosto de 2026)",
  "diagnostico_simplificado": "Máximo 2 frases cortas",
  "medicamentos": [
    {
      "nombre": "Nombre y presentación",
      "dosis_instrucciones": "Dosis y frecuencia en una línea",
      "observaciones": "Solo si es indispensable, o null"
    }
  ],
  "siguientes_pasos": [
    {
      "lugar_o_servicio": "Servicio o unidad",
      "accion": "Acción corta con verbo"
    }
  ],
  "cuidados": [
    "Cuidado en casa corto, con verbo"
  ],
  "senales_de_alarma": [
    "Síntoma corto para ir a Urgencias"
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