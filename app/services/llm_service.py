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
# PROMPT 1: leer la nota (todas las páginas juntas)
# ─────────────────────────────────────────────────────────────
PROMPT = """
Eres un asistente que explica Notas de Egreso / Hojas de Alta del IMSS a adultos mayores y a sus cuidadores.
Las imágenes que recibes son las páginas, EN ORDEN, de UNA SOLA nota de egreso de UN solo paciente.
Léelas como un solo documento: combina la información y NO repitas datos que aparezcan en varias páginas.

ESTILO (muy importante):
- Español sencillo, frases cortas, sin tecnicismos. Habla directo al paciente.
- Respuestas MUY breves y orientadas a la acción. Nada de explicaciones largas: un adulto mayor se abruma fácil.
- Cada acción empieza con un verbo en imperativo (Tome, Acuda, Saque, Evite, Regrese...).

OMITE por completo laboratorios, paraclínicos, escalas clínicas (MELD, CHILD PUGH, KDIGO, etc.) y detalles técnicos hospitalarios.

PRIORIDAD: se acortan las FRASES, nunca la información de seguridad. Brevedad sí, pero NO omitas
ningún medicamento, dosis, cuidado, cita ni señal de alarma que aparezca en la nota.

REGLAS POR CAMPO:
- diagnostico_simplificado: MÁXIMO 2 frases cortas (unas 30 palabras). Qué tuvo y cómo sale del hospital.
- medicamentos: incluye TODOS los que aparezcan, sin repetir y sin límite. "dosis_instrucciones" en una sola
  línea corta (Ej. "15 ml cada 12 horas"). "observaciones" solo si es indispensable (subir, suspender,
  con alimentos); si no, null.
- siguientes_pasos: incluye TODAS las citas e indicaciones de seguimiento, sin límite. Cada "accion" en una
  línea, empezando con verbo. Incluye fecha u hora si aparece.
- cuidados: indicaciones para cuidarse en casa que aparezcan en la nota: dieta, actividad, qué evitar o
  suspender (Ej. "Evite el ibuprofeno y otros antiinflamatorios", "Coma con poca sal"). Incluye TODAS,
  cada una corta y empezando con verbo. Si la nota no trae ninguna, devuelve una lista vacía [].
- senales_de_alarma: incluye TODAS las que aparezcan en la nota, sin omitir ninguna. Cada una corta,
  de menos de 10 palabras (Ej. "Fiebre de más de 38 grados").

NUNCA inventes dosis, fechas, nombres ni diagnósticos. Si un dato no aparece en la nota, usa
"No aparece en el documento" (o null en observaciones). Si un medicamento, dosis o fecha está borroso,
cortado o no se alcanza a leer, escribe "No se alcanza a leer, confirme con su médico" en lugar de adivinar.
No agregues consejos médicos que no estén en la nota.

Responde EXCLUSIVAMENTE con un JSON válido que cumpla este esquema:

{
  "paciente_nombre": "Nombre completo del paciente en mayúsculas",
  "fecha_egreso": "Fecha de alta (Ej. 05 de Agosto de 2026)",
  "diagnostico_simplificado": "Máximo 2 frases cortas",
  "medicamentos": [
    {
      "nombre": "Nombre y presentación (Ej. LACTULOSA Suspensión)",
      "dosis_instrucciones": "Dosis y frecuencia en una línea (Ej. 15 ml cada 12 horas)",
      "observaciones": "Solo si es indispensable, o null"
    }
  ],
  "siguientes_pasos": [
    {
      "lugar_o_servicio": "Servicio o unidad (Ej. Consulta Externa de Gastroenterología)",
      "accion": "Acción corta con verbo (Ej. Saque cita de seguimiento)"
    }
  ],
  "cuidados": [
    "Cuidado en casa corto, con verbo (una entrada por cada cuidado de la nota)"
  ],
  "senales_de_alarma": [
    "Síntoma corto para ir a Urgencias (una entrada por cada señal de la nota)"
  ]
}
"""

# ─────────────────────────────────────────────────────────────
# PROMPT 2: re-simplificar el resumen ("No, explícamelo más fácil")
# ─────────────────────────────────────────────────────────────
SIMPLIFY_PROMPT = """
Eres un asistente que ayuda a adultos mayores a entender su hoja de alta del IMSS.
Recibes el RESUMEN (en JSON) de una hoja de alta. La persona leyó ese resumen y NO lo entendió bien.
Vuelve a explicarlo en nivel PRINCIPIANTE, como si hablaras con alguien que nunca ha oído términos médicos.

ESTILO (muy importante):
- Oraciones MUY cortas (de 8 a 12 palabras como máximo). Una idea por oración.
- Palabras de todos los días. Nada de tecnicismos. Si un término médico es inevitable, explícalo
  entre paréntesis con palabras simples.
- Habla directo a la persona ("usted"), con calma y calidez.
- En "que_tuve" puedes usar UNA analogía cotidiana sencilla (Ej. "El hígado es como un filtro del cuerpo"),
  pero solo si es fiel al diagnóstico del resumen. No agregues diagnósticos, causas ni promesas nuevas.

REGLAS DE SEGURIDAD (obligatorias):
- NO cambies ni inventes dosis, horarios, fechas, nombres, lugares ni medicamentos.
  Las cantidades y los horarios deben quedar IGUAL que en el resumen (solo escritos más simple).
- Conserva TODOS los medicamentos, cuidados, citas y señales de alarma del resumen. No omitas ninguno.
- Si algo dice "No aparece en el documento" o "No se alcanza a leer, confirme con su médico", déjalo igual.
- No agregues consejos médicos que no estén en el resumen.

CAMPOS:
- que_tuve: 2 a 4 oraciones muy cortas.
- medicamentos: un elemento por cada medicamento del resumen. "como_tomarlo": cuánto, cuándo y (si aparece) por cuánto tiempo.
- que_hacer: las citas y pasos de seguimiento, cada uno como una acción simple con verbo.
- cuidados: los cuidados en casa, cada uno con verbo. Lista vacía [] si el resumen no trae.
- senales_de_alarma: cada señal en palabras simples. Empieza cada una describiendo lo que se siente o se ve.

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

    # Solo reintenta errores del servidor (5xx). Los errores 4xx no se arreglan reintentando.
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
                temperature=temperature,  # menos creatividad = menos riesgo de inventar datos
            ),
        )

    async def _generate_json(self, contents: list, temperature: float = 0.2) -> dict:
        """Llama a Gemini y devuelve el JSON ya convertido a dict, con manejo de errores común."""
        try:
            # La llamada del SDK es síncrona: se manda a un hilo para no bloquear FastAPI
            response = await asyncio.to_thread(self._call_gemini_api, contents, temperature)

            if not response.text:
                raise HTTPException(status_code=500, detail="La API de IA no devolvió contenido.")

            return json.loads(response.text)

        except HTTPException:
            raise
        except ClientError as e:
            print("\n" + "!" * 60)
            print("ERROR CLIENTE GEMINI (4xx):", e)
            print("!" * 60 + "\n")
            # 429 = cuota agotada (en plan gratis suele ser un límite diario)
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
            print(f"\n[ServerError Gemini]: {e}\n")
            raise HTTPException(
                status_code=503,
                detail="El servicio de Inteligencia Artificial está saturado. Inténtalo de nuevo en un minuto.",
            )
        except json.JSONDecodeError as e:
            print(f"\n[JSONDecodeError]: {e}\n")
            raise HTTPException(
                status_code=500,
                detail="Error al procesar la respuesta en formato JSON.",
            )

    async def analyze_discharge_note(self, pages: List[PageData]) -> dict:
        """Recibe TODAS las páginas de una misma nota y las manda juntas en una sola llamada."""
        total = len(pages)

        contents: list = [PROMPT]
        for i, (data, mime) in enumerate(pages, start=1):
            contents.append(f"Página {i} de {total}:")
            contents.append(types.Part.from_bytes(data=data, mime_type=mime or "image/jpeg"))
        contents.append("Fin de la nota. Devuelve el JSON del resumen de TODA la nota.")

        return await self._generate_json(contents, temperature=0.2)

    async def simplify_summary(self, resumen: dict) -> dict:
        """Toma el resumen estructurado y lo re-explica a nivel principiante."""
        contents: list = [
            SIMPLIFY_PROMPT,
            "RESUMEN ORIGINAL (JSON):",
            json.dumps(resumen, ensure_ascii=False),
            "Devuelve el JSON de la explicación simplificada.",
        ]
        # Un poco más de flexibilidad para la analogía, sin dejar que invente datos
        return await self._generate_json(contents, temperature=0.3)