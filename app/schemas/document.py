from pydantic import BaseModel, Field
from typing import List, Optional


class MedicamentoAlta(BaseModel):
    nombre: str
    dosis_instrucciones: str
    observaciones: Optional[str] = None  # Para indicaciones especiales como "subir/suspender dosis"


class ProximoPasoCita(BaseModel):
    lugar_o_servicio: str  # Ej. "Consulta Externa de Gastroenterología" o "Unidad de Medicina Familiar"
    accion: str            # Ej. "Sacar cita", "Acudir a seguimiento", "Recoger recetas"


class ResumenNotaEgreso(BaseModel):
    paciente_nombre: str
    fecha_egreso: str
    diagnostico_simplificado: str
    medicamentos: List[MedicamentoAlta] = Field(default_factory=list)
    siguientes_pasos: List[ProximoPasoCita] = Field(default_factory=list)
    cuidados: List[str] = Field(default_factory=list)  # NUEVO
    senales_de_alarma: List[str] = Field(default_factory=list)