from typing import List
from pydantic import BaseModel, Field


class MedicamentoSimple(BaseModel):
    nombre: str
    como_tomarlo: str


class ResumenSimplificado(BaseModel):
    que_tuve: str
    medicamentos: List[MedicamentoSimple] = Field(default_factory=list)
    que_hacer: List[str] = Field(default_factory=list)
    cuidados: List[str] = Field(default_factory=list)
    senales_de_alarma: List[str] = Field(default_factory=list)