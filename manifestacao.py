from pydantic import BaseModel
from typing import List, Optional
from fastapi import Query
from uuid import UUID


class Manifestacao(BaseModel):
    relato: str
    n_assuntos: Optional[int] = None


class ManifestacaoInDB(BaseModel):
    uuid: UUID
    texto_hash: str

