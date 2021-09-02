from fastapi import FastAPI, Query, status
from manifestacao import Manifestacao
from joblib import load
import classificador as clf
import numpy as np
from enum import Enum

app = FastAPI()

# app.include_router(router_classificacao, tags=["classificador", "classificacao"])
@app.on_event("startup")
async def load_model():
    """
    Carrega os modelos de classificacao e assunto
    """
    clf.model_classificacao = load('modelos/RF_class.joblib')
    # clf.model_assunto = load('modelos/calib_assuntos_desc.joblib')
    clf.model_assunto_id = load('modelos/2021-08-23_assuntos.joblib')


@app.post("/predict_assunto_id", tags=['classificadores', "assunto"])
async def predict_assunto_id(manifestacao: Manifestacao,
         n_assuntos: int = Query(10, title="Número de assuntos a retornar")):
    data = dict(manifestacao)['relato']
    assunto = clf.model_assunto_id.predict_proba([data])
    top_n = np.argsort(assunto)[:n_assuntos].tolist()
    return {'assuntos': clf.model_assunto_id.classes_[tuple(top_n)].tolist()[:-n_assuntos:-1],
            'probabilidades': np.sort(assunto).flatten().tolist()[:-n_assuntos:-1]}

@app.post("/predict_classificacao",
          tags=["classificadores", "classificação"])
async def predict_classificacao(manifestacao: Manifestacao):
    data = dict(manifestacao)['relato']
    resposta ={"classificação": clf.model_classificacao.predict([data]).tolist()}
    return resposta

@app.get("/health_check", tags=['meta', 'status'])
async def health_check(status_code=status.HTTP_200_OK):
    return True
