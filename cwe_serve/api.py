"""api.py -- service REST du classifieur CVE -> CWE (FastAPI).

Endpoints :
  GET  /health            -> etat du service
  POST /predict           -> { description } -> { cwe, confidence, abstain, threshold, top }

Lancer :  uv run uvicorn cwe_serve.api:app --reload    (http://127.0.0.1:8001 ; docs: /docs)
Le modele est charge paresseusement au premier appel (voir cwe_serve.predict).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cwe_serve.predict import predict, resoudre_texte

app = FastAPI(title="CVE -> CWE", description="Classifieur de type de faille (DistilBERT fine-tune).")

# le front (Vite) tourne sur un autre port -> on autorise l'appel navigateur
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Requete(BaseModel):
    # on accepte soit une description brute, soit le JSON complet d'une CVE (description extraite)
    description: str | None = None
    cve: dict | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predire(req: Requete):
    texte = resoudre_texte(req.description, req.cve)
    if not texte:
        return {"error": "aucune description trouvee (texte vide ou JSON sans description)."}
    return predict(texte)
