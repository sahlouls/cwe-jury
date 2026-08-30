# Démo de déploiement CVE -> CWE (Bloc 5, compétence C4)

> Une petite application qui **sert le modèle DistilBERT fine-tuné** hors notebook : on colle la
> description d'une CVE, le service renvoie le **type de faille (CWE)** prédit, sa **confiance**, et
> **s'abstient** quand il n'est pas assez sûr (le "contrat" : ~81,5 % de précision sur 57,9 % du flux).
>
> Architecture (même esprit que l'app du projet PEP) :
> **API FastAPI** (`cwe_serve/`) ← modèle TensorFlow - **Front React/Vite/Tailwind** (`frontend/`).

## Option A - Local (recommandé pour la démo live)

Le plus rapide et le plus fiable devant le jury. Deux terminaux :

```bash
# 1) l'API (charge le modèle au 1er appel, quelques secondes)
uv run uvicorn cwe_serve.api:app --port 8001         # http://127.0.0.1:8001  (docs: /docs)

# 2) le front
cd frontend && npm install && npm run dev             # http://localhost:5173
```

Puis ouvrir le front, coller une description (ou cliquer un exemple) -> **Classer**.

Test API en ligne de commande :

```bash
curl -s -X POST http://127.0.0.1:8001/predict -H 'Content-Type: application/json' \
  -d '{"description":"Cross-site scripting (XSS) in the login page via the username parameter."}'
# -> {"cwe":"CWE-79","confidence":0.99,"abstain":false,"threshold":0.964,"top":[...]}
```

## Option B - Docker (le livrable "déploiement")

Une commande démarre api + front. L'image API embarque TensorFlow + les poids (~1,3 Go) -> **build long**
(à réserver à la démonstration de la mise en production, pas au live).

```bash
docker compose up --build          # front: http://localhost:5174  |  api: http://localhost:8001
docker compose down
```

## Ce que la démo illustre

- **C4 - déploiement / MLOps** : le modèle est servi par une API versionnable, conteneurisable, appelée
  par une interface web - pas seulement un notebook.
- **Le contrat d'abstention** : coller une fiche claire (XSS, injection SQL...) -> CWE confiant ; coller une
  fiche vague -> le modèle **répond "pas assez de confiance pour trancher"** au lieu d'inventer. C'est le
  message central du projet : *savoir dire "je ne sais pas".*

## Détails techniques

- `cwe_serve/predict.py` : reconstruit l'architecture d'entraînement (DistilBERT -> `[CLS]` -> Dropout ->
  Dense softmax), recharge les poids `71cl_full`, tokenise (`MAX_LENGTH=192`), applique le **seuil calibré**
  (0,964, cible de précision 90 %) lu depuis `runs/seuils_...71cl_full.json` - rien n'est codé en dur.
- `cwe_serve/api.py` : FastAPI, `GET /health` + `POST /predict`.
- `frontend/` : React 19 + Vite + Tailwind v4 (codestyle barkahub, comme le front PEP).
