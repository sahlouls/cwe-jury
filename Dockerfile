# API d'inference CVE -> CWE (FastAPI + DistilBERT TensorFlow).
# Image lourde (TensorFlow + poids ~1,3 Go) : c'est la realite d'un deploiement de modele DL.
# Pour une DEMO LIVE, lancer plutot en local (voir docs/lancer-demo.md) : plus rapide et fiable.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    TF_USE_LEGACY_KERAS=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    HF_HOME=/app/.hf

# 1) Dependances (couche cachee tant que le lock ne change pas)
COPY pyproject.toml uv.lock ./
COPY vendor/ ./vendor/
RUN uv sync --frozen --no-install-project --no-dev

# 2) Code + artefacts strictement necessaires a l'inference
COPY cwe_serve/ ./cwe_serve/
COPY data/cwe71/labels.json ./data/cwe71/labels.json
COPY runs/seuils_distilbert-base-uncased_finetune_71cl_full.json ./runs/
COPY best_distilbert-base-uncased_finetune_71cl_full.weights.h5 ./

EXPOSE 8001
CMD ["uv", "run", "--no-dev", "uvicorn", "cwe_serve.api:app", "--host", "0.0.0.0", "--port", "8001"]
