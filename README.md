# CVE -> CWE - Classifieur de type de faille (Alyra, Bloc 5)

Classe une **CVE** dans son **type de faille (CWE)** a partir de sa description, par **transfer learning /
fine-tuning** d'un transformer pre-entraine (DistilBERT, SecureBERT) en TensorFlow/Keras.

> Depot **jury** (version epuree) : les notebooks, le code, le front et la partie MLOps (Docker).
> Les **poids et les gros jeux de donnees ne sont pas inclus** (voir plus bas). Les notebooks sont livres
> **pre-executes** (figures et chiffres visibles a l'ouverture). Le notebook "argument" est **re-jouable
> tel quel** : il ne re-entraine rien, il relit les resultats pre-calcules de `runs/*.json`.

## Contenu

```
llm_fine_tuning_cwe.ipynb    le livrable "argument" (pourquoi ces choix, resultats, limites) - re-jouable
07_cwe_finetuning.ipynb      la "preuve" : le code d'entrainement execute sur GPU (sorties incluses)
cwe_serve/                   inference hors notebook : predict.py (charge le modele) + api.py (FastAPI)
frontend/                    application React + Vite + Tailwind (coller une CVE -> CWE + confiance)
Dockerfile, docker-compose.yml   deploiement : api (:8001) + front (:5174)
*.py                         pipeline & analyses : dataset_cwe, entrainement_famille, baseline_tfidf,
                             seuils_par_classe, hierarchie_cwe, comparaison_granularite, ...
vendor/cuda-preload/         correctif de chargement des bibliotheques CUDA (GPU 6 Go)
runs/*.json                  resultats pre-calcules (lus par le notebook argument)
data/cwe71/labels.json       mapping des 71 classes (requis a l'inference)
docs/                        lancer-demo.md (guide operationnel)
RAPPORT.md                   le rapport detaille
```

## Prerequis

- [uv](https://docs.astral.sh/uv/), Python 3.12. GPU NVIDIA optionnel (l'inference marche aussi sur CPU).

```bash
uv sync                       # installe tensorflow, transformers, fastapi, etc.
```

## Lancer le notebook

```bash
uv run jupyter lab            # ouvrir llm_fine_tuning_cwe.ipynb
```

Le notebook "argument" se re-execute **tel quel** (il lit `runs/*.json`). Le `07_...ipynb` (entrainement)
demande un GPU + les donnees ; il est livre **pre-execute** pour lecture.

## Lancer l'application (demo de deploiement, C4)

Deux terminaux (recommande pour la demo live) :

```bash
uv run uvicorn cwe_serve.api:app --port 8001      # API (charge le modele au 1er appel)
cd frontend && npm install && npm run dev          # front -> http://localhost:5173
```

Ou tout en Docker : `docker compose up --build` (front :5174, api :8001).

> L'API a besoin des **poids** (non inclus, voir ci-dessous). Sans eux, le notebook argument reste
> pleinement demontrable via `runs/*.json`.

## Non inclus (a fournir pour tout re-executer)

- **`best_distilbert-base-uncased_finetune_71cl_full.weights.h5`** (~1,3 Go) - les poids du modele
  livre, requis par `cwe_serve` et la demo finale du notebook. A regenerer via `entrainement_famille.py`
  (ou a recuperer aupres de l'auteur).
- **`data/cwe71/{train,val,test}.parquet`** - les splits (issus du projet PEP voisin).
- **`runs/*.npz`** - probabilites brutes (volumineuses) ; les `runs/*.json` (inclus) suffisent au notebook.

## Resultats (resume)

macro-F1 (test 2025, 71 classes) : gele **0,170** < TF-IDF **0,424** < fine-tune **0,467**. Message central :
l'exactitude en laboratoire ne predit pas la valeur en production (config 10 classes : 96,9 % labo ->
39,8 % reel). Reponse honnete = un **contrat** (81,5 % de precision sur 57,9 % du flux ; famille 89,2 % /
86,6 %). Details : `RAPPORT.md`.
