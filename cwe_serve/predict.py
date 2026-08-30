"""predict.py -- inference CVE -> CWE hors notebook (chargement paresseux du modele).

Reconstruit l'architecture exacte de l'entrainement (DistilBERT -> [CLS] -> Dropout -> Dense softmax),
recharge les poids, tokenise la description et renvoie :
  - le CWE predit + la confiance (proba max)
  - un drapeau d'abstention : si la confiance < seuil calibre (contrat 90 %), on ne tranche pas
  - le top-3 pour la transparence

Le seuil et la liste des classes sont LUS depuis les artefacts (jamais codes en dur).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

# tf-keras (les modeles TF de transformers sont incompatibles Keras 3) -- avant import tensorflow
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")     # silence les logs d'init TF
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 192
ROOT = Path(__file__).resolve().parent.parent          # racine du repo cwe-finetuning
WEIGHTS = ROOT / "best_distilbert-base-uncased_finetune_71cl_full.weights.h5"
LABELS = ROOT / "data" / "cwe71" / "labels.json"
SEUILS = ROOT / "runs" / "seuils_distilbert-base-uncased_finetune_71cl_full.json"
CIBLE = 0.9                                             # contrat retenu : precision cible 90 %


def _seuil() -> float:
    """Lit le seuil calibre pour la cible de precision retenue (contrat 90 %)."""
    data = json.loads(SEUILS.read_text())
    for row in data["balayage_cible_calibration"]:
        if abs(row["cible_cal"] - CIBLE) < 1e-9:
            return float(row["seuil"])
    raise ValueError("cible de calibration introuvable dans les seuils")


@lru_cache(maxsize=1)
def _artefacts():
    """Charge (une seule fois) le modele, le tokenizer, les classes et le seuil."""
    import tensorflow as tf
    from transformers import AutoTokenizer, TFAutoModel

    meta = json.loads(LABELS.read_text())
    id2label = {int(i): c for c, i in meta["label2id"].items()}
    n = meta["num_labels"]

    # architecture identique a l'entrainement (indispensable pour load_weights)
    backbone = TFAutoModel.from_pretrained(MODEL_NAME)
    ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
    mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")
    cls = backbone(ids, attention_mask=mask).last_hidden_state[:, 0, :]
    x = tf.keras.layers.Dropout(0.3)(cls)
    out = tf.keras.layers.Dense(n, activation="softmax", dtype="float32")(x)
    model = tf.keras.Model([ids, mask], out)
    model.load_weights(str(WEIGHTS))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer, id2label, _seuil()


def description_from_cve(cve: dict) -> str:
    """Extrait la description depuis un JSON de CVE (plusieurs formats supportes).

    - feed threat-intel : arbre rich-text `i18n.<lang>.description` (on collecte les champs 'text')
    - NVD API 2.0 / CVE 5.x : liste `descriptions[]` ou `containers.cna.descriptions[]` ({lang, value})
    On privilegie l'anglais. Retourne "" si rien trouve.
    """
    # 1) format du feed : i18n.en.description (rich-text arborescent)
    i18n = cve.get("i18n") if isinstance(cve, dict) else None
    if isinstance(i18n, dict):
        node = i18n.get("en") or next(iter(i18n.values()), None)
        if isinstance(node, dict):
            textes: list[str] = []

            def walk(o):
                if isinstance(o, dict):
                    if isinstance(o.get("text"), str):
                        textes.append(o["text"])
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for x in o:
                        walk(x)

            walk(node.get("description", {}))
            if textes:
                return " ".join(textes).strip()

    # 2) formats NVD : descriptions[] a la racine ou sous containers.cna
    listes = []
    if isinstance(cve.get("descriptions"), list):
        listes = cve["descriptions"]
    else:
        cna = cve.get("containers", {}).get("cna", {}) if isinstance(cve.get("containers"), dict) else {}
        if isinstance(cna.get("descriptions"), list):
            listes = cna["descriptions"]
    if listes:
        en = [d.get("value", "") for d in listes if isinstance(d, dict) and d.get("lang", "").startswith("en")]
        val = (en[0] if en else listes[0].get("value", "")).strip()
        if val:
            return val
    return ""


def resoudre_texte(description: str | None, cve: dict | None) -> str:
    """Renvoie le texte a classer : la description fournie, sinon extraite du JSON de CVE."""
    if description and description.strip():
        return description.strip()
    if isinstance(cve, dict):
        return description_from_cve(cve)
    return ""


def predict(description: str, top_k: int = 3) -> dict:
    """Classe une description de CVE dans son type de faille (CWE).

    Retour : {cwe, confidence, abstain, threshold, top: [{cwe, p}, ...]} ; abstain=True si la
    confiance est sous le seuil du contrat -> on affiche "pas assez de confiance pour trancher".
    """
    import numpy as np

    text = (description or "").strip()
    if len(text) < 10:
        return {"error": "description trop courte pour juger (>= 10 caracteres)."}

    model, tokenizer, id2label, seuil = _artefacts()
    enc = tokenizer([text], truncation=True, padding="max_length",
                    max_length=MAX_LENGTH, return_tensors="np")
    proba = model.predict([enc["input_ids"], enc["attention_mask"]], verbose=0)[0]

    order = np.argsort(-proba)[:top_k]
    top = [{"cwe": id2label[int(i)], "p": round(float(proba[i]), 4)} for i in order]
    best = top[0]
    return {
        "cwe": best["cwe"],
        "confidence": best["p"],
        "abstain": best["p"] < seuil,       # le contrat : sous le seuil, on ne tranche pas
        "threshold": round(seuil, 4),
        "top": top,
    }
