"""Chiffre la derive de description : que devient la prediction si on lit le texte de PUBLICATION ?

Notre evaluation lit la description telle qu'elle est aujourd'hui. En production, on predirait le
jour de la publication, sur un texte parfois moins complet. L'evaluation est donc potentiellement
optimiste -- ce script mesure de combien, au lieu de l'affirmer.

Seules les CVE dont la description a change peuvent produire une prediction differente : pour les
autres, les deux textes sont identiques et l'ecart est nul par construction. On ne re-interroge
donc que celles-la, et on rapporte l'ecart sur l'ECHANTILLON COMPLET.

Petit echantillon : resultat DIRECTIONNEL, pas un contrat.

Usage :
    uv run python derive_description.py
"""
from __future__ import annotations

import os

# Inference sur processeur : le GPU est occupe par l'entrainement, et 34 textes ne le justifient pas.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf
import transformers
from transformers import AutoTokenizer, TFAutoModel

transformers.logging.set_verbosity_error()
API = "https://services.nvd.nist.gov/rest/json/cvehistory/2.0?cveId="
MAX_LENGTH = 192


def texte_t0(cve: str) -> str | None:
    """Description telle qu'ajoutee a la reception de la CVE."""
    d = json.load(urllib.request.urlopen(API + cve, timeout=45))
    for ch in sorted(d.get("cveChanges", []), key=lambda c: c["change"]["created"]):
        for x in ch["change"]["details"]:
            if x["type"] == "Description" and x["action"] == "Added":
                return x.get("newValue")
    return None


def construire(model_name: str, k: int) -> tf.keras.Model:
    bb = TFAutoModel.from_pretrained(model_name)
    ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32)
    msk = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32)
    cls = bb(ids, attention_mask=msk).last_hidden_state[:, 0, :]
    x = tf.keras.layers.Dropout(0.3)(cls)
    return tf.keras.Model([ids, msk], tf.keras.layers.Dense(k, activation="softmax", dtype="float32")(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derive", default="runs/derive_etiquette.json")
    ap.add_argument("--poids", default="best_distilbert-base-uncased_finetune_71cl.weights.h5")
    ap.add_argument("--seuil", type=float, default=None, help="seuil du contrat (defaut : lu dans runs/)")
    a = ap.parse_args()

    detail = [r for r in json.loads(Path(a.derive).read_text())["detail"]
              if "erreur" not in r and r.get("cwe_t0")]
    changees = [r["cve"] for r in detail if r["desc_changee"]]
    print(f"{len(detail)} CVE dans l'echantillon, dont {len(changees)} a description modifiee\n")

    # Texte "enrichi" = celui de notre jeu de donnees, c'est-a-dire celui sur lequel le modele
    # a reellement ete evalue. C'est la bonne reference pour mesurer l'ecart.
    src = (pl.scan_parquet("../cyber_cve/data/dataset.parquet")
             .select("cve_id", "description", "primary_cwe")
             .filter(pl.col("cve_id").is_in(changees)).collect())
    enrichi = {r["cve_id"]: r["description"] for r in src.iter_rows(named=True)}
    verite = {r["cve_id"]: r["primary_cwe"] for r in src.iter_rows(named=True)}

    paires = []
    for cve in changees:
        try:
            t0 = texte_t0(cve)
        except Exception as e:
            print(f"  {cve}: {e}"); t0 = None
        if t0 and cve in enrichi:
            paires.append((cve, t0, enrichi[cve]))
        time.sleep(6.5)
    print(f"{len(paires)} paires (texte T0, texte enrichi) recuperees\n")
    if not paires:
        return

    lab = json.loads(Path("data/cwe71/labels.json").read_text())
    classes = [c for c, _ in sorted(lab["label2id"].items(), key=lambda kv: kv[1])]
    modele = construire("distilbert-base-uncased", len(classes))
    modele.load_weights(a.poids)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    seuil = a.seuil
    if seuil is None:
        seuil = json.loads(Path("runs/seuils_distilbert-base-uncased_finetune_71cl.json").read_text())["global"]["seuil"]
    print(f"seuil du contrat applique : {seuil:.4f}\n")

    def predire(textes):
        e = tok(list(textes), truncation=True, padding="max_length",
                max_length=MAX_LENGTH, return_tensors="tf")
        p = modele([e["input_ids"], e["attention_mask"]], training=False).numpy()
        return np.array(classes)[p.argmax(1)], p.max(1)

    p0, c0 = predire([t for _, t, _ in paires])
    p1, c1 = predire([t for _, _, t in paires])

    n = len(detail)
    chg_pred = int((p0 != p1).sum())
    rep0, rep1 = c0 >= seuil, c1 >= seuil
    chg_dec = int((rep0 != rep1).sum())
    v = np.array([verite[c] for c, _, _ in paires])
    j0 = int(((p0 == v) & rep0).sum()); j1 = int(((p1 == v) & rep1).sum())

    print(f"{'':<40}{'texte T0':>12}{'texte enrichi':>16}")
    print("-" * 68)
    print(f"{'confiance moyenne':<40}{c0.mean():>12.3f}{c1.mean():>16.3f}")
    print(f"{'se prononce (au-dessus du seuil)':<40}{int(rep0.sum()):>12}{int(rep1.sum()):>16}")
    print(f"{'reponses justes':<40}{j0:>12}{j1:>16}")
    print(f"\nsur les {len(paires)} CVE a description modifiee :")
    print(f"   prediction differente        : {chg_pred}  ({chg_pred/len(paires):.0%})")
    print(f"   decision repondre/s'abstenir : {chg_dec}  ({chg_dec/len(paires):.0%})")
    print(f"\nrapporte a l'echantillon complet ({n} CVE) :")
    print(f"   la reponse change pour {chg_pred}/{n} = {chg_pred/n:.1%} des CVE")
    print(f"   ecart de justesse      : {j1 - j0:+d} CVE, soit {(j1-j0)/n:+.1%}")

    Path("runs/derive_description.json").write_text(json.dumps({
        "n_echantillon": n, "n_desc_modifiee": len(paires), "seuil": seuil,
        "confiance_moyenne_t0": float(c0.mean()), "confiance_moyenne_enrichi": float(c1.mean()),
        "n_prediction_differente": chg_pred, "n_decision_differente": chg_dec,
        "justes_t0": j0, "justes_enrichi": j1,
        "ecart_justesse_sur_echantillon": (j1 - j0) / n,
        "note": "petit echantillon : directionnel, pas un contrat",
        "detail": [{"cve": c, "pred_t0": str(x), "pred_enrichi": str(y),
                    "conf_t0": float(u), "conf_enrichi": float(w), "vrai": verite[c]}
                   for (c, _, _), x, y, u, w in zip(paires, p0, p1, c0, c1)],
        "provenance": "derive_description.py",
    }, indent=1, ensure_ascii=False))
    print("\n-> runs/derive_description.json")


if __name__ == "__main__":
    main()
