"""Le modele s'abstient-il ou invente-t-il quand le texte se raccourcit ?

Enjeu produit. La seule population ou il n'existe AUCUN typage est celle des findings qui ne
viennent pas d'une CVE : resultats SAST/DAST, defauts de configuration, regles proprietaires.
Ils n'ont ni CWE ni NVD dont heriter. Mais leur texte ne ressemble pas a une description NVD :
"Missing X-Frame-Options header", "SQL injection in login.jsp:42" -- court et sec, quand nos
descriptions font 304 caracteres en mediane.

Une mesure par tranche de longueur sur les CVE reelles est CONFONDUE : la longueur selectionne
un publieur autant qu'une difficulte. Ici on tronque les MEMES descriptions a differentes
longueurs : meme population, meme modele, seule la quantite de texte change.

Ce que ca teste, et ce que ca ne teste pas :
  OUI  -- le modele s'abstient-il quand l'information se raréfie, ou invente-t-il ?
  NON  -- le comportement sur du vrai vocabulaire de scanner. Une description NVD tronquee
          reste du style NVD. Seul un jeu de findings reels trancherait.

Usage :
    uv run python robustesse_texte_court.py --modele famille
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from transformers import AutoTokenizer, TFAutoModel

transformers.logging.set_verbosity_error()
MAX_LENGTH = 192

CONFIGS = {
    # le modele du contrat "tri / reporting" -- celui que l'usage produit viserait
    "famille": {"poids": "best_distilbert-base-uncased_famille_40k_20ep.weights.h5",
                "data": "data/cwe_famille", "probas": "runs/probas_distilbert-base-uncased_famille_40k_20ep.npz"},
    # le modele du contrat "remediation"
    "fin": {"poids": "best_distilbert-base-uncased_finetune_71cl_full.weights.h5",
            "data": "data/cwe71", "probas": "runs/probas_distilbert-base-uncased_finetune_71cl_full.npz"},
}


def construire(k: int) -> tf.keras.Model:
    bb = TFAutoModel.from_pretrained("distilbert-base-uncased")
    ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32)
    msk = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32)
    cls = bb(ids, attention_mask=msk).last_hidden_state[:, 0, :]
    x = tf.keras.layers.Dropout(0.3)(cls)
    return tf.keras.Model([ids, msk], tf.keras.layers.Dense(k, activation="softmax", dtype="float32")(x))


def seuil_calibre(conf, juste, cible):
    o = np.argsort(-conf)
    prec = np.cumsum(juste[o]) / np.arange(1, len(o) + 1)
    ok = np.where(prec >= cible)[0]
    return float(conf[o][ok[-1]]) if len(ok) else 1.01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", choices=list(CONFIGS), default="famille")
    ap.add_argument("--n", type=int, default=8000, help="echantillon du jeu de test")
    ap.add_argument("--cible", type=float, default=0.90)
    a = ap.parse_args()
    cfg = CONFIGS[a.modele]

    D = Path(cfg["data"])
    meta = json.loads((D / "labels.json").read_text())
    classes = [c for c, _ in sorted(meta["label2id"].items(), key=lambda kv: kv[1])]
    te = pd.read_parquet(D / "test.parquet")
    rng = np.random.default_rng(42)
    te = te.iloc[rng.choice(len(te), min(a.n, len(te)), replace=False)]

    # Le seuil du contrat vient de la CALIBRATION sur 2024, texte complet : c'est le seuil
    # qu'on deploierait. On ne le recalcule pas par longueur -- ce serait tricher.
    z = np.load(cfg["probas"], allow_pickle=True)
    cal_pred, cal_conf = z["cal_proba"].argmax(1), z["cal_proba"].max(1)
    cal_juste = np.asarray(classes)[cal_pred] == z["cal_vrais"]
    i_ind = classes.index("CWE-OTHER") if "CWE-OTHER" in classes else -1
    cal_nomme = np.ones(len(cal_pred), bool) if i_ind < 0 else (cal_pred != i_ind)
    SEUIL = seuil_calibre(cal_conf[cal_nomme], (cal_juste & cal_nomme)[cal_nomme], a.cible)
    print(f"modele '{a.modele}' | {len(classes)} classes | seuil du contrat {SEUIL:.4f}")
    print(f"echantillon : {len(te):,} CVE de 2025\n")

    modele = construire(len(classes))
    modele.load_weights(cfg["poids"])
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vrais = te.label_name.values.astype(str)

    lignes = []
    for n_car in (40, 60, 80, 120, 200, 320, None):
        textes = te.text.str[:n_car].tolist() if n_car else te.text.tolist()
        enc = tok(textes, truncation=True, padding="max_length",
                  max_length=MAX_LENGTH, return_tensors="tf")
        p = modele.predict((enc["input_ids"], enc["attention_mask"]), batch_size=64, verbose=0)
        pred, conf = p.argmax(1), p.max(1)
        nomme = np.ones(len(pred), bool) if i_ind < 0 else (pred != i_ind)
        juste = nomme & (np.asarray(classes)[pred] == vrais)
        keep = nomme & (conf >= SEUIL)
        lignes.append({
            "longueur": n_car or "complete",
            "n_car_moyen": float(np.mean([len(t) for t in textes])),
            "couverture": float(keep.mean()),
            "precision": float(juste[keep].mean()) if keep.sum() else None,
            "confiance_moyenne": float(conf.mean()),
        })
        r = lignes[-1]
        print(f"  {str(r['longueur']):>9} car.  couverture {r['couverture']:>6.1%}   "
              f"precision {r['precision'] if r['precision'] else float('nan'):>6.1%}   "
              f"confiance moy. {r['confiance_moyenne']:.3f}")

    ref, court = lignes[-1], lignes[0]
    out = {"modele": a.modele, "seuil": SEUIL, "n_echantillon": len(te),
           "mesures": lignes,
           "chute_couverture": court["couverture"] / ref["couverture"],
           "variation_precision": (court["precision"] - ref["precision"]) if court["precision"] else None,
           "note": "troncature du MEME texte : isole la longueur. Ne teste PAS le vocabulaire "
                   "d'un scanner -- une description NVD tronquee reste du style NVD.",
           "provenance": "robustesse_texte_court.py"}
    Path(f"runs/robustesse_texte_court_{a.modele}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\nA 40 caracteres, la couverture vaut {out['chute_couverture']:.0%} de celle du texte complet.")
    if court["precision"]:
        print(f"La precision, elle, varie de {out['variation_precision']:+.1%}.")
        verdict = ("le modele S'ABSTIENT quand l'information manque -- il n'invente pas"
                   if out["variation_precision"] > -0.05 else
                   "le modele CONTINUE DE REPONDRE a tort : l'abstention ne protege pas")
        print(f"-> {verdict}")


if __name__ == "__main__":
    main()
