"""Contrat precision/couverture en conditions de PRODUCTION.

Le notebook mesure la precision sur un test *filtre* : uniquement des CVE dont le CWE fait
partie des classes du modele. En production, les CVE arrivent sans qu'on sache leur type --
et celles hors perimetre recoivent quand meme une reponse, forcement fausse.

Ce script mesure donc la courbe precision/couverture sur la POPULATION REELLE : toutes les
CVE etiquetees de l'annee de test, dans leurs proportions naturelles, sans filtrage. Une
prediction sur une CVE hors perimetre compte comme FAUSSE. Une prediction 'CWE-OTHER' compte
comme une ABSTENTION (elle ne nomme pas le type, donc n'enrichit rien).

Sortie : le seuil minimal qui atteint la precision cible, et la couverture a ce seuil.
C'est le contrat livrable : "on attribue automatiquement a >= X% de precision, sur Y% du flux".

Usage :
    uv run python contrat_precision.py \
        --data-dir data/cwe --model ehsanaghaei/SecureBERT \
        --weights best_SecureBERT_finetune_10cl.weights.h5 --run-id SecureBERT_finetune_10cl
"""

from __future__ import annotations

# ⚠️ AVANT tout import TF (compat transformers + Keras 3)
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

MAX_LENGTH = 192
OTHER = "CWE-OTHER"


def construire_modele(model_name: str, num_classes: int, from_pt: bool,
                      activation: str = "softmax"):
    """Reconstruit EXACTEMENT l'architecture du notebook (create_cwe_model)."""
    base = TFAutoModel.from_pretrained(model_name, from_pt=from_pt, return_dict=True)
    base.trainable = True
    input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")
    out = base(input_ids=input_ids, attention_mask=attention_mask)
    cls = out.last_hidden_state[:, 0, :]
    x = tf.keras.layers.Dropout(0.3, name="dropout")(cls)
    pred = tf.keras.layers.Dense(num_classes, activation=activation, dtype="float32",
                                 name="classifier")(x)
    return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=pred, name="CWE_Classifier")


def predire(model, tokenizer, textes: list[str], batch: int = 64) -> np.ndarray:
    enc = tokenizer(textes, truncation=True, padding="max_length", max_length=MAX_LENGTH,
                    return_tensors="tf")
    ds = (tf.data.Dataset
          .from_tensor_slices({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
          .batch(batch).prefetch(tf.data.AUTOTUNE))
    p = model.predict(ds, verbose=0)
    return p.logits if hasattr(p, "logits") else p


def main() -> None:
    p = argparse.ArgumentParser(description="Contrat precision/couverture en production.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--source", default="../cyber_cve/data/dataset.parquet")
    p.add_argument("--cible", type=float, default=0.90, help="precision cible du contrat")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--activation", choices=["softmax", "sigmoid"], default="softmax",
                   help="tete du modele a reconstruire ; 'sigmoid' pour les runs multi-label")
    a = p.parse_args()

    tf.keras.mixed_precision.set_global_policy("mixed_float16")  # comme a l'entrainement

    labels = json.loads(Path(a.data_dir, "labels.json").read_text())
    num_classes = labels["num_labels"]
    id2label = {int(k): v for k, v in labels["id2label"].items()}
    label2id = labels["label2id"]
    classes_reelles = [c for c in label2id if c != OTHER]
    a_bucket = OTHER in label2id
    other_id = label2id.get(OTHER)

    # --- population de production : TOUTES les CVE etiquetees de l'annee de test ---
    test = pl.read_parquet(f"{a.data_dir}/test.parquet")
    annee = int(test["year"].max())
    src = pl.read_parquet(a.source).select("description", "primary_cwe", "year")
    pop = src.filter(
        (pl.col("year") == annee)
        & pl.col("primary_cwe").is_not_null() & (pl.col("primary_cwe") != "missing")
        & pl.col("description").is_not_null() & (pl.col("description").str.len_chars() > 0)
    )
    textes = pop["description"].to_list()
    vrais = pop["primary_cwe"].to_list()
    dans_perimetre = np.array([c in classes_reelles for c in vrais])

    print(f"[population] annee {annee} : {len(textes):,} CVE etiquetees")
    print(f"[population] dans le perimetre du modele : {dans_perimetre.mean():.1%} "
          f"({int(dans_perimetre.sum()):,}) | hors perimetre : {(~dans_perimetre).mean():.1%} "
          f"({int((~dans_perimetre).sum()):,})")
    print(f"[modele] {a.model} | {num_classes} classes | bucket OTHER : {a_bucket}")

    from_pt = any(k in a.model.lower() for k in ("securebert", "cysecbert", "secbert"))
    model = construire_modele(a.model, num_classes, from_pt, a.activation)
    model.load_weights(a.weights)
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    print(f"[modele] poids charges depuis {a.weights}")

    proba = predire(model, tokenizer, textes)
    conf = proba.max(axis=-1)
    pred = proba.argmax(axis=-1)
    noms_pred = np.array([id2label[int(i)] for i in pred])

    # Nommer = donner un CWE precis. 'CWE-OTHER' est une abstention, pas un enrichissement.
    nomme = noms_pred != OTHER if a_bucket else np.ones(len(pred), bool)
    juste = (noms_pred == np.array(vrais)) & dans_perimetre

    print(f"\n[brut] le modele nomme un CWE sur {nomme.mean():.1%} de la population")
    f_hors = float((nomme & ~dans_perimetre).sum() / max(1, int((~dans_perimetre).sum())))
    print(f"[brut] taux de faux nommage sur le HORS-PERIMETRE (f) : {f_hors:.1%}")

    print(f"\n{'seuil':<8}{'couverture':>12}{'CVE traitees':>14}{'precision':>12}   contrat")
    print("-" * 62)
    lignes, seuil_ok = [], None
    for t in [0.0, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999]:
        garde = nomme & (conf >= t)
        n = int(garde.sum())
        cov = n / len(textes)
        prec = float(juste[garde].sum() / n) if n else float("nan")
        atteint = n > 0 and prec >= a.cible
        if atteint and seuil_ok is None:
            seuil_ok = (t, cov, prec, n)
        lignes.append({"seuil": t, "couverture": cov, "n": n, "precision": prec})
        print(f"{t:<8.3f}{cov*100:>11.1f}%{n:>14,}{prec:>12.4f}   "
              f"{'✅ >= cible' if atteint else '❌'}")

    # --- Seuil EXACT : plutot qu'une grille, on trie par confiance decroissante et on cherche
    # la plus grande couverture dont la precision cumulee tient encore la cible. Une grille
    # ratait le point de fonctionnement optimal (entre 0.995 et 0.999 la couverture triple).
    ordre = np.argsort(-conf[nomme])
    justes_tries = juste[nomme][ordre]
    conf_tries = conf[nomme][ordre]
    prec_cum = np.cumsum(justes_tries) / np.arange(1, len(justes_tries) + 1)

    def point(cible: float):
        ok = np.where(prec_cum >= cible)[0]
        if not len(ok):
            return None
        k = int(ok[-1])                       # le plus grand prefixe qui tient la cible
        return {"seuil": float(conf_tries[k]), "n_cve": k + 1,
                "couverture": (k + 1) / len(textes), "precision": float(prec_cum[k])}

    print("Points de fonctionnement optimaux (seuil exact, pas une grille) :")
    print(f"{'cible':<10}{'seuil':>10}{'couverture':>13}{'CVE/an':>10}{'precision':>12}")
    print("-" * 56)
    points = {}
    for cible in [0.80, 0.85, a.cible, 0.95]:
        pt = point(cible)
        points[f"{cible:.2f}"] = pt
        if pt:
            print(f"{cible:<10.0%}{pt['seuil']:>10.4f}{pt['couverture']*100:>12.1f}%"
                  f"{pt['n_cve']:>10,}{pt['precision']:>12.4f}")
        else:
            print(f"{cible:<10.0%}{'—':>10}{'inatteignable':>13}")

    print()
    contrat = point(a.cible)
    if contrat:
        print(f"🎯 CONTRAT ATTEIGNABLE : precision {contrat['precision']:.1%} (>= {a.cible:.0%}) "
              f"au seuil {contrat['seuil']:.4f}")
        print(f"   couverture {contrat['couverture']:.1%} du flux "
              f"({contrat['n_cve']:,} CVE/an attribuees automatiquement)")
        print(f"   les {(1-contrat['couverture'])*100:.1f}% restants partent en revue humaine.")
    else:
        print(f"🔴 CONTRAT INATTEIGNABLE : aucun seuil n'atteint {a.cible:.0%} de precision en")
        print(f"   conditions de production. Precision maximale observee : {prec_cum.max():.1%}")
        print(f"   Ce modele n'est pas livrable tel quel sur le flux complet (il l'est sur un")
        print(f"   flux DEJA filtre en amont).")
    seuil_ok = (contrat['seuil'], contrat['couverture'], contrat['precision'], contrat['n_cve']) \
               if contrat else None

    res = {
        "run_id": a.run_id,
        "annee_population": annee,
        "n_population": len(textes),
        "part_dans_perimetre": float(dans_perimetre.mean()),
        "taux_faux_nommage_hors_perimetre": f_hors,
        "cible_precision": a.cible,
        "precision_max_observee": float(prec_cum.max()),
        "courbe": lignes,
        "points_optimaux": points,
        "contrat": contrat,
        "provenance": "contrat_precision.py",
    }
    out = Path(a.out_dir) / f"contrat_{a.run_id}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[ok] ecrit {out}")


if __name__ == "__main__":
    main()
