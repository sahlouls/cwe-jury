"""Seuils d'abstention PAR CLASSE, au lieu d'un seuil global.

Pourquoi : un seuil global unique traite toutes les classes pareil, alors qu'elles n'ont pas du
tout la meme fiabilite (CWE-79/89 sont quasi litterales dans le texte, CWE-20 est un fourre-tout).
Un seuil global doit donc etre cale sur les classes les PIRES -> on jette de la couverture sur les
bonnes. Un seuil par classe n'est PAS une transformation monotone du score global : il change
reellement l'ordre d'acceptation, donc il peut augmenter la couverture a precision egale.
(A l'inverse, une calibration type temperature scaling est monotone : elle rend le seuil
interpretable mais ne gagne AUCUNE couverture.)

Protocole honnete : les seuils sont ajustes sur l'annee de VALIDATION et evalues sur l'annee de
TEST. Le seuil global de comparaison est ajuste sur la meme annee, sinon la comparaison serait
biaisee en sa defaveur.

Usage :
    uv run python seuils_par_classe.py --data-dir data/cwe71 \
        --model distilbert-base-uncased \
        --weights best_distilbert-base-uncased_finetune_71cl.weights.h5 \
        --run-id distilbert-base-uncased_finetune_71cl
"""

from __future__ import annotations

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
MIN_SUPPORT = 30       # sous ce nombre d'acceptations, on n'estime pas un seuil par classe


def construire_modele(model_name: str, num_classes: int, from_pt: bool,
                      activation: str = "softmax"):
    base = TFAutoModel.from_pretrained(model_name, from_pt=from_pt, return_dict=True)
    base.trainable = True
    ii = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
    am = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")
    cls = base(input_ids=ii, attention_mask=am).last_hidden_state[:, 0, :]
    x = tf.keras.layers.Dropout(0.3, name="dropout")(cls)
    p = tf.keras.layers.Dense(num_classes, activation=activation, dtype="float32",
                              name="classifier")(x)
    return tf.keras.Model(inputs=[ii, am], outputs=p, name="CWE_Classifier")


def population(source: str, annee: int, classes_reelles: list[str]):
    """Toutes les CVE etiquetees de l'annee, proportions naturelles (= conditions de production)."""
    src = pl.read_parquet(source).select("description", "primary_cwe", "year")
    pop = src.filter(
        (pl.col("year") == annee)
        & pl.col("primary_cwe").is_not_null() & (pl.col("primary_cwe") != "missing")
        & pl.col("description").is_not_null() & (pl.col("description").str.len_chars() > 0)
    )
    vrais = pop["primary_cwe"].to_list()
    return pop["description"].to_list(), np.array(vrais), np.isin(vrais, classes_reelles)


def predire(model, tokenizer, textes, batch=64):
    enc = tokenizer(textes, truncation=True, padding="max_length", max_length=MAX_LENGTH,
                    return_tensors="tf")
    ds = (tf.data.Dataset
          .from_tensor_slices({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
          .batch(batch).prefetch(tf.data.AUTOTUNE))
    p = model.predict(ds, verbose=0)
    return p.logits if hasattr(p, "logits") else p


def seuil_global(conf, juste, nomme, cible):
    """Plus petit seuil global dont la precision cumulee tient la cible (ajuste sur ces donnees)."""
    ordre = np.argsort(-conf[nomme])
    prec = np.cumsum(juste[nomme][ordre]) / np.arange(1, int(nomme.sum()) + 1)
    ok = np.where(prec >= cible)[0]
    return float(conf[nomme][ordre][int(ok[-1])]) if len(ok) else 1.01


def main() -> None:
    ap = argparse.ArgumentParser(description="Seuils d'abstention par classe.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--source", default="../cyber_cve/data/dataset.parquet")
    ap.add_argument("--cible", type=float, default=0.90)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--activation", choices=["softmax", "sigmoid"], default="softmax",
                    help="tete du modele a reconstruire ; 'sigmoid' pour les runs multi-label")
    a = ap.parse_args()

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    labels = json.loads(Path(a.data_dir, "labels.json").read_text())
    num_classes = labels["num_labels"]
    id2label = {int(k): v for k, v in labels["id2label"].items()}
    label2id = labels["label2id"]
    classes_reelles = [c for c in label2id if c != OTHER]
    other_id = label2id.get(OTHER)

    annee_test = int(pl.read_parquet(f"{a.data_dir}/test.parquet")["year"].max())
    annee_cal = int(pl.read_parquet(f"{a.data_dir}/val.parquet")["year"].max())
    print(f"[protocole] seuils ajustes sur {annee_cal}, evalues sur {annee_test}")

    # (le modele n'est charge que si le cache est absent ou perime — voir plus bas)

    # Cache des predictions : une passe GPU suffit, les analyses suivantes sont instantanees.
    # ⚠️ Le cache est INVALIDE si les poids ont change (nouveau run, meme run_id). On y stocke donc
    # une empreinte du fichier de poids (taille + date) et on l'ignore si elle ne correspond plus —
    # sinon on servirait silencieusement les predictions d'un modele qui n'existe plus.
    st = Path(a.weights).stat()
    empreinte = f"{st.st_size}:{int(st.st_mtime)}"
    cache = Path(a.out_dir) / f"preds_{a.run_id}.npz"
    donnees = {}
    if cache.exists() and str(np.load(cache, allow_pickle=True).get(
            'empreinte_poids', np.array('')).item()) != empreinte:
        print(f"[cache] poids modifies depuis la mise en cache -> cache ignore, recalcul")
        cache.unlink()
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        for tag in ("cal", "test"):
            donnees[tag] = {k: z[f"{tag}_{k}"] for k in ("conf", "pred", "nomme", "juste")}
            donnees[tag]["annee"] = int(z[f"{tag}_annee"])
            donnees[tag]["n"] = len(donnees[tag]["conf"])
            print(f"[{tag}] {donnees[tag]['annee']} : {donnees[tag]['n']:,} CVE (cache)")
    else:
        from_pt = any(k in a.model.lower() for k in ("securebert", "cysecbert", "secbert"))
        model = construire_modele(a.model, num_classes, from_pt, a.activation)
        model.load_weights(a.weights)
        tokenizer = AutoTokenizer.from_pretrained(a.model)
        arch = {}
        for tag, annee in [("cal", annee_cal), ("test", annee_test)]:
            textes, vrais, dans = population(a.source, annee, classes_reelles)
            proba = predire(model, tokenizer, textes)
            pred = proba.argmax(-1)
            noms = np.array([id2label[int(i)] for i in pred])
            donnees[tag] = {
                "n": len(textes), "annee": annee, "conf": proba.max(-1), "pred": pred,
                "nomme": (noms != OTHER) if other_id is not None else np.ones(len(pred), bool),
                "juste": (noms == vrais) & dans,
            }
            for k in ("conf", "pred", "nomme", "juste"):
                arch[f"{tag}_{k}"] = donnees[tag][k]
            arch[f"{tag}_annee"] = annee
            print(f"[{tag}] {annee} : {len(textes):,} CVE | perimetre {dans.mean():.1%}")
        arch["empreinte_poids"] = np.array(empreinte)
        np.savez_compressed(cache, **arch)
        print(f"[cache] predictions ecrites dans {cache}")

    C, T = donnees["cal"], donnees["test"]

    # --- 1) Seuil GLOBAL ajuste sur la calibration (baseline de comparaison, meme protocole) ---
    tg = seuil_global(C["conf"], C["juste"], C["nomme"], a.cible)
    g_keep = T["nomme"] & (T["conf"] >= tg)
    g_cov = float(g_keep.mean())
    g_prec = float(T["juste"][g_keep].mean()) if g_keep.any() else float("nan")
    print(f"\n[global]  seuil {tg:.4f} (ajuste {annee_cal}) -> sur {annee_test} : "
          f"precision {g_prec:.4f} | couverture {g_cov:.1%} ({int(g_keep.sum()):,} CVE)")

    # --- 2) Seuils PAR CLASSE ajustes sur la calibration ---
    # Pour chaque classe : plus petit seuil ou SA precision tient la cible. Une classe sans assez
    # d'exemples (ou qui ne tient jamais la cible) est mise a 1.01 = jamais attribuee automatiquement.
    seuils, details = {}, []
    for cid in range(num_classes):
        nom = id2label[cid]
        if nom == OTHER:
            continue
        m = (C["pred"] == cid)
        n_tot = int(m.sum())
        if n_tot < MIN_SUPPORT:
            seuils[nom] = 1.01
            details.append({"classe": nom, "seuil": None, "raison": f"support insuffisant ({n_tot})"})
            continue
        ordre = np.argsort(-C["conf"][m])
        prec = np.cumsum(C["juste"][m][ordre]) / np.arange(1, n_tot + 1)
        ok = np.where(prec >= a.cible)[0]
        if len(ok) and int(ok[-1]) + 1 >= MIN_SUPPORT:
            k = int(ok[-1])
            seuils[nom] = float(C["conf"][m][ordre][k])
            details.append({"classe": nom, "seuil": seuils[nom], "n_cal": n_tot,
                            "accept_cal": k + 1, "prec_cal": float(prec[k])})
        else:
            seuils[nom] = 1.01
            details.append({"classe": nom, "seuil": None,
                            "raison": f"n'atteint jamais {a.cible:.0%} (prec max {prec.max():.2f})"})

    seuils_vec = np.array([seuils.get(id2label[int(i)], 1.01) for i in T["pred"]])
    p_keep = T["nomme"] & (T["conf"] >= seuils_vec)
    p_cov = float(p_keep.mean())
    p_prec = float(T["juste"][p_keep].mean()) if p_keep.any() else float("nan")
    n_actives = sum(1 for v in seuils.values() if v <= 1.0)
    print(f"[par cl.] {n_actives}/{len(seuils)} classes activables -> sur {annee_test} : "
          f"precision {p_prec:.4f} | couverture {p_cov:.1%} ({int(p_keep.sum()):,} CVE)")

    print(f"\n{'':<12}{'precision':>11}{'couverture':>13}{'CVE/an':>10}")
    print("-" * 46)
    print(f"{'global':<12}{g_prec:>11.4f}{g_cov*100:>12.1f}%{int(g_keep.sum()):>10,}")
    print(f"{'par classe':<12}{p_prec:>11.4f}{p_cov*100:>12.1f}%{int(p_keep.sum()):>10,}")
    delta = p_cov - g_cov
    print(f"\n➜ Couverture {delta*100:+.1f} pt ({(p_keep.sum()-g_keep.sum()):+,} CVE/an) "
          f"pour {p_prec-g_prec:+.4f} de precision")
    if p_prec < a.cible:
        print(f"⚠️ La precision tombe sous la cible {a.cible:.0%} sur {annee_test} : les seuils")
        print(f"   ajustes sur {annee_cal} ne transferent pas parfaitement (derive temporelle).")
        print(f"   Correctif : viser une cible un peu plus haute a l'ajustement.")

    actifs = sorted([d for d in details if d.get("seuil")], key=lambda d: d["seuil"])
    print(f"\nClasses les plus FIABLES (seuil le plus bas) :")
    for d in actifs[:8]:
        print(f"   {d['classe']:12s} seuil {d['seuil']:.4f}  ({d['accept_cal']:,} acceptees en cal.)")
    jamais = [d["classe"] for d in details if not d.get("seuil")]
    print(f"\n{len(jamais)} classes JAMAIS attribuees automatiquement : {', '.join(jamais[:14])}"
          + (" ..." if len(jamais) > 14 else ""))

    # --- 3) LA question de production : quelle cible viser a l'ajustement pour tenir 90% en reel ?
    # Le contrat mesure "in-sample" (seuil cherche ET evalue sur la meme annee) est OPTIMISTE :
    # en production on choisit le seuil AVANT de voir les donnees. On balaye donc la cible
    # d'ajustement sur l'annee de calibration et on regarde ce qu'elle donne vraiment sur le test.
    print(f"\n{'='*72}\nQuelle cible viser sur {annee_cal} pour tenir {a.cible:.0%} sur {annee_test} ?")
    print(f"{'cible cal.':<12}{'seuil':>9}{'prec. '+str(annee_test):>13}{'couverture':>13}{'CVE/an':>10}   contrat")
    print("-" * 72)
    balayage, retenu = [], None
    for cible_cal in [0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995]:
        t = seuil_global(C["conf"], C["juste"], C["nomme"], cible_cal)
        keep = T["nomme"] & (T["conf"] >= t)
        n = int(keep.sum())
        cov = float(keep.mean())
        prec = float(T["juste"][keep].mean()) if n else float("nan")
        tient = n > 0 and prec >= a.cible
        if tient and retenu is None:
            retenu = {"cible_cal": cible_cal, "seuil": t, "precision": prec,
                      "couverture": cov, "n_cve": n}
        balayage.append({"cible_cal": cible_cal, "seuil": t, "precision_test": prec,
                         "couverture_test": cov, "n_cve": n})
        print(f"{cible_cal:<12.1%}{t:>9.4f}{prec:>13.4f}{cov*100:>12.1f}%{n:>10,}   "
              f"{'✅' if tient else '❌'}")
    print()
    if retenu:
        print(f"🎯 CONTRAT REELLEMENT TENABLE : viser {retenu['cible_cal']:.1%} sur {annee_cal}")
        print(f"   -> seuil {retenu['seuil']:.4f} -> sur {annee_test} : precision "
              f"{retenu['precision']:.1%} sur {retenu['couverture']:.1%} du flux "
              f"({retenu['n_cve']:,} CVE/an)")
    else:
        print(f"🔴 Aucune cible d'ajustement sur {annee_cal} ne garantit {a.cible:.0%} sur {annee_test}.")
        print(f"   La derive temporelle est trop forte : il faut REAJUSTER les seuils en continu")
        print(f"   (sur les CVE etiquetees des N derniers mois) plutot que les figer.")

    res = {
        "run_id": a.run_id, "cible": a.cible,
        "balayage_cible_calibration": balayage,
        "contrat_hors_echantillon": retenu,
        "annee_calibration": annee_cal, "annee_test": annee_test,
        "min_support": MIN_SUPPORT,
        "global": {"seuil": tg, "precision": g_prec, "couverture": g_cov,
                   "n_cve": int(g_keep.sum())},
        "par_classe": {"precision": p_prec, "couverture": p_cov, "n_cve": int(p_keep.sum()),
                       "n_classes_activables": n_actives, "seuils": seuils},
        "details": details,
        "provenance": "seuils_par_classe.py",
    }
    out = Path(a.out_dir) / f"seuils_{a.run_id}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[ok] ecrit {out}")


if __name__ == "__main__":
    main()
