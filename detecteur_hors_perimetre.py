"""Cascade : un detecteur BINAIRE hors-perimetre en amont du classifieur CWE.

Le probleme mesure : le classifieur nomme un CWE sur 71 a 100 % des CVE dont le type est hors de
ses classes (taux `f`). C'est la contrainte qui plafonne la couverture du contrat de production.

Pourquoi un binaire dedie, alors que la classe CWE-OTHER existe deja ? Parce que dans un softmax
a N sorties, OTHER ne gagne que s'il bat les N-1 autres : c'est un seuil implicite enorme, et il
ne recoit qu'une petite part des donnees. Un binaire dedie consacre toute sa capacite a la seule
distinction "dans le perimetre ou non", et son seuil se regle independamment.

On commence PAS CHER : TF-IDF + LogReg (CPU, ~1 min). Si le signal hors-perimetre est dans le sac
de mots, la cascade est viable immediatement ; sinon on saura qu'il faut un transformer pour ca.

La cascade est ensuite evaluee sur le contrat de production, HORS ECHANTILLON : les deux seuils
(detecteur x classifieur) sont ajustes sur l'annee de validation et evalues sur l'annee de test.
Les predictions du classifieur sont relues depuis le cache runs/preds_*.npz (aucun GPU requis).

Usage :
    uv run python detecteur_hors_perimetre.py --data-dir data/cwe71 \
        --run-id distilbert-base-uncased_finetune_71cl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OTHER = "CWE-OTHER"
SEED = 42


def population(source: str, annee: int, classes_reelles: list[str]):
    """Identique a seuils_par_classe.py : meme requete, donc meme ORDRE que le cache .npz."""
    src = pl.read_parquet(source).select("description", "primary_cwe", "year")
    pop = src.filter(
        (pl.col("year") == annee)
        & pl.col("primary_cwe").is_not_null() & (pl.col("primary_cwe") != "missing")
        & pl.col("description").is_not_null() & (pl.col("description").str.len_chars() > 0)
    )
    vrais = pop["primary_cwe"].to_list()
    return pop["description"].to_list(), np.isin(vrais, classes_reelles)


def main() -> None:
    ap = argparse.ArgumentParser(description="Detecteur binaire hors-perimetre + cascade.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--source", default="../cyber_cve/data/dataset.parquet")
    ap.add_argument("--cible", type=float, default=0.90)
    ap.add_argument("--n-train", type=int, default=0, help="0 = tout le train (recommande, CPU)")
    ap.add_argument("--out-dir", default="runs")
    a = ap.parse_args()

    labels = json.loads(Path(a.data_dir, "labels.json").read_text())
    id2label = {int(k): v for k, v in labels["id2label"].items()}
    classes_reelles = [c for c in labels["label2id"] if c != OTHER]
    other_id = labels["label2id"].get(OTHER)

    # --- 1) Entrainement du detecteur binaire sur le TRAIN (<=2023) ---
    tr = pl.read_parquet(f"{a.data_dir}/train.parquet")
    y_tr = (tr["label"].to_numpy() != other_id).astype(int) if other_id is not None \
        else np.ones(len(tr), int)
    x_tr = tr["text"].to_list()
    if a.n_train and a.n_train < len(x_tr):
        idx = np.random.RandomState(SEED).choice(len(x_tr), a.n_train, replace=False)
        x_tr = [x_tr[i] for i in idx]; y_tr = y_tr[idx]
    print(f"[detecteur] train {len(x_tr):,} CVE | dans le perimetre {y_tr.mean():.1%}")

    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(x_tr)
    det = LogisticRegression(max_iter=1000, class_weight="balanced")
    det.fit(Xtr, y_tr)
    print(f"[detecteur] TF-IDF + LogReg entraine ({Xtr.shape[1]:,} features)")

    # --- 2) Score du detecteur sur les populations de calibration et de test ---
    annee_test = int(pl.read_parquet(f"{a.data_dir}/test.parquet")["year"].max())
    annee_cal = int(pl.read_parquet(f"{a.data_dir}/val.parquet")["year"].max())

    cache = Path(a.out_dir) / f"preds_{a.run_id}.npz"
    if not cache.exists():
        raise SystemExit(f"cache introuvable : {cache}\n"
                         f"-> lancer d'abord seuils_par_classe.py pour le produire.")
    z = np.load(cache, allow_pickle=True)

    donnees = {}
    for tag, annee in [("cal", annee_cal), ("test", annee_test)]:
        textes, dans = population(a.source, annee, classes_reelles)
        conf = z[f"{tag}_conf"]
        if len(textes) != len(conf):
            raise SystemExit(f"desalignement {tag}: {len(textes)} textes vs {len(conf)} predictions")
        score = det.predict_proba(vec.transform(textes))[:, 1]
        auc = roc_auc_score(dans, score)
        donnees[tag] = {"annee": annee, "dans": dans, "score": score, "conf": conf,
                        "nomme": z[f"{tag}_nomme"], "juste": z[f"{tag}_juste"]}
        print(f"[{tag}] {annee} : {len(textes):,} CVE | perimetre {dans.mean():.1%} | "
              f"AUC du detecteur {auc:.4f}")
        donnees[tag]["auc"] = float(auc)

    C, T = donnees["cal"], donnees["test"]
    if C["auc"] < 0.60:
        print("\n⚠️ AUC faible : le signal 'hors perimetre' n'est quasiment pas dans le sac de mots.")
        print("   La cascade a peu de chances d'aider en TF-IDF ; il faudrait un transformer dedie.")

    # --- 3) Cascade : ajuste les DEUX seuils sur la calibration, evalue sur le test ---
    # Baseline de reference : seuil classifieur seul, meme protocole (ajuste sur cal).
    def eval_sur(d, t_conf, t_det=0.0):
        keep = d["nomme"] & (d["conf"] >= t_conf) & (d["score"] >= t_det)
        n = int(keep.sum())
        return n, n / len(d["conf"]), (float(d["juste"][keep].mean()) if n else float("nan"))

    grille_conf = np.round(np.arange(0.50, 1.0, 0.005), 4)
    grille_det = np.round(np.arange(0.0, 0.96, 0.05), 3)

    # (a) seuil classifieur seul : le meilleur qui tient la cible sur la CALIBRATION
    meilleur_seul = None
    for tc in grille_conf:
        n, cov, prec = eval_sur(C, tc)
        if n and prec >= a.cible and (meilleur_seul is None or cov > meilleur_seul[1]):
            meilleur_seul = (tc, cov)
    # (b) cascade : idem, sur les deux seuils
    meilleur_casc = None
    for td in grille_det:
        for tc in grille_conf:
            n, cov, prec = eval_sur(C, tc, td)
            if n and prec >= a.cible and (meilleur_casc is None or cov > meilleur_casc[2]):
                meilleur_casc = (tc, td, cov)

    print(f"\n{'='*74}\nContrat >= {a.cible:.0%}, seuils ajustes sur {annee_cal}, evalues sur {annee_test}")
    print(f"{'':<26}{'prec. test':>12}{'couverture':>13}{'CVE/an':>10}")
    print("-" * 62)
    res = {}
    if meilleur_seul:
        tc, _ = meilleur_seul
        n, cov, prec = eval_sur(T, tc)
        res["classifieur_seul"] = {"seuil_conf": float(tc), "precision": prec,
                                   "couverture": cov, "n_cve": n}
        print(f"{'classifieur seul':<26}{prec:>12.4f}{cov*100:>12.1f}%{n:>10,}")
    if meilleur_casc:
        tc, td, _ = meilleur_casc
        n, cov, prec = eval_sur(T, tc, td)
        res["cascade"] = {"seuil_conf": float(tc), "seuil_detecteur": float(td),
                          "precision": prec, "couverture": cov, "n_cve": n}
        print(f"{'cascade (det. + classif.)':<26}{prec:>12.4f}{cov*100:>12.1f}%{n:>10,}")
        print(f"\nseuils retenus : classifieur {tc:.3f} | detecteur {td:.2f}")

    if "classifieur_seul" in res and "cascade" in res:
        d_cov = res["cascade"]["couverture"] - res["classifieur_seul"]["couverture"]
        d_n = res["cascade"]["n_cve"] - res["classifieur_seul"]["n_cve"]
        print(f"\n➜ Apport de la cascade : {d_cov*100:+.1f} pt de couverture ({d_n:+,} CVE/an)")
        if d_cov > 0.02:
            print("   ✅ La cascade AIDE : le detecteur apporte un signal que la confiance du")
            print("      classifieur ne contient pas.")
        elif d_cov > -0.01:
            print("   ➖ Apport NUL : le detecteur est redondant avec la confiance du classifieur.")
            print("      Le signal 'hors perimetre' qu'il capte est deja dans la proba max.")
        else:
            print("   ❌ La cascade DEGRADE (sur-ajustement des deux seuils sur la calibration).")
        for k in ("classifieur_seul", "cascade"):
            if res[k]["precision"] < a.cible:
                print(f"   ⚠️ {k} : precision {res[k]['precision']:.3f} < cible sur {annee_test} "
                      f"(derive temporelle)")

    sortie = {"run_id": a.run_id, "cible": a.cible,
              "auc_detecteur": {"cal": C["auc"], "test": T["auc"]},
              "annee_calibration": annee_cal, "annee_test": annee_test,
              "resultats": res, "provenance": "detecteur_hors_perimetre.py"}
    out = Path(a.out_dir) / f"cascade_{a.run_id}.json"
    out.write_text(json.dumps(sortie, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[ok] ecrit {out}")


if __name__ == "__main__":
    main()
