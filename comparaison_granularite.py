"""Supervision FINE + agregation, ou supervision GROSSIERE ? Comparaison a trois voies.

granularite_famille.py montrait qu'un modele a 71 classes RELU au niveau famille bat le meme
modele lu au niveau CWE. Restait la vraie question : un modele ENTRAINE sur les familles fait-il
mieux, ou moins bien, que cette relecture ?

Ce n'est pas acquis. Entrainer sur 11 familles jette l'information qui separe CWE-89 de CWE-79
a l'interieur de 'neutralization'. Le modele fin l'a apprise ; agreger ses probabilites preserve
la structure de confiance qui en resulte.

Trois configurations, TOUTES a 40 000 exemples d'entrainement, meme population de test (2025),
meme protocole hors echantillon (cible fixee d'avance, seuil calibre sur 2024) :

  A. 71 classes, cible fine        -- il faut nommer le CWE exact
  B. 71 classes, relue en famille  -- meme modele, sorties agregees      <- borne inferieure
  C. 11 familles, entrainee        -- supervision grossiere directe      <- la question

Usage :
    uv run python comparaison_granularite.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("runs")


def seuil_calibre(conf: np.ndarray, juste: np.ndarray, cible: float) -> float:
    """Plus petit seuil atteignant la cible sur l'annee de CALIBRATION (tri exact, pas de grille)."""
    o = np.argsort(-conf)
    prec = np.cumsum(juste[o]) / np.arange(1, len(o) + 1)
    ok = np.where(prec >= cible)[0]
    return float(conf[o][ok[-1]]) if len(ok) else 1.01


def contrat(p_cal, v_cal, p_te, v_te, etiquettes, i_ind, cible):
    """Contrat hors echantillon. i_ind = indice de la sortie 'pas une reponse' (-1 si aucune)."""
    def mesure(p, v):
        pred, conf = p.argmax(1), p.max(1)
        nomme = np.ones(len(p), bool) if i_ind < 0 else (pred != i_ind)
        return conf, nomme & (np.asarray(etiquettes)[pred] == v), nomme

    c1, j1, n1 = mesure(p_cal, v_cal)
    c2, j2, n2 = mesure(p_te, v_te)
    t = seuil_calibre(c1[n1], j1[n1], cible)
    keep = n2 & (c2 >= t)
    return {"seuil": t,
            "precision": float(j2[keep].mean()) if keep.sum() else float("nan"),
            "couverture": float(keep.mean()),
            "n_cve": int(keep.sum())}


def courbe(p, v, etiquettes, i_ind, n=12):
    pred, conf = p.argmax(1), p.max(1)
    nomme = np.ones(len(p), bool) if i_ind < 0 else (pred != i_ind)
    juste = nomme & (np.asarray(etiquettes)[pred] == v)
    pts = []
    for q in np.linspace(0, 0.95, n):
        t = np.quantile(conf[nomme], q)
        k = nomme & (conf >= t)
        if k.sum():
            pts.append({"precision": float(juste[k].mean()), "couverture": float(k.mean())})
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fin", default="runs/probas_distilbert-base-uncased_finetune_71cl_reference.npz")
    ap.add_argument("--famille", default="runs/probas_distilbert-base-uncased_famille_40k.npz")
    ap.add_argument("--labels-fin", default="data/cwe71/labels.json")
    ap.add_argument("--dir-famille", default="data/cwe_famille")
    ap.add_argument("--cible", type=float, default=0.90)
    a = ap.parse_args()

    zf = np.load(a.fin, allow_pickle=True)
    zg = np.load(a.famille, allow_pickle=True)

    cls_fin = [c for c, _ in sorted(json.loads(Path(a.labels_fin).read_text())["label2id"].items(),
                                    key=lambda kv: kv[1])]
    i_other = cls_fin.index("CWE-OTHER")

    # Verite terrain famille : les deux jeux sont alignes ligne a ligne (meme filtre, meme ordre).
    # On le VERIFIE plutot que de le supposer -- un decalage silencieux fausserait tout.
    fam_cal = pd.read_parquet(Path(a.dir_famille) / "val.parquet")
    fam_te = pd.read_parquet(Path(a.dir_famille) / "test.parquet")
    assert len(fam_cal) == len(zf["cal_proba"]) and len(fam_te) == len(zf["test_proba"]), \
        "desalignement entre le modele fin et le jeu familles"
    v_fam_cal = fam_cal.label_name.values.astype(str)
    v_fam_te = fam_te.label_name.values.astype(str)

    cls_fam = list(zg["classes"]) if "classes" in zg.files else sorted(set(v_fam_te))
    assert list(zg["test_vrais"]) == list(v_fam_te), "le modele famille n'est pas aligne sur le test"

    # ---------------- A. 71 classes, cible fine ----------------
    A = contrat(zf["cal_proba"], zf["cal_vrais"], zf["test_proba"], zf["test_vrais"],
                cls_fin, i_other, a.cible)

    # ---------------- B. 71 classes, relue en famille ----------------
    # On additionne les masses de probabilite des classes appartenant a une meme famille.
    carte = json.loads(Path("../cyber_cve/src/cve_exploit/data/cwe_family.json").read_text())
    col_fam = [carte.get(c) for c in cls_fin]          # None pour CWE-OTHER et les CWE hors carte
    i_ind = len(cls_fam)

    def agreger(p):
        o = np.zeros((len(p), len(cls_fam) + 1), np.float32)
        for j, f in enumerate(col_fam):
            o[:, cls_fam.index(f) if f in cls_fam else i_ind] += p[:, j]
        return o

    B = contrat(agreger(zf["cal_proba"]), v_fam_cal, agreger(zf["test_proba"]), v_fam_te,
                cls_fam + ["NON-ATTRIBUABLE"], i_ind, a.cible)

    # ---------------- C. 11 familles, entrainee ----------------
    # Aucune sortie 'indecidable' : toutes les familles sont des reponses valides.
    C = contrat(zg["cal_proba"], zg["cal_vrais"], zg["test_proba"], zg["test_vrais"],
                cls_fam, -1, a.cible)

    out = {
        "cible": a.cible,
        "n_train": 40000,
        "note": "trois configurations a budget d'entrainement identique (40k), "
                "meme population de test 2025, meme protocole hors echantillon",
        "A_71_classes_cible_fine": A,
        "B_71_classes_relue_famille": B,
        "C_11_familles_entrainee": C,
        "verdict_B_contre_C": {
            "precision": C["precision"] - B["precision"],
            "couverture": C["couverture"] - B["couverture"],
            "n_cve": C["n_cve"] - B["n_cve"],
        },
        "courbe_A": courbe(zf["test_proba"], zf["test_vrais"], cls_fin, i_other),
        "courbe_B": courbe(agreger(zf["test_proba"]), v_fam_te, cls_fam + ["NON-ATTRIBUABLE"], i_ind),
        "courbe_C": courbe(zg["test_proba"], zg["test_vrais"], cls_fam, -1),
        "provenance": "comparaison_granularite.py",
    }
    (RUNS / "comparaison_granularite.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"Trois configurations, 40 000 exemples d'entrainement chacune, test 2025\n")
    print(f"{'configuration':<38}{'precision':>11}{'couverture':>13}{'CVE/an':>10}")
    print("-" * 72)
    for nom, v in (("A. 71 classes, cible fine", A),
                   ("B. 71 classes, relue en famille", B),
                   ("C. 11 familles, entrainee", C)):
        print(f"{nom:<38}{v['precision']:>10.1%}{v['couverture']:>13.1%}{v['n_cve']:>10,}")

    d = out["verdict_B_contre_C"]
    print(f"\nC contre B : precision {d['precision']:+.1%} | couverture {d['couverture']:+.1%} | "
          f"{d['n_cve']:+,} CVE")
    gagnant = ("la supervision GROSSIERE gagne" if d["n_cve"] > 0
               else "la supervision FINE + agregation gagne")
    print(f"-> {gagnant}")


if __name__ == "__main__":
    main()
