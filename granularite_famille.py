"""Douzieme piste : et si on changeait la GRANULARITE de la cible ?

Les onze pistes precedentes cherchaient a mieux predire 71 classes. Celle-ci change la question :
predire la FAMILLE (le Pillar MITRE) plutot que le CWE precis.

Ce n'est PAS la meme chose que "reduire a 10 classes", piste deja refutee :
  - reduire le perimetre  -> on garde les 10 CWE les plus frequents et on jette le reste dehors
                             (59 % des CVE se retrouvent hors perimetre)
  - changer la granularite -> les 969 CWE sont TOUS ranges dans une famille
                             (le hors-perimetre disparait par construction)

Protocole identique a seuils_par_classe.py, pour que la comparaison soit valide :
cible de precision fixee D'AVANCE, seuil calibre sur l'annee de validation, applique a l'annee
de test. Aucun regard sur l'annee de test dans le choix du seuil.

Aucun entrainement : on relit les probabilites deja sauvegardees. Consequence assumee -- le
resultat est une BORNE INFERIEURE, un modele entraine directement sur les familles devrait
faire au moins aussi bien.

Usage :
    uv run python granularite_famille.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

RUNS = Path("runs")
SOURCE_FAMILLES = "../cyber_cve/src/cve_exploit/data/cwe_family.json"
SOURCE_CVE = "../cyber_cve/data/dataset.parquet"


def charger_familles() -> dict[str, str]:
    """Carte CWE -> famille (les 10 Pillars officiels MITRE, plus 'autre')."""
    p = Path(SOURCE_FAMILLES)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # Repli : la carte est aussi derivable du jeu de donnees, qui porte les deux colonnes.
    d = (pl.scan_parquet(SOURCE_CVE).select("primary_cwe", "cwe_family").unique().collect())
    return {c: f for c, f in zip(d["primary_cwe"], d["cwe_family"]) if c != "missing"}


def seuil_calibre(conf: np.ndarray, juste: np.ndarray, cible: float) -> float:
    """Plus petit seuil atteignant la precision cible, mesure sur l'annee de CALIBRATION.

    On trie par confiance decroissante et on lit la precision cumulee : c'est exact, la ou une
    grille de seuils arrondirait. Si la cible n'est jamais atteinte, on renvoie 1.01 (jamais
    franchi) plutot que de rendre un seuil qui ne tient pas.
    """
    o = np.argsort(-conf)
    prec = np.cumsum(juste[o]) / np.arange(1, len(o) + 1)
    ok = np.where(prec >= cible)[0]
    return float(conf[o][ok[-1]]) if len(ok) else 1.01


def mesurer(proba, vrais, etiquettes, i_indecidable):
    """Prediction, confiance, et justesse pour une granularite donnee."""
    pred = proba.argmax(1)
    conf = proba.max(1)
    # Une sortie 'indecidable' (CWE-OTHER, ou famille non attribuable) n'est pas une reponse.
    nomme = pred != i_indecidable
    juste = nomme & (np.asarray(etiquettes)[pred] == vrais)
    return conf, juste, nomme


def contrat(proba_cal, vrais_cal, proba_te, vrais_te, etiquettes, i_ind, cible):
    """Contrat hors echantillon : seuil calibre sur l'annee de validation, applique au test."""
    c1, j1, n1 = mesurer(proba_cal, vrais_cal, etiquettes, i_ind)
    c2, j2, n2 = mesurer(proba_te, vrais_te, etiquettes, i_ind)
    t = seuil_calibre(c1[n1], j1[n1], cible)
    keep = n2 & (c2 >= t)
    return {
        "seuil": t,
        "precision": float(j2[keep].mean()) if keep.sum() else float("nan"),
        "couverture": float(keep.mean()),
        "n_cve": int(keep.sum()),
    }


def courbe(proba, vrais, etiquettes, i_ind, n_points=12):
    """Compromis precision/couverture accessible, pour tracer la courbe complete."""
    conf, juste, nomme = mesurer(proba, vrais, etiquettes, i_ind)
    pts = []
    for q in np.linspace(0, 0.95, n_points):
        t = np.quantile(conf[nomme], q)
        keep = nomme & (conf >= t)
        if keep.sum():
            pts.append({"seuil": float(t), "precision": float(juste[keep].mean()),
                        "couverture": float(keep.mean()), "n": int(keep.sum())})
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probas", default="runs/probas_distilbert-base-uncased_finetune_71cl_full.npz")
    ap.add_argument("--labels", default="data/cwe71/labels.json")
    ap.add_argument("--cible", type=float, default=0.90)
    a = ap.parse_args()

    FAM = charger_familles()
    familles = sorted(set(FAM.values()))
    z = np.load(a.probas, allow_pickle=True)

    # Ordre des colonnes de sortie du modele.
    l2i = json.loads(Path(a.labels).read_text())["label2id"]
    classes = [c for c, _ in sorted(l2i.items(), key=lambda kv: kv[1])]
    i_other = classes.index("CWE-OTHER")

    vrais_cal, vrais_te = z["cal_vrais"], z["test_vrais"]

    # ---------------------------------------------------------------- A. 71 classes
    ref = contrat(z["cal_proba"], vrais_cal, z["test_proba"], vrais_te,
                  classes, i_other, a.cible)
    hors_71 = float(np.mean([v not in classes for v in vrais_te]))

    # ---------------------------------------------------------------- B. familles
    # Chaque colonne de sortie est rattachee a sa famille ; les masses de probabilite s'additionnent.
    col_fam = [FAM.get(c) for c in classes]
    i_ind = len(familles)

    def agreger(p):
        out = np.zeros((len(p), len(familles) + 1), dtype=np.float32)
        for j, f in enumerate(col_fam):
            out[:, familles.index(f) if f else i_ind] += p[:, j]
        return out

    fam_cal = np.array([FAM.get(v, "autre") for v in vrais_cal])
    fam_te = np.array([FAM.get(v, "autre") for v in vrais_te])
    etiq_fam = familles + ["NON-ATTRIBUABLE"]

    p_cal, p_te = agreger(z["cal_proba"]), agreger(z["test_proba"])
    fam = contrat(p_cal, fam_cal, p_te, fam_te, etiq_fam, i_ind, a.cible)
    hors_fam = float(np.mean([FAM.get(v) is None for v in vrais_te]))

    # ---------------------------------------------------------------- detail par famille
    conf, juste, nomme = mesurer(p_te, fam_te, etiq_fam, i_ind)
    keep = nomme & (conf >= fam["seuil"])
    noms_pred = np.asarray(etiq_fam)[p_te.argmax(1)]
    detail = []
    for f in familles:
        m = keep & (noms_pred == f)
        detail.append({
            "famille": f,
            "n_cve_reelles": int((fam_te == f).sum()),
            "n_nommees": int(m.sum()),
            "precision": float(juste[m].mean()) if m.sum() else None,
        })

    out = {
        "cible": a.cible,
        "source_probabilites": a.probas,
        "note": "borne inferieure : relecture d'un modele entraine sur 71 classes, "
                "sans reentrainement sur les familles",
        "familles": familles,
        "n_familles": len(familles),
        "granularite_71_classes": {**ref, "part_hors_perimetre": hors_71},
        "granularite_famille": {**fam, "part_hors_perimetre": hors_fam},
        "ecart": {
            "precision": fam["precision"] - ref["precision"],
            "couverture": fam["couverture"] - ref["couverture"],
            "n_cve": fam["n_cve"] - ref["n_cve"],
        },
        "courbe_71_classes": courbe(z["test_proba"], vrais_te, classes, i_other),
        "courbe_famille": courbe(p_te, fam_te, etiq_fam, i_ind),
        "detail_par_famille": detail,
        "provenance": "granularite_famille.py",
    }
    chemin = RUNS / "granularite_famille.json"
    chemin.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"{len(familles)} familles (les 10 Pillars MITRE + 'autre')\n")
    print(f"{'granularite':<16}{'precision':>11}{'couverture':>13}{'CVE/an':>10}{'hors perim.':>13}")
    print("-" * 63)
    print(f"{'71 classes':<16}{ref['precision']:>10.1%}{ref['couverture']:>13.1%}"
          f"{ref['n_cve']:>10,}{hors_71:>13.1%}")
    print(f"{'11 familles':<16}{fam['precision']:>10.1%}{fam['couverture']:>13.1%}"
          f"{fam['n_cve']:>10,}{hors_fam:>13.1%}")
    print(f"\necart : precision {out['ecart']['precision']:+.1%} | "
          f"couverture {out['ecart']['couverture']:+.1%} | "
          f"CVE nommees {out['ecart']['n_cve']:+,}")
    print(f"\n-> ecrit dans {chemin}")


if __name__ == "__main__":
    main()
