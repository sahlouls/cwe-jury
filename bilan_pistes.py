"""Bilan des pistes testees, mesure sous LES DEUX protocoles -- et l'ecart entre eux est un resultat.

Correction d'une erreur de notre part. Le premier bilan comparait les configurations au
"niveau 1" : le seuil de confiance y est cherche SUR L'ANNEE DE TEST pour atteindre exactement
90 % de precision, puis on compare les couvertures. C'est precisement le protocole que nous
denoncons par ailleurs comme fuite -- nous l'avions applique sans le voir.

Ici, chaque piste est mesuree aussi au "niveau 3" : cible fixee d'avance, seuil calibre sur
l'annee de validation, applique tel quel a l'annee de test. Les deux axes bougent alors, donc
on ne peut plus comparer un seul nombre. Le verdict devient une relation de DOMINATION :

    aide      -> meilleure que la reference sur la precision ET la couverture
    refutee   -> moins bonne sur les deux
    ambigue   -> meilleure sur un axe, moins bonne sur l'autre (aucun classement possible)

Ce critere est mecanique : il ne laisse aucune latitude de redaction.

Usage :
    uv run python bilan_pistes.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REFERENCE = "distilbert-base-uncased_finetune_71cl"

PISTES = [
    ("distilbert-base-uncased_finetune_71cl_full", "5 fois plus de donnees (218k CVE)",
     "le levier le plus classique en apprentissage profond"),
    ("distilbert-base-uncased_finetune_121cl", "Elargir le perimetre a 121 classes",
     "plus de couverture theorique"),
    ("distilbert-base-uncased_finetune_10cl", "Restreindre a 10 classes",
     "moins de classes a confondre"),
    ("SecureBERT_finetune_71cl", "Modele de domaine (SecureBERT)",
     "pre-entraine sur du texte cyber : il connait deja le jargon"),
    ("distilbert-base-uncased_sigmoidesimple_71cl", "Sorties sigmoides (abstention possible)",
     "des sigmoides peuvent toutes etre basses -- le softmax, non"),
]


def lire(run: str) -> dict | None:
    c, s = Path(f"runs/contrat_{run}.json"), Path(f"runs/seuils_{run}.json")
    if not (c.exists() and s.exists()):
        return None
    return {"niveau1": json.loads(c.read_text())["contrat"],
            "niveau3": json.loads(s.read_text())["global"]}


def justes(v: dict) -> float:
    """CVE correctement nommees par an = precision x nombre de reponses.

    C'est la quantite que l'utilisateur recoit reellement. La domination sur les deux axes est
    un critere strict mais grossier : elle classe "ambigu" un compromis qui echange 49 points de
    couverture contre 11 de precision, alors qu'il divise le service par sept.
    """
    return v["precision"] * v["n_cve"]


def verdict(ref: dict, v: dict) -> str:
    """Relation de domination sur les deux axes du contrat, au protocole honnete."""
    dp = v["precision"] - ref["precision"]
    dc = v["couverture"] - ref["couverture"]
    if dp > 0 and dc > 0:
        return "AIDE"
    if dp < 0 and dc < 0:
        return "refutee"
    return "ambigue"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=REFERENCE)
    a = ap.parse_args()

    ref = lire(a.reference)
    if ref is None:
        raise SystemExit(f"reference introuvable : {a.reference}")
    r1, r3 = ref["niveau1"], ref["niveau3"]

    lignes = []
    for run, nom, intuition in PISTES:
        d = lire(run)
        if d is None:
            print(f"  (ignoree, fichiers absents) {nom}")
            continue
        v1, v3 = d["niveau1"], d["niveau3"]
        lignes.append({
            "run": run, "piste": nom, "intuition": intuition,
            "niveau1_n_cve": v1["n_cve"], "niveau1_ecart": v1["n_cve"] - r1["n_cve"],
            "niveau3_precision": v3["precision"], "niveau3_couverture": v3["couverture"],
            "niveau3_n_cve": v3["n_cve"],
            "ecart_precision": v3["precision"] - r3["precision"],
            "ecart_couverture": v3["couverture"] - r3["couverture"],
            "niveau3_n_justes": round(justes(v3)),
            "ecart_justes": round(justes(v3) - justes(r3)),
            "verdict": verdict(r3, v3),
        })

    out = {
        "reference": {"run": a.reference, "niveau1": r1, "niveau3": r3,
                      "niveau3_n_justes": round(justes(r3))},
        "note": "niveau 1 = seuil cherche sur l'annee de test (fuite) ; "
                "niveau 3 = cible fixee d'avance, calibree sur la validation",
        "critere_verdict": "domination sur les deux axes du contrat au niveau 3",
        "pistes": lignes,
        "provenance": "bilan_pistes.py",
    }
    Path("runs/bilan_pistes.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"Reference : {a.reference}")
    print(f"   niveau 1 : {r1['precision']:.0%} de precision, {r1['n_cve']:,} CVE")
    print(f"   niveau 3 : {r3['precision']:.1%} / {r3['couverture']:.1%}, {r3['n_cve']:,} CVE\n")
    print(f"{'piste':<40}{'niv.1':>10}{'precision':>11}{'couverture':>12}"
          f"{'CVE justes/an':>15}{'verdict':>10}")
    print("-" * 98)
    for l in lignes:
        print(f"{l['piste']:<40}{l['niveau1_ecart']:>+10,}"
              f"{l['ecart_precision']:>+11.1%}{l['ecart_couverture']:>+12.1%}"
              f"{l['ecart_justes']:>+15,}{l['verdict']:>10}")
    print(f"\n(reference : {round(justes(r3)):,} CVE correctement nommees par an)")

    n_aide = sum(1 for l in lignes if l["verdict"] == "AIDE")
    print(f"\n{n_aide} piste(s) ameliorent la reference sur les DEUX axes au protocole honnete.")
    for l in lignes:
        if l["verdict"] == "AIDE":
            print(f"   -> {l['piste']} : {l['niveau3_n_cve']:,} CVE "
                  f"contre {r3['n_cve']:,} pour la reference")


if __name__ == "__main__":
    main()
