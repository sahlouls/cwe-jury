"""Un CWE predit sert-il a choisir une remediation ? Mesure de l'ACTIONNABILITE par niveau.

La douzieme piste (granularite famille) fait monter precision ET couverture. Restait a verifier
l'hypothese qui la motivait : une famille suffit-elle pour choisir un controle compensatoire ?

MITRE attache ses mitigations (Potential_Mitigations) aux faiblesses, pas aux regroupements.
Ce script mesure a quel niveau elles existent reellement. Si les niveaux generaux en sont
depourvus, la granularite famille sert au tri et au reporting -- pas a la remediation.

Mesure aussi le croisement : les familles que le modele ne predit jamais sont-elles aussi
celles que MITRE documente le moins ? (les deux trous se composeraient)

Usage :
    uv run python actionnabilite_cwe.py
"""

from __future__ import annotations

import argparse
import glob
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

CARTE = "../cyber_cve/src/cve_exploit/data/cwe_family.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default="data/cwe_catalog")
    ap.add_argument("--carte", default=CARTE)
    ap.add_argument("--granularite", default="runs/granularite_famille.json")
    a = ap.parse_args()

    root = ET.parse(glob.glob(f"{a.catalogue}/*.xml")[0]).getroot()
    NS = {"c": root.tag.split("}")[0].strip("{")}

    # --- 1. mitigations par niveau d'abstraction ---
    # DEUX mesures distinctes, et l'ecart entre elles est le resultat :
    #   - un TEXTE de mitigation existe (element Mitigation, prose libre)
    #   - une STRATEGIE NOMMEE existe (element Strategy, vocabulaire ferme MITRE)
    # La seconde est ce qu'exige une porte de verification automatisable ; la premiere ne fait
    # que documenter en langage naturel.
    total, avec, avec_strat = Counter(), Counter(), Counter()
    mit, vocabulaire = {}, Counter()
    for w in root.findall("c:Weaknesses/c:Weakness", NS):
        niv, cid = w.get("Abstraction"), "CWE-" + w.get("ID")
        mits = w.findall("c:Potential_Mitigations/c:Mitigation", NS)
        strategies = [s.text for m in mits for s in m.findall("c:Strategy", NS) if s.text]
        total[niv] += 1
        avec[niv] += bool(mits)
        avec_strat[niv] += bool(strategies)
        mit[cid] = bool(mits)
        vocabulaire.update(strategies)

    cats = root.findall("c:Categories/c:Category", NS)
    cat_avec = sum(1 for c in cats if c.findall(".//c:Mitigation", NS))

    par_niveau = [{"niveau": n, "n": total[n],
                   "avec_mitigation": avec[n], "part": avec[n] / total[n],
                   "avec_strategie": avec_strat[n], "part_strategie": avec_strat[n] / total[n]}
                  for n in ("Pillar", "Class", "Base", "Variant", "Compound") if total[n]]
    par_niveau.append({"niveau": "Category", "n": len(cats), "avec_mitigation": cat_avec,
                       "part": cat_avec / len(cats), "avec_strategie": 0, "part_strategie": 0.0})

    # --- 2. croisement avec les familles que le modele ne predit jamais ---
    carte = json.loads(Path(a.carte).read_text(encoding="utf-8"))
    gr = json.loads(Path(a.granularite).read_text())
    jamais = {d["famille"] for d in gr["detail_par_famille"] if d["n_nommees"] == 0}

    par_fam = defaultdict(lambda: [0, 0])
    for cwe, fam in carte.items():
        if cwe in mit:
            par_fam[fam][0] += 1
            par_fam[fam][1] += mit[cwe]

    familles = [{"famille": f, "n_cwe": n, "avec_mitigation": v, "part": v / n,
                 "jamais_predite": f in jamais}
                for f, (n, v) in sorted(par_fam.items(), key=lambda kv: -kv[1][0])]

    def agrege(pred):
        n = sum(f["n_cwe"] for f in familles if pred(f))
        v = sum(f["avec_mitigation"] for f in familles if pred(f))
        return {"n_cwe": n, "avec_mitigation": v, "part": v / n if n else None}

    # 'autre' est un bucket residuel sans identite MITRE : il n'a AUCUNE mitigation par
    # construction et gonflerait l'effet. On donne les deux chiffres.
    croisement = {
        "familles_jamais_predites": agrege(lambda f: f["jamais_predite"]),
        "familles_jamais_predites_hors_autre": agrege(
            lambda f: f["jamais_predite"] and f["famille"] != "autre"),
        "familles_predites": agrege(lambda f: not f["jamais_predite"]),
    }

    fichier = glob.glob(f"{a.catalogue}/*.xml")[0]
    out = {
        "source": f"catalogue MITRE COMPLET ({fichier.split('/')[-1]}) -- "
                  "et non un export de vue (1000.csv ne contient aucune Category)",
        "n_faiblesses_catalogue": sum(total.values()),
        "n_categories_catalogue": len(cats),
        "vocabulaire_strategies": dict(vocabulaire.most_common()),
        "par_niveau_abstraction": par_niveau,
        "n_faiblesses_sans_mitigation": sum(1 for v in mit.values() if not v),
        "par_famille": familles,
        "croisement_familles_jamais_predites": croisement,
        "provenance": "actionnabilite_cwe.py",
    }
    Path("runs/actionnabilite_cwe.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"Source : {out['source']}\n")
    print(f"  {'niveau':<12}{'CWE':>7}{'texte de mitigation':>22}{'STRATEGIE nommee':>20}")
    print("  " + "-" * 61)
    for r in par_niveau:
        print(f"  {r['niveau']:<12}{r['n']:>7}{r['part']:>21.0%}{r['part_strategie']:>20.0%}")
    print(f"\n  vocabulaire ferme : {len(vocabulaire)} strategies distinctes")

    print(f"\nCroisement avec les familles jamais predites :")
    for cle, v in croisement.items():
        print(f"  {cle:<38}{v['avec_mitigation']:>4}/{v['n_cwe']:<5} = {v['part']:.0%}")
    print(f"\n-> runs/actionnabilite_cwe.json")


if __name__ == "__main__":
    main()
