"""La verite terrain derive-t-elle apres publication ? Mesure sur echantillon via l'API NVD.

Ce n'est PAS une question de couverture (le CWE est present des T0, fourni par le CNA) mais de
QUALITE : le CWE attribue a la publication peut etre revise ensuite par le NVD.

Enjeu operationnel : une plateforme de gestion des vulnerabilites met le CWE en cache a
l'ingestion. S'il est revise apres coup, la remediation batie dessus devient fausse et rien ne
se redeclenche.

Trois mesures :
  1. part des CVE dont le CWE est revise apres publication, et delai median
  2. la revision change-t-elle de FAMILLE, ou raffine-t-elle a l'interieur ?
     -> c'est LE chiffre : un changement de famille invalide la mitigation, pas un raffinement
  3. la description derive-t-elle, et de combien ? (c'est le texte sur lequel on predit)

Usage :
    uv run python derive_etiquette.py --n 120
"""
from __future__ import annotations

import argparse, difflib, json, re, time, urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

API = "https://services.nvd.nist.gov/rest/json/cvehistory/2.0?cveId="
CARTE = json.loads(Path("../cyber_cve/src/cve_exploit/data/cwe_family.json").read_text())


def cwe_de(v: str | None):
    m = re.search(r"CWE-\d+", v or "")
    return m.group(0) if m else None


def historique(cve: str):
    return json.load(urllib.request.urlopen(API + cve, timeout=45))


def analyser(cve: str):
    d = historique(cve)
    ev = sorted(d.get("cveChanges", []), key=lambda c: c["change"]["created"])
    t0 = cwe_t0 = desc_t0 = desc_courante = None
    revisions = []          # (date, cwe, source)
    desc_changee_le = None

    for ch in ev:
        c = ch["change"]
        quand = c["created"]
        for x in c["details"]:
            val = x.get("newValue") or ""
            if x["type"] == "CWE" and x["action"] in ("Added", "Changed"):
                w = cwe_de(val)
                if not w:
                    continue
                if cwe_t0 is None:
                    cwe_t0, t0 = w, quand
                revisions.append((quand, w, c["sourceIdentifier"]))
            if x["type"] == "Description":
                if desc_t0 is None and x["action"] == "Added":
                    desc_t0 = val
                if x["action"] == "Changed":
                    desc_courante = val
                    desc_changee_le = desc_changee_le or quand

    # Une revision compte si elle introduit un CWE DIFFERENT de celui de T0.
    posterieures = [(q, w, s) for q, w, s in revisions if w != cwe_t0]
    cwe_final = posterieures[-1][1] if posterieures else cwe_t0

    delai = None
    if posterieures and t0:
        a = datetime.fromisoformat(t0[:19]); b = datetime.fromisoformat(posterieures[0][0][:19])
        delai = (b - a).total_seconds() / 86400

    fam0, fam1 = CARTE.get(cwe_t0), CARTE.get(cwe_final)
    sim = None
    if desc_t0 and desc_courante:
        sim = difflib.SequenceMatcher(None, desc_t0, desc_courante).ratio()

    return {"cve": cve, "cwe_t0": cwe_t0, "cwe_final": cwe_final,
            "revise": bool(posterieures), "delai_jours": delai,
            "famille_t0": fam0, "famille_finale": fam1,
            "change_de_famille": bool(posterieures) and fam0 != fam1,
            "revise_par_nvd": any("nvd@nist.gov" in s for _, _, s in posterieures),
            "desc_changee": desc_changee_le is not None,
            "similarite_desc": sim, "n_evenements": len(ev)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--pause", type=float, default=6.5)   # limite NVD sans cle : 5 req / 30 s
    a = ap.parse_args()

    src = (pl.scan_parquet("../cyber_cve/data/dataset.parquet")
             .select("cve_id", "primary_cwe", "year")
             .filter((pl.col("year") == 2025) & (pl.col("primary_cwe") != "missing")).collect())
    rng = np.random.default_rng(42)
    ech = src[rng.choice(src.height, a.n, replace=False)]["cve_id"].to_list()

    res = []
    for k, cve in enumerate(ech):
        try:
            res.append(analyser(cve))
        except Exception as e:
            res.append({"cve": cve, "erreur": str(e)[:60]})
        if k % 20 == 0:
            print(f"  {k}/{a.n}", flush=True)
        time.sleep(a.pause)

    ok = [r for r in res if "erreur" not in r and r.get("cwe_t0")]
    rev = [r for r in ok if r["revise"]]
    fam = [r for r in rev if r["change_de_famille"]]
    dsc = [r for r in ok if r["desc_changee"]]
    delais = [r["delai_jours"] for r in rev if r["delai_jours"] is not None]
    sims = [r["similarite_desc"] for r in dsc if r["similarite_desc"] is not None]

    out = {
        "n_interroge": len(res), "n_exploitable": len(ok),
        "cwe_revise": {"n": len(rev), "part": len(rev) / len(ok) if ok else None,
                       "delai_median_jours": float(np.median(delais)) if delais else None,
                       "revise_par_nvd": sum(r["revise_par_nvd"] for r in rev)},
        "change_de_famille": {"n": len(fam),
                              "part_des_revisions": len(fam) / len(rev) if rev else None,
                              "part_du_total": len(fam) / len(ok) if ok else None},
        "description": {"n_changee": len(dsc), "part": len(dsc) / len(ok) if ok else None,
                        "similarite_mediane": float(np.median(sims)) if sims else None},
        "detail": res, "provenance": "derive_etiquette.py",
    }
    Path("runs/derive_etiquette.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"\n{len(ok)} CVE exploitables sur {len(res)} interrogees\n")
    print(f"  CWE revise apres publication : {len(rev):>4}  ({len(rev)/len(ok):.1%})")
    if delais:
        print(f"     delai median               : {np.median(delais):.1f} jours")
        print(f"     dont revise par le NVD     : {sum(r['revise_par_nvd'] for r in rev)}")
    print(f"  CHANGEMENT DE FAMILLE        : {len(fam):>4}  ({len(fam)/len(ok):.1%} du total, "
          f"{len(fam)/len(rev) if rev else 0:.0%} des revisions)")
    print(f"  description modifiee         : {len(dsc):>4}  ({len(dsc)/len(ok):.1%})")
    if sims:
        print(f"     similarite mediane T0/final: {np.median(sims):.3f}")


if __name__ == "__main__":
    main()
