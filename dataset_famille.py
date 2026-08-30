"""Construit le dataset CVE -> FAMILLE (Pillar MITRE), pour la douzieme piste.

Difference essentielle avec dataset_cwe.py, et c'est tout l'objet de l'experience :
  dataset_cwe.py    garde les N CWE les plus frequents, jette le reste dans CWE-OTHER
                    -> 15 % des CVE se retrouvent hors perimetre
  dataset_famille.py range TOUS les CWE dans leur famille
                    -> il n'y a plus de hors perimetre

Le decoupage temporel est IDENTIQUE (train <=2023, val 2024, test 2025), pour que la comparaison
avec les runs a 71 classes reste valide.

Usage :
    uv run python dataset_famille.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

MISSING = "missing"
AUTRE = "autre"
CARTE = "../cyber_cve/src/cve_exploit/data/cwe_family.json"


def build(data_path: str, out_dir: str, carte_path: str,
          train_max_year: int = 2023, val_year: int = 2024, test_year: int = 2025) -> None:
    carte: dict[str, str] = json.loads(Path(carte_path).read_text(encoding="utf-8"))

    df = (pl.scan_parquet(data_path)
            .select("description", "primary_cwe", "year")
            .filter(pl.col("primary_cwe") != MISSING)
            .filter(pl.col("description").is_not_null() & (pl.col("description").str.len_chars() > 0))
            .collect())

    # Tout CWE recoit une famille. Les identifiants absents de la carte sont d'anciennes
    # CATEGORIES MITRE depreciees (CWE-189, 255, 264, 310, 399...) : elles vont dans 'autre',
    # qui est une vraie classe et non un rebut hors perimetre.
    df = df.with_columns(
        pl.col("primary_cwe").replace_strict(carte, default=AUTRE).alias("label_name"))

    classes = sorted(df["label_name"].unique().to_list())
    label2id = {c: i for i, c in enumerate(classes)}
    df = df.with_columns(pl.col("label_name").replace_strict(label2id).alias("label"))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": df.filter(pl.col("year") <= train_max_year),
        "val": df.filter(pl.col("year") == val_year),
        "test": df.filter(pl.col("year") == test_year),
    }
    for nom, part in splits.items():
        (part.select(pl.col("description").alias("text"), "label", "label_name", "year")
             .write_parquet(out / f"{nom}.parquet"))
        print(f"  {nom:<6} {part.height:>8,} CVE   annees {part['year'].min()}-{part['year'].max()}")

    (out / "labels.json").write_text(json.dumps(
        {"label2id": label2id,
         "id2label": {str(i): c for c, i in label2id.items()},
         "num_labels": len(classes)}, indent=2, ensure_ascii=False))

    print(f"\n[ok] {out}/ — {len(classes)} classes : {classes}")
    print("\nrepartition du train :")
    for r in (splits["train"].group_by("label_name").agg(n=pl.len())
              .sort("n", descending=True).iter_rows(named=True)):
        print(f"    {r['label_name']:<24} {r['n']:>8,}")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Dataset CVE -> famille (Pillar MITRE).")
    p.add_argument("--data", default="../cyber_cve/data/dataset.parquet")
    p.add_argument("--out-dir", default="data/cwe_famille")
    p.add_argument("--carte", default=CARTE)
    p.add_argument("--train-max-year", type=int, default=2023)
    p.add_argument("--val-year", type=int, default=2024)
    p.add_argument("--test-year", type=int, default=2025)
    a = p.parse_args()
    build(a.data, a.out_dir, a.carte, a.train_max_year, a.val_year, a.test_year)


if __name__ == "__main__":
    _cli()
