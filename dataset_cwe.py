"""Etape 1 — Construire le dataset CVE -> CWE (top-N + OTHER, split temporel).

Depuis data/dataset.parquet (colonnes: description, primary_cwe, year) :
  1. filtre les CWE 'missing' et les descriptions vides
  2. garde les top-N CWE (+ bucket OTHER), le LEVIER de fiabilite
  3. split temporel (train <=2023, val 2024, test 2025)
  4. ecrit train/val/test.parquet + labels.json (id2label / label2id)

Lit le dataset CVE du projet cyber_cve et ecrit les splits CWE en local.

Usage :
    uv run python dataset_cwe.py --min-count 500
"""

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

OTHER_LABEL = "CWE-OTHER"
MISSING = "missing"


# %%
def build(
    data_path: str,
    out_dir: str,
    top_n: int = 25,
    min_count: int | None = None,
    cwe_list: list[str] | None = None,
    train_max_year: int = 2023,
    val_year: int = 2024,
    test_year: int = 2025,
    drop_other: bool = False,
) -> None:
    """Construit les splits CWE et les ecrit dans out_dir.

    :param data_path: chemin du parquet source (description, primary_cwe, year)
    :param out_dir: dossier de sortie (train/val/test.parquet + labels.json)
    :param top_n: nombre de CWE frequents gardes comme classes propres
    :param min_count: si defini, garde plutot les CWE ayant >= min_count CVE (seuil de
        support, plus fiable qu'un top-N arbitraire) ; ignore top_n
    :param train_max_year: annee max incluse dans le train
    :param val_year: annee de validation
    :param test_year: annee de test
    :param drop_other: si True, jette les CWE hors selection au lieu de les mettre dans OTHER
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) charge + nettoie : CWE connu, description non vide, annees exploitables
    lf = (
        pl.scan_parquet(data_path)
        .select("description", "primary_cwe", "year")
        .filter(pl.col("primary_cwe").is_not_null() & (pl.col("primary_cwe") != MISSING))
        .filter(pl.col("description").is_not_null() & (pl.col("description").str.len_chars() > 0))
        .filter(pl.col("year").is_in([*range(2000, test_year + 1)]))
    )
    df = lf.collect()
    print(f"[nettoyage] CVE avec CWE + description : {len(df):,}")

    # 2) selection des classes ; le reste -> OTHER (ou drop)
    #    - par seuil de support (min_count) : plus fiable, chaque classe a assez d'exemples
    #    - sinon par top-N frequents
    counts = df["primary_cwe"].value_counts(sort=True)
    if cwe_list is not None:
        top = list(cwe_list)
        present_counts = {n: c for n, c in zip(counts["primary_cwe"].to_list(), counts["count"].to_list())}
        # Une faute de frappe dans --cwe-list creerait sinon une classe VIDE (0 exemple) : le
        # modele aurait une sortie inutilisable et le macro-F1 serait plombe par un F1=0 fantome.
        absents = [c for c in top if present_counts.get(c, 0) == 0]
        if absents:
            print(f"[cwe-list] ⚠️ ignores (0 CVE dans la source, faute de frappe ?) : {', '.join(absents)}")
            top = [c for c in top if c not in absents]
            if not top:
                raise ValueError("--cwe-list : aucun des CWE demandes n'existe dans la source")
        faibles = [(c, present_counts[c]) for c in top if present_counts[c] < 500]
        if faibles:
            print("[cwe-list] ⚠️ classes a faible support (<500 CVE) : "
                  + ", ".join(f"{c} ({n})" for c, n in faibles))
        coverage = sum(present_counts[c] for c in top) / len(df)
        print(f"[cwe-list] {len(top)} classes (curated) | couverture : {coverage:.1%} des CVE")
    elif min_count is not None:
        kept = counts.filter(pl.col("count") >= min_count)
        top = kept["primary_cwe"].to_list()
        coverage = kept["count"].sum() / len(df)
        print(f"[min-count>={min_count}] {len(top)} classes | couverture : {coverage:.1%} des CVE")
    else:
        top = counts.head(top_n)["primary_cwe"].to_list()
        coverage = counts.head(top_n)["count"].sum() / len(df)
        print(f"[top-{top_n}] couverture : {coverage:.1%} des CVE")
    print(f"[classes] {top}")

    df = df.with_columns(
        pl.when(pl.col("primary_cwe").is_in(top))
        .then(pl.col("primary_cwe"))
        .otherwise(pl.lit(OTHER_LABEL))
        .alias("label_name")
    )
    if drop_other:
        df = df.filter(pl.col("label_name") != OTHER_LABEL)
        classes = sorted(top)
    else:
        classes = sorted(top) + [OTHER_LABEL]

    # 3) table de correspondance label <-> id (le modele veut des entiers)
    label2id = {name: i for i, name in enumerate(classes)}
    id2label = {i: name for name, i in label2id.items()}
    df = df.with_columns(
        pl.col("label_name").replace_strict(label2id, return_dtype=pl.Int64).alias("label"),
        pl.col("description").alias("text"),
    )

    # 4) split temporel (cf. PEP : jamais de futur dans le passe)
    splits = {
        "train": df.filter(pl.col("year") <= train_max_year),
        "val": df.filter(pl.col("year") == val_year),
        "test": df.filter(pl.col("year") == test_year),
    }
    cols = ["text", "label", "label_name", "year"]
    for name, part in splits.items():
        part.select(cols).write_parquet(out / f"{name}.parquet")
        print(f"[split] {name:5s} : {len(part):,} lignes")

    (out / "labels.json").write_text(
        json.dumps(
            {"label2id": label2id, "id2label": id2label, "num_labels": len(classes)},
            indent=2,
        )
    )
    print(f"[ok] ecrit dans {out}/ (labels.json : {len(classes)} classes)")


# %%
def _cli() -> None:
    p = argparse.ArgumentParser(description="Construit le dataset CVE->CWE (top-N + OTHER).")
    p.add_argument("--data", default="../cyber_cve/data/dataset.parquet")
    p.add_argument("--out-dir", default="data/cwe")
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--min-count", type=int, default=None,
                   help="garde les CWE ayant >= N CVE (seuil de support ; ignore --top-n)")
    p.add_argument("--cwe-list", type=str, default=None,
                   help="liste explicite de CWE a garder, séparés par des virgules (ignore top-n/min-count)")
    p.add_argument("--train-max-year", type=int, default=2023)
    p.add_argument("--val-year", type=int, default=2024)
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--drop-other", action="store_true")
    a = p.parse_args()
    build(
        a.data,
        a.out_dir,
        top_n=a.top_n,
        min_count=a.min_count,
        cwe_list=[s.strip() for s in a.cwe_list.split(",")] if a.cwe_list else None,
        train_max_year=a.train_max_year,
        val_year=a.val_year,
        test_year=a.test_year,
        drop_other=a.drop_other,
    )


if __name__ == "__main__":
    _cli()
