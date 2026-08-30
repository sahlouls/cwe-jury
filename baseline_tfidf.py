"""Baseline classique TF-IDF + regression logistique, sur un ou plusieurs sets de CWE.

Le notebook mesure ce baseline pour SON set de classes. Ce script fait la meme chose
*hors* notebook, sur autant de sets qu'on veut, et archive chaque resultat dans runs/
au meme format que les runs de fine-tuning -> le tableau d'ablation du notebook les
affiche a cote des transformers.

Pourquoi c'est important : un transformer coute ~75 min de GPU, ce baseline coute 20 s
de CPU. Si l'ecart est nul, le transformer n'est pas justifie -- et c'est un resultat,
pas un echec. On ne peut le dire qu'en mesurant.

Usage :
    uv run python baseline_tfidf.py                                  # data/cwe
    uv run python baseline_tfidf.py --data-dir data/cwe --data-dir /tmp/d17
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

SEED = 42


def charger(data_dir: str, split: str) -> tuple[list[str], np.ndarray]:
    df = pl.read_parquet(f"{data_dir}/{split}.parquet")
    return df["text"].to_list(), df["label"].to_numpy()


def bench(data_dir: str, n_train: int | None, out_dir: str, rapport: bool = False) -> dict:
    """Entraine TF-IDF + LogReg sur data_dir et archive les metriques dans out_dir."""
    labels = json.loads(Path(data_dir, "labels.json").read_text())
    num_classes = labels["num_labels"]
    id2label = {int(k): v for k, v in labels["id2label"].items()}

    train_texts, train_labels = charger(data_dir, "train")
    test_texts, test_labels = charger(data_dir, "test")

    # Meme sous-echantillonnage que le notebook (meme graine) : comparaison a armes egales.
    if n_train is not None and n_train < len(train_texts):
        idx = np.random.RandomState(SEED).choice(len(train_texts), size=n_train, replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_labels = train_labels[idx]

    t0 = time.time()
    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    x_train = vectorizer.fit_transform(train_texts)   # fit sur le TRAIN seul (pas de fuite)
    x_test = vectorizer.transform(test_texts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, train_labels)
    y_pred = clf.predict(x_test)
    duree = time.time() - t0

    # La taille d'entrainement fait partie de l'identifiant des qu'elle s'ecarte du defaut (40k),
    # sinon un baseline sur le jeu complet ecraserait celui de reference.
    _taille = "" if n_train == 40000 else ("_full" if not n_train else f"_{n_train // 1000}k")

    majoritaire = int(np.bincount(train_labels).argmax())
    y_maj = np.full_like(test_labels, majoritaire)
    run = {
        "run_id": f"tfidf-logreg_baseline_{num_classes}cl{_taille}",
        "config": {
            "model_name": "tfidf+logreg",
            "mode": "baseline_classique",
            "num_classes": num_classes,
            "classes": [id2label[i] for i in range(num_classes)],
            "vectorizer": "TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf, min_df=2)",
            "n_features": int(x_train.shape[1]),
            "n_train": len(train_texts),
            "n_test": len(test_texts),
            "seed": SEED,
            "data_dir": data_dir,
        },
        "test": {
            "accuracy": float(accuracy_score(test_labels, y_pred)),
            "f1_macro": float(f1_score(test_labels, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(test_labels, y_pred, average="weighted", zero_division=0)),
        },
        "baselines": {
            "majority": {
                "accuracy": float(accuracy_score(test_labels, y_maj)),
                "f1_macro": float(f1_score(test_labels, y_maj, average="macro", zero_division=0)),
            }
        },
        "train_seconds": round(duree, 1),
        "provenance": "baseline_tfidf.py",
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run['run_id']}.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")

    print(
        f"{num_classes:>3} classes | test {len(test_texts):>6,} | "
        f"acc={run['test']['accuracy']:.4f} macro-F1={run['test']['f1_macro']:.4f} | "
        f"majoritaire acc={run['baselines']['majority']['accuracy']:.3f} | {duree:.0f}s "
        f"-> {out / (run['run_id'] + '.json')}"
    )
    if rapport:
        print(classification_report(test_labels, y_pred, labels=list(range(num_classes)),
                                    target_names=[id2label[i] for i in range(num_classes)],
                                    zero_division=0))
    return run


def _cli() -> None:
    p = argparse.ArgumentParser(description="Baseline TF-IDF + LogReg (archive dans runs/).")
    p.add_argument("--data-dir", action="append", default=None,
                   help="dossier de splits (repetable pour comparer plusieurs sets de CWE)")
    p.add_argument("--n-train", type=int, default=40000,
                   help="sous-echantillon d'entrainement, comme le notebook (0 = tout)")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--report", action="store_true", help="affiche le classification_report par classe")
    a = p.parse_args()

    for d in a.data_dir or ["data/cwe"]:
        bench(d, a.n_train or None, a.out_dir, rapport=a.report)


if __name__ == "__main__":
    _cli()
