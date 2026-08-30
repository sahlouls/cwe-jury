"""Douzieme piste, volet 2 : ENTRAINER un modele sur les familles, au lieu de relire un modele fin.

granularite_famille.py agrege les sorties d'un modele entraine sur 71 classes. C'est une borne
inferieure. Ce script tranche la vraie question :

    la supervision GROSSIERE (11 familles) bat-elle la supervision FINE (71 classes) suivie
    d'une agregation, quand on ne veut de toute facon qu'une famille en sortie ?

Ce n'est pas acquis d'avance. Entrainer sur 11 familles JETTE l'information qui separe CWE-89 de
CWE-79 a l'interieur de 'neutralization'. Le modele fin, lui, l'a apprise, et l'agregation de ses
probabilites preserve la structure de confiance qui en decoule. Les deux issues sont instructives.

La configuration reproduit a l'identique celle du notebook (meme backbone, meme budget, meme graine,
memes patiences) : seule la CIBLE change. Test a variable unique.

Usage :
    uv run python entrainement_famille.py                       # entraine sur les familles
    uv run python entrainement_famille.py --poids X.weights.h5 \
        --data-dir data/cwe71                                   # inference seule (baseline 71 cl.)
"""

from __future__ import annotations

import os

# ⚠️ AVANT tout import TF : les modeles TF de transformers sont ecrits pour Keras 2.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, TFAutoModel

MAX_LENGTH = 192
SEED = 42


def construire(model_name: str, num_classes: int, from_pt: bool) -> tf.keras.Model:
    """Architecture identique au notebook : [CLS] -> Dropout -> Dense softmax float32."""
    backbone = TFAutoModel.from_pretrained(model_name, from_pt=from_pt)
    ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
    mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")
    cls = backbone(ids, attention_mask=mask).last_hidden_state[:, 0, :]
    x = tf.keras.layers.Dropout(0.3)(cls)
    # float32 impose : en precision mixte le softmax deborde et la perte devient NaN.
    out = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model([ids, mask], out)


def encoder(tokenizer, textes):
    return tokenizer(list(textes), truncation=True,
                     padding="max_length",   # ⚠️ PAS padding=True : l'entree est de forme FIXE
                     max_length=MAX_LENGTH, return_tensors="np")


def jeu(enc, labels, batch, melange=False):
    d = tf.data.Dataset.from_tensor_slices(
        ((enc["input_ids"], enc["attention_mask"]), labels))
    if melange:
        d = d.shuffle(8192, seed=SEED)
    return d.batch(batch).prefetch(tf.data.AUTOTUNE)


class F1Macro(tf.keras.callbacks.Callback):
    """macro-F1 sur la validation, a la fin de chaque epoque : c'est la metrique de pilotage."""

    def __init__(self, ds, y):
        super().__init__()
        self.ds, self.y = ds, y

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.ds, verbose=0).argmax(1)
        logs["val_f1_macro"] = f1_score(self.y, p, average="macro", zero_division=0)
        print(f"   val_f1_macro = {logs['val_f1_macro']:.4f}")


def probabilites(model, tokenizer, textes, batch=64):
    enc = encoder(tokenizer, textes)
    return model.predict((enc["input_ids"], enc["attention_mask"]), batch_size=batch, verbose=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/cwe_famille")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--poids-max", type=float, default=10.0)
    ap.add_argument("--poids", default=None,
                    help="chemin de poids existants : saute l'entrainement, fait l'inference seule")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    D = Path(a.data_dir)
    meta = json.loads((D / "labels.json").read_text())
    K = meta["num_labels"]
    classes = [c for c, _ in sorted(meta["label2id"].items(), key=lambda kv: kv[1])]
    tag = a.tag or f"{a.model.split('/')[-1]}_{D.name}_{a.n_train // 1000}k"

    tr = pd.read_parquet(D / "train.parquet")
    va = pd.read_parquet(D / "val.parquet")
    te = pd.read_parquet(D / "test.parquet")
    print(f"{K} classes | train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    modele = construire(a.model, K, from_pt=False)

    if a.poids:
        modele.load_weights(a.poids)
        print(f"poids charges depuis {a.poids} — inference seule")
    else:
        # Sous-echantillonnage UNIFORME, a l'identique du notebook (sous_echantillon()) :
        # un tirage stratifie changerait une seconde variable et invaliderait la comparaison.
        # Consequence assumee et connue : la distribution naturelle est preservee, donc les
        # classes rares restent rares dans l'echantillon.
        def sous_echantillon(df, n):
            if n is None or n >= len(df):
                return df
            idx = np.random.RandomState(SEED).choice(len(df), size=n, replace=False)
            return df.iloc[idx]

        tr = sous_echantillon(tr, a.n_train)
        # Validation : meme regle que le notebook, max(3000, 150 * nb_classes).
        va_s = sous_echantillon(va, min(len(va), max(3000, 150 * K)))
        print(f"echantillons retenus : train {len(tr):,} | val {len(va_s):,}")

        # Poids de classe plafonnes : le rapport brut rendrait la mise a jour instable.
        pres = np.unique(tr.label.values)
        w = compute_class_weight("balanced", classes=pres, y=tr.label.values)
        poids = {int(c): float(min(v, a.poids_max)) for c, v in zip(pres, w)}
        print(f"poids de classe : min {min(poids.values()):.2f} | max {max(poids.values()):.2f}")

        d_tr = jeu(encoder(tokenizer, tr.text), tr.label.values, a.batch, melange=True)
        d_va = jeu(encoder(tokenizer, va_s.text), va_s.label.values, a.batch)

        modele.compile(optimizer=tf.keras.optimizers.Adam(a.lr),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        ckpt = f"best_{tag}.weights.h5"
        modele.fit(
            d_tr, validation_data=d_va, epochs=a.epochs, class_weight=poids,
            callbacks=[
                F1Macro(d_va, va_s.label.values),
                tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_f1_macro", mode="max",
                                                   save_best_only=True, save_weights_only=True),
                tf.keras.callbacks.EarlyStopping(monitor="val_f1_macro", mode="max", patience=3,
                                                 restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_f1_macro", mode="max",
                                                     factor=0.5, patience=2),
                tf.keras.callbacks.CSVLogger(f"training_log_{tag}.csv"),
            ], verbose=1)
        modele.load_weights(ckpt)

    # --- probabilites sur les annees COMPLETES de validation et de test ---
    # (et non sur l'echantillon d'entrainement : c'est ce qui permet de calibrer puis mesurer)
    print("\ninference sur la validation complete...")
    p_val = probabilites(modele, tokenizer, va.text)
    print("inference sur le test complet...")
    p_te = probabilites(modele, tokenizer, te.text)

    np.savez_compressed(f"runs/probas_{tag}.npz",
                        cal_proba=p_val.astype(np.float32), cal_vrais=va.label_name.values.astype(str),
                        test_proba=p_te.astype(np.float32), test_vrais=te.label_name.values.astype(str),
                        classes=np.array(classes))

    pred = p_te.argmax(1)
    res = {
        "run_id": tag, "data_dir": str(D), "num_classes": K, "n_train": int(len(tr)),
        "test": {
            "accuracy": float((pred == te.label.values).mean()),
            "f1_macro": float(f1_score(te.label.values, pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(te.label.values, pred, average="weighted", zero_division=0)),
        },
        "provenance": "entrainement_famille.py",
    }
    Path(f"runs/{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n{tag} | exactitude {res['test']['accuracy']:.4f} | "
          f"macro-F1 {res['test']['f1_macro']:.4f}")
    print(f"-> runs/{tag}.json et runs/probas_{tag}.npz")


if __name__ == "__main__":
    main()
