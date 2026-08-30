"""Entrainement HIERARCHIQUE MULTI-LABEL : sigmoides + etiquettes propagees aux ancetres.

Deux defauts mesures de l'entrainement actuel, que ce script corrige :

1. **71 paires de nos classes sont imbriquees** (CWE-119 contient CWE-787, CWE-125, CWE-416...).
   Le softmax les traite comme mutuellement exclusives : quand la verite est CWE-787, la loss
   PENALISE le modele d'avoir mis de la masse sur CWE-119 — alors que CWE-119 est aussi vrai.
   On lui apprend activement une faussete.

2. **37 de nos 70 classes ne sont jamais attribuables** faute d'exemples. Propager l'etiquette
   vers ses ancetres fait profiter une classe rare du gradient de son parent : c'est du
   transfert de connaissance A L'INTERIEUR de l'espace des etiquettes — le seul levier non
   essaye sur ce point (5,45x plus de donnees ne l'a pas debloque).
   ⚠️ Pour que ce second point fonctionne, les ANCETRES doivent eux-memes etre des sorties du
   modele. Restreints a nos 71 classes, il n'y a que 80 liens et les plus rares (CWE-98,
   CWE-266...) n'ont aucun ancetre dans l'ensemble : elles ne gagneraient rien. On etend donc
   l'espace de sortie a nos classes PLUS tous leurs ancetres (~385 unites). Le surcout est
   negligeable — la tete passe de 768x71 a 768x385 — et le partage devient reel.

Formulation : sigmoides independantes + BinaryCrossentropy ponderee. Cible multi-hot = la classe
vraie ET tous ses ancetres presents dans nos classes.
Inference : la classe la PLUS PROFONDE au-dessus du seuil.

⚠️ Le risque a surveiller : les ancetres sont plus faciles (positifs plus souvent). Le modele
pourrait s'y cantonner et negliger les feuilles — bon score global, prediction exacte degradee,
soit l'inverse du but. On mesure donc explicitement le TAUX DE REPONSES EXACTES.

Usage :
    uv run python entrainement_hierarchique.py --data-dir data/cwe71 --n-train 40000
"""

from __future__ import annotations

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, TFAutoModel

import hierarchie_cwe as hc

MAX_LENGTH = 192
SEED = 42
OTHER = "CWE-OTHER"


def cibles_hierarchiques(labels_idx, classes, ancetres_dans_nos_classes):
    """Multi-hot : la classe vraie + tous ses ancetres presents dans nos classes."""
    Y = np.zeros((len(labels_idx), len(classes)), dtype=np.float32)
    for i, l in enumerate(labels_idx):
        Y[i, l] = 1.0
        for a in ancetres_dans_nos_classes[l]:
            Y[i, a] = 1.0
    return Y


def construire(model_name, num_classes, from_pt):
    base = TFAutoModel.from_pretrained(model_name, from_pt=from_pt, return_dict=True)
    base.trainable = True
    ii = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
    am = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")
    cls = base(input_ids=ii, attention_mask=am).last_hidden_state[:, 0, :]
    x = tf.keras.layers.Dropout(0.3, name="dropout")(cls)
    # SIGMOIDE, pas softmax : les sorties ne somment plus a 1, chaque classe est independante
    out = tf.keras.layers.Dense(num_classes, activation="sigmoid", dtype="float32",
                                name="classifier")(x)
    return tf.keras.Model(inputs=[ii, am], outputs=out)


def bce_ponderee(pos_weight):
    """BCE avec un poids par classe sur les positifs (Keras class_weight ne gere pas le multi-label).

    Le poids compense le desequilibre : une classe rare pese davantage quand elle est presente.
    Plafonne comme ailleurs dans le projet, pour ne pas exploser la variance du gradient.
    """
    pw = tf.constant(pos_weight, dtype=tf.float32)

    def loss(y_vrai, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        terme = -(pw * y_vrai * tf.math.log(y_pred) + (1 - y_vrai) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(terme)

    return loss


def profondeur_des_classes(classes, ancetres):
    """Nombre d'ancetres de chaque classe : sert a choisir la reponse la plus PROFONDE."""
    return np.array([len(ancetres(c)) for c in classes])


def predire(proba, profondeur, seuil):
    """La classe la plus profonde au-dessus du seuil ; -1 si aucune."""
    ok = proba >= seuil
    score = np.where(ok, profondeur[None, :] * 1000 + proba, -np.inf)
    choix = score.argmax(axis=1)
    return np.where(ok.any(axis=1), choix, -1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrainement hierarchique multi-label.")
    ap.add_argument("--data-dir", default="data/cwe71")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--catalogue", default="data/cwe_catalog")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--poids-max", type=float, default=10.0)
    ap.add_argument("--variante", choices=["hierarchique", "sigmoide-simple"],
                    default="hierarchique",
                    help="'sigmoide-simple' = SEULE l'activation/loss change vs le softmax de "
                         "reference : 71 classes, une etiquette par CVE, memes poids via "
                         "sample_weight. C'est le test a variable unique.")
    ap.add_argument("--out-dir", default="runs")
    a = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    for g in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    labels = json.loads(Path(a.data_dir, "labels.json").read_text())
    num_classes = labels["num_labels"]
    id2label = {int(k): v for k, v in labels["id2label"].items()}
    classes = [id2label[i] for i in range(num_classes)]

    parents, _ = hc.charger_graphe(a.catalogue)
    ancetres = hc.faire_ancetres(parents)

    # Espace de SORTIE etendu : nos classes + tous leurs ancetres. C'est ce qui permet a une
    # classe rare de partager le gradient de son parent (cf. defaut n°2 en tete de fichier).
    # En variante 'sigmoide-simple' on ne l'etend PAS : on isole l'effet de la sigmoide seule.
    sorties = list(classes)
    if a.variante == "hierarchique":
        for c in classes:
            for p in ancetres(c):
                if p not in sorties:
                    sorties.append(p)
    n_sorties = len(sorties)
    idx_de = {c: i for i, c in enumerate(sorties)}
    anc_nos = [[idx_de[p] for p in ancetres(c) if p in idx_de] for c in sorties]
    n_liens = sum(len(v) for v in anc_nos[:num_classes])
    print(f"[graphe] espace de sortie etendu : {num_classes} classes -> {n_sorties} unites "
          f"(nos classes + leurs ancetres)")
    print(f"         {n_liens} liens ancetre/descendant depuis nos classes")
    sans_anc = [classes[i] for i in range(num_classes) if not anc_nos[i]]
    print(f"         classes sans aucun ancetre : {len(sans_anc)} ({', '.join(sans_anc[:5])}...)")

    def charger(split):
        df = pl.read_parquet(f"{a.data_dir}/{split}.parquet")
        return df["text"].to_list(), df["label"].to_numpy()

    xtr, ytr = charger("train")
    xva, yva = charger("val")
    xte, yte = charger("test")
    if a.n_train and a.n_train < len(xtr):
        i = np.random.RandomState(SEED).choice(len(xtr), a.n_train, replace=False)
        xtr, ytr = [xtr[k] for k in i], ytr[i]
    n_val = max(3000, 150 * num_classes)
    if n_val < len(xva):
        i = np.random.RandomState(SEED).choice(len(xva), n_val, replace=False)
        xva, yva = [xva[k] for k in i], yva[i]
    print(f"[donnees] train {len(xtr):,} | val {len(xva):,} | test {len(xte):,}")

    if a.variante == "hierarchique":
        Ytr = cibles_hierarchiques(ytr, sorties, anc_nos)
        Yva = cibles_hierarchiques(yva, sorties, anc_nos)
    else:
        # une seule etiquette par CVE : exactement la cible du softmax, en one-hot
        Ytr = np.eye(n_sorties, dtype=np.float32)[ytr]
        Yva = np.eye(n_sorties, dtype=np.float32)[yva]
    print(f"[cibles] {Ytr.sum(axis=1).mean():.2f} etiquettes positives par CVE en moyenne "
          f"(1.0 = pas de propagation)")

    if a.variante == "hierarchique":
        freq = Ytr.mean(axis=0)
        pos_weight = np.clip(np.where(freq > 0, (1 - freq) / np.maximum(freq, 1e-6), 1.0),
                             1.0, a.poids_max).astype(np.float32)
        poids_ech = None
        print(f"[poids] positifs : min={pos_weight.min():.1f} max={pos_weight.max():.1f} "
              f"(plafond {a.poids_max})")
    else:
        # ⚠️ VARIABLE UNIQUE : on reprend EXACTEMENT les poids du run softmax de reference
        # (compute_class_weight 'balanced', plafonnes) et on les applique par echantillon —
        # c'est ce que fait Keras en interne pour class_weight en mono-label. Inventer un autre
        # schema de ponderation ferait varier deux choses au lieu d'une.
        from sklearn.utils.class_weight import compute_class_weight
        presentes = np.unique(ytr)
        w = np.minimum(compute_class_weight("balanced", classes=presentes, y=ytr), a.poids_max)
        table = np.ones(n_sorties, np.float32)
        for c, v in zip(presentes, w):
            table[c] = v
        poids_ech = table[ytr].astype(np.float32)
        pos_weight = np.ones(n_sorties, np.float32)
        print(f"[poids] par echantillon (identiques au run softmax) : "
              f"min={table.min():.2f} max={table.max():.1f}")

    tok = AutoTokenizer.from_pretrained(a.model)
    def enc(t):
        e = tok(t, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="tf")
        return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}

    btc = 16
    tuple_tr = (enc(xtr), Ytr) if poids_ech is None else (enc(xtr), Ytr, poids_ech)
    dtr = tf.data.Dataset.from_tensor_slices(tuple_tr).shuffle(1000).batch(btc).prefetch(tf.data.AUTOTUNE)
    dva = tf.data.Dataset.from_tensor_slices((enc(xva), Yva)).batch(btc).prefetch(tf.data.AUTOTUNE)
    dte = tf.data.Dataset.from_tensor_slices(enc(xte)).batch(64).prefetch(tf.data.AUTOTUNE)

    from_pt = any(k in a.model.lower() for k in ("securebert", "cysecbert", "secbert"))
    model = construire(a.model, n_sorties, from_pt)
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-5), loss=bce_ponderee(pos_weight))

    profondeur = profondeur_des_classes(sorties, ancetres)
    # pour la prediction EXACTE (ce qui alimente le PEP) on ne retient que nos classes
    est_nos_classes = np.array([i < num_classes for i in range(n_sorties)])
    run_id = f"{a.model.split('/')[-1]}_{a.variante.replace('-', '')}_{num_classes}cl"
    ckpt = f"best_{run_id}.weights.h5"

    class SuivreExactitude(tf.keras.callbacks.Callback):
        """Surveille le macro-F1 des reponses EXACTES — pas la BCE, qui recompenserait un modele
        qui se contente des ancetres faciles. C'est le risque identifie de cette approche."""
        def __init__(self): super().__init__(); self.best = -1.0
        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(dva, verbose=0)
            # on n'evalue que sur NOS classes : c'est la prediction exacte qui compte
            pn = np.where(est_nos_classes[None, :], p, -np.inf)
            pred = predire(pn, profondeur, 0.5)
            pred = np.where(pred < 0, pn.argmax(axis=1), pred)   # repli : argmax si rien au-dessus
            f1 = f1_score(yva, pred, average="macro", zero_division=0)
            (logs or {})["val_f1_exact"] = f1
            self.best = max(self.best, f1)
            print(f"   📈 val_f1_exact (macro-F1 sur la classe vraie) : {f1:.4f}")

    suivi = SuivreExactitude()
    cbs = [suivi,
           tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_f1_exact", mode="max",
                                              save_best_only=True, save_weights_only=True, verbose=1),
           tf.keras.callbacks.EarlyStopping(monitor="val_f1_exact", mode="max", patience=3,
                                            restore_best_weights=True, verbose=1),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=2,
                                                min_lr=1e-6, verbose=1),
           tf.keras.callbacks.CSVLogger(f"training_log_{run_id}.csv", append=False)]

    print(f"\n[entrainement] {run_id} — sigmoides + etiquettes propagees\n")
    hist = model.fit(dtr, validation_data=dva, epochs=a.epochs, callbacks=cbs, verbose=2)

    model.load_weights(ckpt)
    proba = model.predict(dte, verbose=0)
    pn = np.where(est_nos_classes[None, :], proba, -np.inf)
    pred = predire(pn, profondeur, 0.5)
    pred_f = np.where(pred < 0, pn.argmax(axis=1), pred)
    f1m = f1_score(yte, pred_f, average="macro", zero_division=0)
    acc = float((pred_f == yte).mean())
    print(f"\n=== TEST ===\n  accuracy {acc:.4f} | macro-F1 {f1m:.4f}")
    print(f"  (reference softmax 40k : accuracy 0.5974 | macro-F1 0.4671)")

    res = {"run_id": run_id,
           "config": {"model_name": a.model, "mode": a.variante,
                      "variante": a.variante,
                      "num_classes": num_classes, "n_train": len(xtr), "n_val": len(xva),
                      "n_test": len(xte), "seed": SEED, "epochs_effectuees": len(hist.history["loss"]),
                      "poids_max": a.poids_max, "liens_hierarchiques": n_liens,
                      "n_sorties": n_sorties,
                      "etiquettes_par_cve": float(Ytr.sum(axis=1).mean()), "checkpoint": ckpt},
           "validation": {"best_f1_exact": float(suivi.best)},
           "test": {"accuracy": acc, "f1_macro": float(f1m)},
           "provenance": "entrainement_hierarchique.py"}
    out = Path(a.out_dir) / f"{run_id}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(f"[ok] ecrit {out}")


if __name__ == "__main__":
    main()
