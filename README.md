# CVE -> CWE - Fine-tuning de transformer (Alyra, sujet 2)

Notebook TensorFlow/Keras qui classe une **CVE** dans son **type de faille (CWE)** par
**fine-tuning** d'un transformer pré-entraîné (DistilBERT -> SecureBERT). Projet **isolé** (env DL
lourd) qui **lit le dataset source du projet `cyber_cve`** (`../cyber_cve/data/dataset.parquet`).

## Prérequis
- [uv](https://docs.astral.sh/uv/), Python 3.12, GPU NVIDIA optionnel (voir [GPU](#gpu)).
- Le dataset source : `../cyber_cve/data/dataset.parquet` (projet voisin `cyber_cve`).

## Installation
```bash
uv sync
```
`uv` crée le `.venv`, installe les dépendances (dont `tensorflow[and-cuda]` ~4 Go) et le paquet
local `cuda-preload`.

Pour **SecureBERT** (poids PyTorch -> conversion `from_pt=True`), ajouter torch (CPU suffit) :
```bash
uv add torch --index https://download.pytorch.org/whl/cpu
```

## Préparer les données puis (re)générer le notebook
```bash
# Config A : 10 CWE critiques DISTINCTES, sans OTHER (haute précision, couverture 40,5 %)
DIX="CWE-79,CWE-89,CWE-78,CWE-22,CWE-352,CWE-434,CWE-787,CWE-416,CWE-476,CWE-862"
uv run python dataset_cwe.py --cwe-list "$DIX" --drop-other --out-dir data/cwe
# Config B : 71 classes = 70 CWE fréquents + CWE-OTHER (couverture 89,8 %)
uv run python dataset_cwe.py --min-count 500 --out-dir data/cwe71

uv run python baseline_tfidf.py --data-dir data/cwe --data-dir data/cwe71   # baselines CPU -> runs/
uv run python build_cwe_nb.py                                   # écrit 07_cwe_finetuning.ipynb
```
Le set utilisé est choisi par la variable `DATA_DIR` de la cellule *Paramètres* du notebook, avec
`MODEL_NAME` et `FREEZE_BASE` : ce sont **les trois variables de l'ablation**. Tout le reste
(nombre de classes, taille de validation, époques, patience, plafond des poids de classe, nom du
checkpoint) en est **dérivé** - rien n'est codé en dur.

Exécution non interactive (produit les sorties dans le `.ipynb`) :
```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 07_cwe_finetuning.ipynb
```

## ⚠️ L'usage visé a été vérifié - et réfuté

Le CWE devait alimenter le **PEP**, un modèle d'exploitabilité développé par ailleurs. Vérification faite sur ce modèle, en fin de projet :

| Variante du PEP | ROC | PR |
|---|---|---|
| avec `primary_cwe` - production actuelle | 0.9209 | 0.8369 |
| **sans CWE du tout** | **0.9251** | **0.8492** |

**Lui retirer le CWE l'améliore** - et c'est avec le **vrai** CWE du NVD, information parfaite. Le PEP lit déjà la description complète (83,5 % de son poids) ; notre classifieur prend cette même description en entrée et n'ajoute donc rien. Aucun classifieur ne peut servir cet usage.

Ça ne condamne pas le typage CWE en soi - les usages humains (triage, reporting, filtrage) n'ont jamais été cadrés. Ça condamne **la justification retenue**. Détail dans [`RAPPORT.md`](RAPPORT.md) §8bis.

**Livrable inattendu** : retirer `primary_cwe` du PEP gagne +0.012 de PR et 121 dimensions.

---

## Résultats - l'apport du transformer dépend de la **difficulté** de la tâche

Chaque ligne compare le transformer à un **baseline TF-IDF + LogReg** entraîné sur les mêmes splits, le même sous-échantillon de 40 000 et la même graine. Test = les CVE de 2025, jamais vues.

| Set de classes | Couverture | TF-IDF + LogReg (~20 s CPU) | Transformer fine-tuné (GPU) | Gain du transformer |
|---|---|---|---|---|
| **10 CWE distinctes** (test 17 747) | 40,5 % | acc 0.9651 - **F1 0.9391** | acc 0.9691 - F1 0.9468 (DistilBERT) | +0.008 F1 -> **+0,8 % relatif** |
| **17 CWE critiques** (test 21 951) | 59,2 % | acc 0.8807 - **F1 0.7909** | acc 0.8940 - F1 0.8050 (SecureBERT) 🟡 | +0.014 F1 -> **+1,8 % relatif** |
| **71 classes** (test 43 229) | 89,8 % | acc 0.5528 - **F1 0.4243** | acc 0.5974 - F1 0.4671 (DistilBERT) | +0.042 F1 -> **+9,8 % relatif** |
| **71 classes** - modèle domaine | 89,8 % | acc 0.5528 - **F1 0.4243** | acc **0.6097** - F1 **0.4825** (SecureBERT) | +0.058 F1 -> **+13,7 % relatif** |
| **121 classes** (test 43 229) | 95,2 % | acc 0.5402 - **F1 0.3703** | acc **0.5765** - F1 **0.4139** (DistilBERT) | +0.044 F1 -> **+11,8 % relatif** |

🟡 = chiffre **transcrit** d'une session antérieure, sous l'ancien protocole (`seed=None`, `val=3000`).
Seuls les runs 71 et 121 classes ont été mesurés sous le protocole corrigé - et **seul DistilBERT a
été entraîné** ; SecureBERT n'a servi qu'en inférence. Voir [`docs/etat-avancement.md`](docs/etat-avancement.md).

**Le résultat central du projet** : plus la tâche est dure, plus le transfer learning paie.

- Sur **10 classes bien séparées**, un sac de mots de 2003 égale le transformer à 0,9 % près en 20 secondes de CPU. Le 0.97 ne mesure pas la qualité du modèle, il mesure la **facilité de la tâche** : les descriptions de CVE sont quasi-templatées ("*SQL injection in...*"), le nom du CWE est presque écrit dans le texte. Le transformer est un **luxe**.
- Sur **71 classes** (nombreuses, rares, sémantiquement proches), il prend **+9,8 %** relatifs. Là, il se justifie.

⚠️ Le premier run à 71 classes donnait 0.33 et "perdait" contre TF-IDF. C'était un **échec de protocole**, pas du modèle : validation trop petite pour un macro-F1 (24 classes < 10 exemples), sous-échantillon écrasant les classes rares, poids de classe à 28x. Corrigé -> 0.466. Détail dans `docs/cwe-cadrage.md` §5ter.

**Les quatre ablations suivent la même loi** : leur apport croît avec la difficulté de la tâche.

| Ablation | À 10 classes | À 71 classes |
|---|---|---|
| Transformer vs sac de mots | +0,8 % relatif | +9,8 % (DistilBERT) - +13,7 % (SecureBERT) |
| Fine-tuning vs backbone gelé | +35 % relatif | **+176 %** relatif |
| Modèle domaine vs généraliste | +0.0005 (sous le bruit) | +0.0154 (~8x le bruit) |

À 71 classes, le transformer **à backbone gelé** (macro-F1 0.1695) tombe même **60 % sous le TF-IDF** (0.4243) : des représentations génériques valent moins qu'un sac de mots ajusté sur le corpus. C'est l'argument le plus direct en faveur du fine-tuning.

**Le plus gros effet en laboratoire** est le **choix des classes** : macro-F1 0.414 (121) -> 0.467 (71) -> 0.791 (17) -> 0.947 (10), soit **+0.53** - contre +0.24 pour le fine-tuning vs backbone gelé et +0.015 pour le modèle domaine. ⚠️ Mais c'est un **piège** : ce gain se paie en couverture (89,8 % -> 40,5 % des CVE), et il **détruit** la valeur en production (section suivante). Ce n'est pas un meilleur modèle, c'est une tâche plus facile.

📌 **Plancher de bruit** : deux exécutions de la même config avec la même graine diffèrent de **~0.002** de macro-F1 (cuDNN n'est pas entièrement déterministe). Tout écart inférieur n'est pas interprétable - c'est le cas du gain domaine à 10 classes (+0.0005).

## ⚠️ En production, le classement s'inverse (`contrat_precision.py`)

L'accuracy ci-dessus est mesurée sur un test **filtré** : uniquement des CVE dont le CWE appartient aux classes du modèle. En production les CVE arrivent sans qu'on sache leur type. Mesuré sur la population réelle (les 43 229 CVE étiquetées de 2025, proportions naturelles) :

| | 10 classes (SecureBERT) | 71 classes (DistilBERT) |
|---|---|---|
| Accuracy "labo" | **0.9702** | 0.5974 |
| Part du flux dans le périmètre | 41,1 % | **84,5 %** |
| Faux nommage hors périmètre (`f`) | **100 %** | 71,7 % |
| **Précision réelle** sans seuil | **0.3983** | **0.5955** |
| **Contrat honnête** (cible fixée d'avance) | 89,2 % sur **6,3 %** du flux | 78,3 % sur **55,3 %** |
| **CVE justes par an** | **2 426** | **18 732** |

**La config qui affiche 40 points d'accuracy de plus est la seule des deux qui n'est pas livrable.** Les 10 classes ne couvrent que 41 % des CVE, donc 59 % du flux est condamné à être faux ; et `f` = 100 % (le modèle nomme un CWE sur *chaque* CVE hors périmètre, à n'importe quelle confiance - sa précision *redescend* même au-delà de 0.995).

**Leçon centrale** : restreindre les classes pour faire monter l'accuracy **détruit la valeur en production**. Le +0.48 de macro-F1 entre 71 et 10 classes n'est pas un gain, c'est un **transfert de l'erreur vers une zone que le test ne mesurait pas**.

### Le contrat livrable - cible fixée AVANT de voir l'année de test

Un seuil cherché *et* évalué sur la même année est optimiste ; choisir la *cible* au vu des résultats l'est encore. On fixe donc la cible d'avance, on calibre le seuil sur 2024, on l'applique tel quel à 2025.

| Cible visée sur 2024 | Seuil | Précision **2025** | Couverture | CVE/an |
|---|---|---|---|---|
| 85 % | 0.9271 | 77,4 % | 65,3 % | 28 261 |
| **90 %** | **0.9640** | **81,5 %** | **57,9 %** | **25 016** |
| 95 % | 0.9973 | 90,3 % | 31,9 % | 13 789 |
| 97 % | 0.9999 | 98,4 % | 3,7 % | 1 617 |

➜ **81,5 % de précision sur 57,9 % du flux** = 25 016 CVE/an typées, dont ~20 400 correctement (modèle entraîné sur les 218 000). La dérive temporelle coûte ~9 points : **recalibrer périodiquement**, ne pas figer le seuil.

> ⚠️ **Ce n'est plus "90,3 % sur 31,9 %".** Les versions précédentes retenaient la ligne "cible 95 %" - choisie *parce qu'*elle atteignait 90 % sur l'année de test. Le seuil était bien calibré sur 2024, mais le **choix de la cible** regardait la réponse. Le tableau reste utile comme curseur ; en faire un contrat après coup ne l'est pas.

### Comparaison des configurations, au protocole honnête

Même modèle, même protocole, seul le jeu de classes ou les données changent. Comme les deux axes bougent, on ajoute la quantité qui compte : les **CVE correctement nommées par an**.

| Configuration | Précision | Couverture | CVE/an | **CVE justes/an** |
|---|---|---|---|---|
| 10 classes | 89,2 % | **6,3 %** | 2 720 | **2 426** |
| **71 classes - 40k** (réf. ablations) | 78,3 % | 55,3 % | 23 921 | 18 732 |
| **71 classes - 218k** | **81,5 %** | **57,9 %** | 25 016 | **20 400** |
| 121 classes | 77,5 % | 56,8 % | 24 543 | 19 021 |
| SecureBERT - 71 cl. | 78,7 % | 53,7 % | 23 232 | 18 286 |

**Restreindre à 10 classes reste désastreux** : +10,9 points de précision pour **−49 de couverture**, soit 16 300 CVE justes en moins par an. Le périmètre étroit condamne 59 % du flux à être hors périmètre, et un softmax ne sait pas s'abstenir.

**SecureBERT est indistinguable de DistilBERT** (−446 CVE justes sur 18 700) : le pré-entraînement cyber n'apporte rien quand on a de quoi spécialiser un modèle généraliste.

**Plus de données fonctionne** : +3,2 points de précision *et* +2,5 de couverture, ~1 700 CVE justes de plus par an.

> ⚠️ **Correction d'un bilan antérieur.** Ce README affirmait que "5,45x plus de données" n'apportait rien. C'était mesuré au protocole où le seuil est cherché sur l'année de test - celui-là même que la section précédente dénonce. Détail dans `RAPPORT.md` §7.

### La douzième piste : changer la granularité de la cible

Les onze pistes ci-dessus cherchent toutes à **mieux prédire 71 classes**. La douzième change la cible : prédire la **famille** MITRE (les 10 `Pillar`) plutôt que le CWE précis.

| Configuration | Précision | Couverture | CVE/an |
|---|---|---|---|
| 71 classes, cible fine | 78,1 % | 55,7 % | 24 063 |
| **11 familles** | **89,2 %** | **86,6 %** | **37 420** |

**+13 400 CVE/an**, précision **et** couverture en hausse. Mécanisme : la partition en familles recouvre l'espace entier, donc le hors-périmètre passe de **15,5 % à 0,3 %**.

⚠️ **Mais une famille ne suffit pas pour agir.** MITRE attache ses mitigations aux CWE précis : 81 % des `Variant` en portent une, contre 30 % des `Pillar` et **0 des 422 catégories**. Et sur les *stratégies nommées* (vocabulaire fermé, vérifiable automatiquement), la couverture tombe à 19-33 % **sans gradient**.

| | Chiffre | Usage |
|---|---|---|
| CWE précis | **81,5 % / 57,9 %** | chaîne vers la **remédiation** |
| Famille | **89,2 % / 86,6 %** | **tri** et **reporting** |

```bash
uv run python contrat_precision.py  --data-dir data/cwe71 --model distilbert-base-uncased \
    --weights best_distilbert-base-uncased_finetune_71cl.weights.h5 \
    --run-id distilbert-base-uncased_finetune_71cl      # contrat in-sample -> runs/contrat_*.json
uv run python seuils_par_classe.py  --data-dir data/cwe71 --model distilbert-base-uncased \
    --weights best_distilbert-base-uncased_finetune_71cl.weights.h5 \
    --run-id distilbert-base-uncased_finetune_71cl      # hors echantillon -> runs/seuils_*.json
```

### Les CVE sans CWE : il n'y a pas le gisement qu'on croyait

Ce README annonçait **33 567 CVE sans CWE** comme un gisement à enrichir. Vérification faite, ce chiffre ne dit pas ce qu'il semble dire.

| Sur les 37 732 CVE sans CWE (toutes années) | |
|---|---|
| **CVE rejetées** - ce ne sont pas des vulnérabilités | **16 866 (45 %)** |
| Héritage ancien, jamais analysé | 16 154 |
| Réellement en attente d'analyse | 4 367 |

Sur l'année de test, l'angle mort réel est de **405 CVE, soit 0,9 %**.

**Trois formes de l'argument "trou de couverture" ont été testées, les trois sont tombées :**

- *La latence du NVD.* Fausse - le CWE arrive **avec** la CVE, fourni par le CNA, au même horodatage que la description (vérifié dans l'API `cvehistory/2.0`). Les CVE de 0-7 jours ont 1,3 % de CWE manquant ; celles de plus de 5 ans, 14,5 %.
- *La concentration sur certains publieurs.* Fausse - le CNA du noyau Linux est **le mieux couvert** (0,3 % sur 12 566 CVE). Le manque désigne des enregistrements vides : **38 caractères** de description médiane.
- *Un CWE mis en cache puis révisé.* Faux problème - 18,6 % des CWE sont révisés, mais **20 fois sur 21 le jour même**.

➜ **Ce projet ne se défend pas sur un trou de couverture qu'il comblerait, mais sur le typage et le reporting.**

**Ce qui reste vrai, et qui est une limite réelle** : le bruit d'étiquetage. 51 % des erreurs du modèle se produisent entre types **parent et enfant** de la taxonomie MITRE (contre 3 % attendus au hasard), et seulement 0,5 % sans lien taxonomique (contre 14 %). Le modèle hésite sur le niveau de détail, presque jamais sur la nature de la faille.

### L'usage qui reste : unifier un axe, pas combler un trou

Il n'y a rien à combler sur les CVE. Mais dans une plateforme de gestion des vulnérabilités, une grande partie des findings **ne vient pas d'une CVE** - SAST/DAST, défauts de configuration, règles propriétaires. Ceux-là n'ont ni CWE ni NVD dont hériter : c'est la seule population où il n'existe réellement rien.

**La valeur n'est pas de combler, c'est d'unifier.** Un finding de scanner ne peut pas figurer dans un rapport "par type de faille" à côté des findings CVE. Les typer **à la famille** met les deux populations sur le même axe.

| Sortie | Contrat | Usage |
|---|---|---|
| **Famille** | **89,2 % / 86,6 %** | tri, regroupement, reporting |
| CWE fin | 81,5 % / 57,9 % | choisir une remédiation - mais MITRE ne nomme une stratégie vérifiable que sur 19-33 % du catalogue |

**Le modèle encaisse-t-il le format court d'un finding ?** Mesuré, après une première réponse erronée : **la longueur n'est pas le problème, la densité l'est.**

| Sur les mêmes CVE | Longueur | Couverture | Précision |
|---|---|---|---|
| Texte complet | 653 car. | 98,7 % | 97,9 % |
| Tronqué depuis le début | 60 car. | 68,7 % | 82,5 % |
| **Fenêtre autour du type** | **58 car.** | **98,5 %** | **97,4 %** |

➜ **Le premier test n'est donc pas la couverture, mais la correspondance de vocabulaire** : un scanner disant "Missing X-Frame-Options header" atterrit-il dans la bonne famille, alors que le modèle n'a appris que des formulations NVD ? Détail et protocole dans `RAPPORT.md` §8ter.

**Contraintes d'intégration** : ne **jamais** écraser le CWE du feed - champ distinct, avec provenance, confiance et version de modèle. Feature flag, comme pour le PEP.

## Utilisation dans VS Code
1. Extensions **Python** + **Jupyter** (Microsoft).
2. Ouvrir `07_cwe_finetuning.ipynb`.
3. **Select Kernel -> `./.venv/bin/python`** (pas le kernelspec).
4. Exécuter la 1re cellule -> doit afficher `GPU available: ✅ Oui`.

### ⚠️ Après tout `uv sync` / `uv add`
**Restart Kernel** (`Ctrl+Shift+P` -> *Jupyter: Restart Kernel*) - un kernel déjà lancé ne voit pas
les paquets fraîchement installés (cause n°1 de "GPU ❌").

## GPU
```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# attendu : [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Le paquet `vendor/cuda-preload` (repris de bert)
La wheel TensorFlow 2.21 a un `RUNPATH` cassé -> TF échoue à `dlopen` libcudart/libcudnn/libcublas ->
bascule **silencieuse** en CPU. `cuda-preload` charge ces `.so` en `RTLD_GLOBAL` au démarrage de
l'interpréteur (via un `.pth` dans le site-packages) -> indépendant de `LD_LIBRARY_PATH` et du
kernelspec, marche à tous les coups dans VS Code. Réinstallé par `uv sync` (dépendance workspace).

## Contraintes de versions
| Paquet | Contrainte | Raison |
|---|---|---|
| `transformers` | `>=4.49,<4.50` | >=4.50 casse `TFAutoModel.from_pretrained` ; v5 retire le support TF |
| `tf-keras` | requis | modèles TF de `transformers` incompatibles Keras 3 |
| `tensorflow[and-cuda]` | `>=2.19,<2.22` | l'extra `and-cuda` tire les wheels CUDA (~4 Go) |

⚠️ Ne pas faire `uv add transformers` sans contrainte.

## Données & doc
- **État d'avancement / récap : [`docs/etat-avancement.md`](docs/etat-avancement.md)** - à lire en premier.
- Données : `data/cwe/` (générées par `dataset_cwe.py` depuis `../cyber_cve/data/dataset.parquet` ; variable `DATA_DIR`).
- Cadrage + étapes pédagogiques : `docs/cwe-cadrage.md`, `docs/cwe-etapes.md`.

## Registre de runs (`runs/`)
Chaque exécution complète du notebook archive sa **config** (modèle, mode, nb de classes, batch, lr,
graine) **et ses métriques** dans `runs/<RUN_ID>.json`. Le tableau d'ablation du notebook est
**généré depuis ces fichiers** - aucun chiffre recopié à la main. Le champ `provenance` distingue :
`notebook` (mesuré par un run), `baseline_tfidf.py`, `manual` (relevé à la main avant la mise en place
du registre - **à re-jouer pour certifier**).

## Structure
```
07_cwe_finetuning.ipynb     notebook principal (généré par build_cwe_nb.py)
build_cwe_nb.py             builder du notebook (nbformat)
dataset_cwe.py              prépa des splits CWE (lit ../cyber_cve/data/dataset.parquet)
baseline_tfidf.py           baseline TF-IDF + LogReg (CPU) sur un ou plusieurs sets de CWE
contrat_precision.py        courbe précision/couverture en conditions RÉELLES (in-sample)
seuils_par_classe.py        contrat HORS ÉCHANTILLON (ajuste sur val, évalue sur test) + seuils/classe
detecteur_hors_perimetre.py cascade : détecteur binaire hors-périmètre en amont du classifieur
bilan_pistes.py             bilan des pistes sous LES DEUX protocoles (corrige une erreur de §7)
dataset_famille.py          splits CVE -> famille MITRE (les 10 Pillar + résiduel)
entrainement_famille.py     entraînement sur les familles, et mode inférence seule
granularite_famille.py      relecture d'un modèle fin au niveau famille
comparaison_granularite.py  comparaison à trois voies : cible fine / relue / entraînée
actionnabilite_cwe.py       mitigations MITRE par niveau d'abstraction (texte vs stratégie nommée)
derive_etiquette.py         dérive de la vérité terrain via l'API NVD cvehistory/2.0
derive_description.py       effet de la dérive de description sur la prédiction
llm_fine_tuning_cwe.ipynb   notebook de SOUTENANCE (lecture seule des artefacts, ~10 s)
build_soutenance_nb.py      builder du notebook de soutenance
runs/                       registre des runs + contrats de production (JSON) - source des ablations
docs/                       cadrage + étapes pédagogiques
pyproject.toml              dépendances + workspace uv
vendor/cuda-preload/        correctif GPU (paquet local, repris de bert)
data/cwe/                   splits 10 CWE distinctes      (git-ignoré, regénérable)
data/cwe71/                 splits 71 classes + OTHER     (git-ignoré, regénérable)
reports/                    figures (git-ignoré)
best_<RUN_ID>.weights.h5    poids du meilleur modèle par run (git-ignoré, ~2,5 Go)
training_log_<RUN_ID>.csv   journal par époque (git-ignoré)
```
