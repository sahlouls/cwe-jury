# Classer une vulnérabilité par son type de faille

**Rapport de certification - Alyra, sujet 2 : transfer learning et fine-tuning**
Sahbi Sahloul - août 2026

---

## Résumé

À partir de la description textuelle d'une vulnérabilité (CVE), ce projet prédit son **type de faille** (CWE) en fine-tunant un transformer pré-entraîné, en TensorFlow/Keras, sur un GPU de portable de 6 Go.

**Le livrable** n'est pas une accuracy mais un **contrat** : le modèle attribue un CWE automatiquement quand il est fiable, et s'abstient sinon. Avec une cible fixée *avant* de voir l'année de test, il tient **81,5 % de précision sur 57,9 % du flux** - environ **25 000 CVE par an** typées sans intervention humaine, dont **20 400 correctement**.

> ⚠️ **Correction par rapport aux versions précédentes de ce rapport.** Nous annoncions "90,3 % sur 31,9 %". Ce chiffre était obtenu en choisissant la *cible de calibration* au vu des résultats de l'année de test - une fuite plus subtile que celle du seuil lui-même, mais une fuite. Le chiffre ci-dessus est le seul qui n'ait rien appris de l'année sur laquelle il est mesuré. Il est plus bas en précision, plus haut en couverture, et c'est le seul défendable.

**Un second contrat, pour un autre usage.** En prédisant la **famille** MITRE (les 10 `Pillar`) plutôt que le CWE précis, le même modèle atteint **89,2 % de précision sur 86,6 % du flux**. Les deux chiffres ne servent pas à la même chose : le premier alimente une chaîne de remédiation, le second le tri et le reporting (§7ter).

⚠️ **L'usage visé ne tient pas, et c'est mesuré.** Le CWE devait alimenter un modèle d'exploitabilité (PEP). Vérification faite sur ce modèle : lui retirer le CWE - le **vrai**, celui du NVD - l'**améliore** (PR 0.8369 -> 0.8492). Aucun classifieur, si bon soit-il, ne peut servir cet usage. Détail en §8bis.

**Le résultat central est méthodologique, et il est contre-intuitif** : sur cette tâche, *l'accuracy ne prédit pas la valeur en production*. La configuration qui affiche la meilleure accuracy du projet (0.9702) s'effondre à **0.3983** de précision sur le flux réel et n'est pas livrable. Trois axes indépendants - largeur du périmètre, nombre de classes, qualité du modèle - le confirment : ce qui détermine la valeur d'usage est le **comportement d'abstention** du modèle, une propriété qu'aucune métrique du rapport de classification ne mesure.

---

## 1. Le problème

Une **CVE** est une vulnérabilité précise dans un produit précis (`CVE-2021-44228`, la faille Log4Shell). Un **CWE** est la catégorie de défaut sous-jacente (`CWE-502`, désérialisation non fiable). L'image : la CVE est "la serrure cassée de l'appartement 4B", le CWE est "les serrures de cette marque se crochètent".

Chaque CVE devrait porter un CWE, sa cause racine. Dans la base utilisée, **37 732 CVE n'en ont aucun** - mais ce chiffre, que les versions précédentes de ce rapport présentaient comme un gisement, ne veut pas dire ce qu'il semble dire : **45 % sont des CVE rejetées**, retirées du catalogue parce que ce ne sont pas des vulnérabilités. Sur l'année de test, l'angle mort réel se réduit à **405 CVE, soit 0,9 %** (§8). Les typer automatiquement devait avoir une valeur opérationnelle : triage, reporting, et - combiné à un modèle d'exploitabilité développé par ailleurs (PEP) - une vulnérabilité enrichie `{type de faille + probabilité d'exploitation}`.

⚠️ **Cette dernière justification, la principale, a été vérifiée en fin de projet et elle est fausse** (§8bis). Les usages "humains" - triage, reporting, filtrage par famille - n'ont jamais été cadrés et restent ouverts.

La tâche est une **classification multi-classe** sur du texte technique en anglais.

---

## 2. Les données

**Source** : 369 677 CVE avec description, CWE principal et année. Après exclusion des CWE manquants et des descriptions vides : **299 418 CVE exploitables**.

**Le vrai enjeu est la sélection des classes.** Le référentiel CWE v4.20 compte **969 faiblesses**, réparties en **422 catégories** et organisées par **59 vues** ; ~788 identifiants apparaissent dans les données, avec un déséquilibre extrême (CWE-79 à lui seul représente plus de 40 % des CVE). Prédire 756 classes n'a pas de sens ; il faut choisir un sous-ensemble, et **ce choix s'est révélé être la décision la plus lourde de conséquences du projet**.

Quatre configurations ont été construites et évaluées :

| Configuration | Classes | Part du flux couverte |
|---|---|---|
| 10 CWE critiques distinctes | 10 | 40,5 % |
| 17 CWE critiques (Top 25 MITRE ∩ >= 3000 CVE) | 17 | 59,2 % |
| **CWE >= 500 CVE + fourre-tout `CWE-OTHER`** | **71** | **89,8 %** |
| CWE >= 200 CVE + fourre-tout | 121 | 95,2 % |

**Split temporel**, et non aléatoire : entraînement jusqu'à 2023, validation sur 2024, test sur 2025. Un split aléatoire aurait laissé fuiter des vulnérabilités contemporaines entre les jeux et surestimé la performance. Le split temporel simule la réalité - on classe des vulnérabilités futures avec un modèle entraîné sur le passé - et il fait apparaître une **dérive temporelle** que la section 6 quantifie.

Tailles pour la configuration retenue : 217 966 CVE en entraînement, 38 223 en validation, **43 229 en test**. L'entraînement est sous-échantillonné à 40 000 CVE pour tenir dans un temps raisonnable ; **le test est utilisé en entier**.

---

## 3. La méthode

### 3.1 Transfer learning et fine-tuning

Le **transfer learning** est le principe : réutiliser un modèle qui connaît déjà l'anglais plutôt que d'apprendre la langue de zéro. Le **fine-tuning** est la méthode : poursuivre son entraînement sur nos données.

L'architecture reprend celle du cours : `TFAutoModel` charge le transformer **nu**, on extrait le vecteur du token `[CLS]` (768 dimensions, résumé contextuel de la description), et on y branche notre propre tête - `Dropout(0.3)` puis `Dense(NUM_CLASSES)` avec softmax.

**Deux modes de transfer learning sont comparés** : le *fine-tuning complet*, où tout le réseau est réentraîné à faible learning rate (2e-5, valeur recommandée par le papier BERT), et la *feature extraction*, où le backbone est gelé et seule la tête apprend, à learning rate élevé (1e-3, puisqu'elle part de zéro et qu'il n'y a aucun poids pré-entraîné à préserver).

**Deux modèles sont comparés** : DistilBERT (66 M paramètres, généraliste) et SecureBERT (125 M, pré-entraîné sur du texte de cybersécurité).

### 3.2 Contraintes matérielles et leurs conséquences

Le GPU dispose de 6 Go. La **précision mixte** (`mixed_float16`) divise par deux l'occupation mémoire et rend le fine-tuning possible ; la couche de sortie est forcée en `float32` pour la stabilité du softmax. SecureBERT, deux fois plus gros, n'entre qu'à batch 8 contre 16 pour DistilBERT - **la comparaison des deux modèles fait donc varier aussi la taille de batch**, contrainte subie qu'il faut signaler.

### 3.3 Le protocole expérimental

Trois variables pilotent l'expérience - modèle, mode, jeu de classes - et **tout le reste en est dérivé** : nombre de classes lu dans les données, taille de validation, nombre d'époques, patience, plafond des poids de classe. C'est ce qui garantit qu'en changeant une variable on ne change qu'elle.

Chaque exécution archive sa configuration complète et ses métriques dans `runs/<id>.json`. Les tableaux d'ablation du notebook sont **générés depuis ces fichiers** : aucun chiffre n'est recopié à la main.

**Plancher de bruit mesuré** : trois exécutions de la même configuration à graine identique diffèrent de **~0.002** de macro-F1 (certains noyaux cuDNN ne sont pas déterministes). Tout écart inférieur n'est pas interprétable - ce chiffre sert de référence à toutes les comparaisons qui suivent.

### 3.4 Le déséquilibre des classes

CWE-79 domine massivement. Sans correction, le modèle a intérêt à le prédire souvent. On applique donc des **poids de classe** inversement proportionnels à la fréquence, et on pilote sur le **macro-F1** - qui traite toutes les classes à poids égal - plutôt que sur l'accuracy.

Ces poids sont **plafonnés à 10x**. Sans plafond, la formule attribuait 28x à une classe de 20 exemples (rapport max/min de 271x), produisant des gradients à très forte variance sur une classe de toute façon inapprenable. ⚠️ Un poids de classe redistribue l'attention du modèle ; **il ne crée pas d'information**.

---

## 4. Résultats en laboratoire

Chaque configuration est comparée à un baseline **TF-IDF + régression logistique** entraîné sur les mêmes splits, le même sous-échantillon et la même graine - 33 secondes de CPU contre plusieurs heures de GPU.

| Jeu de classes | TF-IDF (macro-F1) | Transformer (macro-F1) | Gain relatif |
|---|---|---|---|
| 10 CWE distinctes | 0.9391 | 0.9468 | **+0,8 %** |
| 17 CWE critiques | 0.7909 | 0.8050 | +1,8 % |
| 71 classes | 0.4243 | 0.4671 | +9,8 % |
| 71 classes (SecureBERT) | 0.4243 | **0.4825** | **+13,7 %** |
| 71 classes (**jeu complet, 218k**) | 0.4638 | **0.5001** | +7,8 % |
| 121 classes | 0.3703 | 0.4139 | +11,8 % |

Sur la ligne "jeu complet", les **deux** modèles ont été réentraînés sur les 218 000 CVE : le baseline TF-IDF progresse lui aussi (0.4243 -> 0.4638), et l'écart relatif se resserre. Les données profitent aux deux.

**Le transformer ne se justifie que sur les tâches difficiles.** Sur 10 classes bien séparées, un modèle linéaire de 2003 l'égale à 0,8 % près : les descriptions de CVE sont quasi-templatées ("*SQL injection in the login form...*"), le nom du type de faille est presque écrit dans le texte, ce qu'un sac de mots capte parfaitement. Le 0.97 ne mesure pas la qualité du modèle, il mesure la **facilité de la tâche**.

### Les quatre ablations

| Ablation | Mesure | Lecture |
|---|---|---|
| **Classes** | 0.4139 (121) -> 0.4671 (71) -> 0.7909 (17) -> 0.9473 (10) | **+0.53** - le plus gros effet en laboratoire |
| **Mode** | gelé 0.7030 -> fine-tuning 0.9468 (10 cl.) - gelé **0.1695** -> fine-tuning **0.4671** (71 cl.) | le fine-tuning est indispensable, et **d'autant plus que la tâche est dure** |
| **Modèle** | domaine +0.0005 (10 cl.) vs **+0.0154** (71 cl.) | l'apport du modèle domaine **croît avec la difficulté** |

**L'ablation "mode" suit la même loi que les autres.** En absolu, le fine-tuning apporte +0.24 de macro-F1 à 10 classes et +0.30 à 71 classes - écart modeste. Mais en relatif, il fait passer de 0.7030 à 0.9468 (**+35 %**) sur la tâche facile, et de 0.1695 à 0.4671 (**+176 %**, soit près du triple) sur la tâche difficile.

Un détail vaut d'être relevé : à 71 classes, le transformer **à backbone gelé** (0.1695) est **60 % en dessous du TF-IDF** (0.4243). Des représentations génériques, aussi riches soient-elles, sont moins utiles pour cette tâche qu'un sac de mots *ajusté sur ce corpus*. C'est l'argument le plus direct en faveur du fine-tuning : ce n'est pas la puissance du pré-entraînement qui compte, c'est son **adaptation**.

Sur l'ablation "modèle", le détail est instructif : à 10 classes, l'écart entre DistilBERT et SecureBERT (+0.0005) est **inférieur au plancher de bruit** - il n'est pas interprétable. À 71 classes, il atteint +0.0154, soit huit fois le bruit. Et la trajectoire d'apprentissage explique pourquoi on pouvait s'y tromper : SecureBERT est **derrière pendant cinq époques**, croise à la sixième, finit devant. Avec un budget de 4 à 5 époques, il aurait perdu. Une partie de ce qu'on croyait mesurer sur "généraliste contre domaine" mesurait en réalité la vitesse de convergence.

---

## 5. Le résultat central : l'accuracy ne survit pas au flux réel

Les scores ci-dessus sont mesurés sur un test **filtré** : uniquement des CVE dont le type appartient aux classes du modèle. **En production, les CVE arrivent sans qu'on sache leur type.**

Un modèle à softmax a *N* sorties dont la somme fait 1. Il doit répartir sa masse de probabilité quelque part : **il ne peut structurellement pas "ne pas répondre"**. Face à une vulnérabilité d'un type qu'il n'a jamais appris, il attribue quand même un CWE - et cette réponse est forcément fausse.

Pour la configuration à 10 classes, qui ne couvre que 41 % des CVE réelles :

```
test filtré  0.9702   ──►   flux réel 2025  0.3983      57 points d'écart
```

Le modèle nomme un CWE sur **100 %** des CVE hors de son périmètre, à n'importe quel niveau de confiance. Sa précision *redescend* même au-delà de 0.995 : il est aussi sûr de lui, voire plus, quand il se trompe. C'est logique - il n'a jamais vu un seul contre-exemple pendant son entraînement.

### Comparaison à modèle constant

Pour isoler l'effet du périmètre, les trois configurations partagent le **même modèle et le même protocole** ; seul le jeu de classes change. Contrat mesuré au **protocole honnête** : cible fixée d'avance, seuil calibré sur 2024, appliqué à 2025. Les deux axes bougeant, on ajoute la seule quantité que l'utilisateur reçoit — les **CVE correctement nommées par an**.

| Périmètre | Flux dans le périmètre | Faux nommage `f` | Précision | Couverture | **CVE justes/an** |
|---|---|---|---|---|---|
| 10 classes | 41,1 % | **100 %** | 89,2 % | **6,3 %** | **2 426** |
| **71 classes** | 84,5 % | 71,7 % | 78,3 % | **55,3 %** | **18 732** |
| 121 classes | 90,8 % | 86,9 % | 77,5 % | **56,8 %** | **19 021** |

**Le périmètre étroit s'effondre** : 10 classes gagne 11 points de précision et perd **49 points de couverture**, soit 16 300 CVE justes en moins par an. La cause est celle du §5 : 59 % du flux est hors périmètre, et le modèle y nomme un CWE dans **100 %** des cas.

**Mais 71 et 121 classes sont indistinguables** — 121 gagne 1,5 point de couverture et perd 0,8 de précision. On retient 71 pour la lisibilité et le coût, **pas** parce que la mesure désigne un optimum.

> ⚠️ **Correction.** Les versions précédentes de ce rapport annonçaient ici « 0,4 % / 31,9 % / 8,6 %, facteur 77, relation non monotone, il existe un optimum et il se mesure ». Ces chiffres venaient du protocole qui impose 90 % de précision **sur l'année de test** — celui-là même que le §6 dénonce. **L'argument de l'optimum mesuré ne tient pas** (voir §7).

### Le modèle de domaine ne change rien

| Sur 71 classes | macro-F1 | `f` | Précision | Couverture | CVE justes/an |
|---|---|---|---|---|---|
| **DistilBERT** (40k) | 0.4671 | 71,7 % | 78,3 % | **55,3 %** | **18 732** |
| SecureBERT (40k) | **0.4825** | 78,4 % | 78,7 % | 53,7 % | 18 286 |

SecureBERT est meilleur en laboratoire (+0.015 de macro-F1) et **indistinguable** en production : 446 CVE justes d'écart sur 18 700, dans le sens défavorable. Le pré-entraînement sur du texte de cybersécurité n'apporte rien — l'explication plausible étant que le vocabulaire des descriptions de CVE est très stéréotypé, et que DistilBERT a largement assez d'exemples pour l'apprendre en fine-tuning.

> ⚠️ **Correction.** Ce paragraphe s'intitulait « Le meilleur modèle donne le pire contrat » et annonçait « 43 % de couverture en moins ». Au protocole honnête, l'écart est de 1,6 point de couverture. L'affirmation était un artefact de mesure, pas un résultat.

### La leçon : le protocole d'évaluation décide du classement

| Axe modifié | Effet en laboratoire | Effet en production (CVE justes/an) |
|---|---|---|
| Périmètre 71 → 10 classes | **+0.48** macro-F1 | **18 732 → 2 426** ❌ |
| Périmètre 71 → 121 classes | −0.05 macro-F1 | 18 732 → 19 021 (indistinguable) |
| Modèle DistilBERT → SecureBERT | **+0.015** macro-F1 | 18 732 → 18 286 (indistinguable) |
| **5,45× plus de données** (40k → 218k) | **+0.033** macro-F1 | **18 732 → 20 400** ✅ |

**Un seul axe détruit réellement la valeur d'usage : réduire le périmètre.** Deux sont neutres. Et le quatrième — plus de données — **aide**, sur les deux axes du contrat à la fois.

> ⚠️ **Correction, et c'est la plus instructive du rapport.** Ce tableau concluait auparavant : « quatre fois, améliorer la métrique de laboratoire ne se traduit pas en valeur d'usage — et trois fois sur quatre, elle la dégrade », avec « plus de données : 32,2 % → 31,9 % ». C'était mesuré au protocole fuité. Corrigé, **le levier le plus banal de l'apprentissage profond fonctionne** : +1 668 CVE justes par an. Nous nous étions trompés en utilisant, pour comparer, la méthode que nous dénoncions par ailleurs. Détail en §7.

**Ce qui reste vrai, et c'est le résultat central.** Sur la configuration à 10 classes, le hors-périmètre représente **59 % du flux** et le modèle y nomme un CWE dans 100 % des cas : la moitié du flux est fausse par construction, quelle que soit la qualité du modèle. C'est une **taxe imposée par le périmètre, que la qualité du modèle ne rembourse pas** — améliorer la classification *à l'intérieur* du périmètre ne retire aucune de ces erreurs. Sur 71 classes cette taxe tombe à 15,5 % × 71,7 % ≈ 11 % du flux, ce qui la rend supportable.

---

## 6. Le livrable : un contrat précision/couverture

Plutôt que de répondre à tout, le modèle **s'abstient** sous un seuil de confiance. Le seuil est **calibré sur l'année de validation et évalué sur l'année de test** - en production, on choisit le seuil avant de voir les données.

| Cible visée sur 2024 | Seuil | Précision **2025** | Couverture | CVE/an |
|---|---|---|---|---|
| 85 % | 0.9271 | 77,4 % | 65,3 % | 28 261 |
| **90 %** | **0.9640** | **81,5 %** | **57,9 %** | **25 016** |
| 95 % | 0.9973 | 90,3 % | 31,9 % | 13 789 |
| 97 % | 0.9999 | 98,4 % | 3,7 % | 1 617 |

**La dérive temporelle coûte environ 9 points** : un seuil calé pour 90 % sur 2024 ne rend que 81,5 % sur 2025. Le vocabulaire des vulnérabilités bouge en un an, et le modèle est un peu moins bien calibré qu'il ne l'était. En exploitation, **il faut recalibrer périodiquement** - c'est une conséquence opérationnelle directe de cette mesure.

➜ **Contrat livrable : 81,5 % de précision sur 57,9 % du flux**, soit 25 016 CVE par an typées automatiquement, dont environ 20 400 correctement ; les 42 % restants partent en revue humaine.

> ⚠️ **Pourquoi ce n'est plus "90,3 % sur 31,9 %".** Les versions précédentes retenaient la ligne "cible 95 %", en la choisissant parce qu'elle atteignait 90 % **sur l'année de test**. Le seuil, lui, était bien calibré sur 2024 - mais le *choix de la cible* regardait la réponse. C'est une fuite d'un cran, et elle produit un chiffre qui ne se reproduira pas : on annonce une précision qu'on a sélectionnée après coup, sur une couverture deux fois moindre.
>
> Le tableau ci-dessus reste utile comme **curseur** : viser plus haut fonctionne, mais le prix est brutal. Ce qu'il ne faut pas faire, c'est choisir la ligne *après* avoir vu les résultats et l'annoncer comme un contrat.

> ⚖️ **Choix entre les deux modèles à 71 classes.** Au protocole honnête, celui entraîné sur **218 000 CVE domine** celui entraîné sur 40 000 sur les deux axes : 81,5 % contre 78,3 % de précision, 57,9 % contre 55,3 % de couverture. C'est lui qu'on retient. Le modèle à 40 000 reste la **référence des ablations**, parce que toutes les comparaisons à variable unique ont été menées contre lui.

⚠️ **Le compromis se paie cher dans les hautes précisions** : viser 97 % au lieu de 95 % à la calibration fait chuter la couverture de 31,9 % à 3,7 %. Le point retenu (cible 90 %) est en revanche sur une partie plate de la courbe, donc robuste - c'est aussi une raison de le préférer.

---

## 7. Ce qui n'a pas marché - et une erreur de notre part

Onze hypothèses ont été formulées puis mesurées. **Notre premier bilan concluait que les onze étaient réfutées. Il était faux**, et nous le racontons plutôt que de le corriger en silence : l'erreur est la même que celle dénoncée en §5, commise cette fois par nous.

### L'erreur de protocole

Pour comparer les configurations, nous demandions à chacune d'atteindre **exactement 90 % de précision**, puis nous regardions laquelle couvrait le plus de CVE. Or ce seuil de 90 % était **cherché sur l'année de test**. Toutes les configurations étaient traitées pareil, ce qui donnait au procédé une apparence d'équité - mais un seuil ajusté sur la réponse ne classe pas les modèles selon ce qu'ils valent, il les classe selon la facilité avec laquelle on peut les forcer à 90 %.

### Le bilan corrigé

Au protocole honnête (cible fixée d'avance, seuil calibré sur 2024), les deux axes bougent. Le critère devient une **relation de domination** : une piste n'aide que si elle améliore *à la fois* la précision et la couverture. On y ajoute la seule quantité qui compte pour un utilisateur - les **CVE correctement nommées par an**, soit précision x réponses.

| Hypothèse | Ancien bilan | Précision | Couverture | **CVE justes/an** | Verdict |
|---|---|---|---|---|---|
| **5,45x plus de données** (40k -> 218k) | −1 150 | **+3,2 %** | **+2,5 %** | **+1 668** | **AIDE** |
| Élargir à 121 classes | −10 231 | −0,8 % | +1,4 % | +291 | ambiguë |
| Sorties sigmoïdes | −12 893 | +1,2 % | −1,2 % | −120 | ambiguë |
| Modèle de domaine (SecureBERT) | −6 943 | +0,4 % | −1,6 % | −446 | ambiguë |
| Restreindre à 10 classes | −15 271 | +10,9 % | **−49,0 %** | **−16 305** | ambiguë |

*(Référence : 23 921 CVE nommées à 78,3 % de précision, soit 18 732 CVE justes par an - modèle 40k.)*

**Plus de données fonctionne.** Le levier le plus banal de l'apprentissage profond gagne 3,2 points de précision *et* 2,5 de couverture, soit environ **1 700 CVE justes de plus par an**. Notre premier bilan le rangeait parmi les échecs. Les deux mesures portent sur les mêmes modèles et les mêmes fichiers ; seul le protocole change.

**Les quatre autres ne dominent sur aucun axe.** Elles échangent de la précision contre de la couverture. Aucune n'est réfutée au sens strict, aucune n'aide. Le cas de "10 classes" montre la limite du critère de domination : +10,9 points de précision pour −49 de couverture, ce que le critère classe "ambigu" et que le bon sens classe désastreux - 16 300 CVE justes en moins par an.

### Les pistes jugées sur d'autres axes

| Hypothèse | Attendu | Mesuré |
|---|---|---|
| Une classe `CWE-OTHER` permet de refuser | il dira "autre" face à l'inconnu | il ne l'utilise que dans **28,3 %** des cas |
| Des seuils par classe libèrent de la couverture | gain gratuit | **déplacement, pas gain** : 82,5 % de précision mais 47,2 % de couverture - 3 500 CVE nommées **en moins** |
| Un détecteur binaire hors-périmètre en cascade | attaque le faux nommage de front | **nul**, malgré une AUC de 0.859 : le signal est déjà dans la confiance du classifieur |
| Prédiction hiérarchique avec repli | répondre au niveau dont on est sûr | réponses **exactes** 11,4 %, granularité moyenne 7,2 classes : trop vague |
| Entraînement multi-étiquette hiérarchique | apprendre les ancêtres en même temps | macro-F1 **0.414** contre 0.467 |
| Sigmoïdes seules (test à variable unique) | isoler l'effet de la sigmoïde | macro-F1 **0.462** contre 0.467 |

### La leçon, et c'est la troisième fois

| §5 | Le protocole d'**évaluation** classait les modèles à l'envers |
| :-- | :-- |
| §6 | Le protocole de **calibration** produisait un chiffre optimiste de 9 à 12 points |
| **§7** | Le protocole de **comparaison** nous a fait rejeter un levier qui fonctionne |

Trois fois le même mécanisme : une décision de mesure prise sans y penser, qui oriente une conclusion. La troisième fois, nous l'avions déjà écrite deux fois - et nous y sommes tombés quand même. C'est le résultat le plus transférable de ce travail, et il ne porte pas sur les CWE.

---

## 7bis. La douzième piste : changer la granularité de la cible

Les onze pistes précédentes partagent un présupposé jamais interrogé : elles cherchent toutes à **mieux prédire 71 classes**. La douzième change la cible - prédire la **famille** MITRE (les 10 `Pillar` + un bucket résiduel) plutôt que le CWE précis.

**Ce n'est pas "réduire à 10 classes"**, piste déjà testée. Ce sont deux opérations opposées : réduire le périmètre garde les CWE les plus fréquents et **jette le reste dehors** (59 % des CVE hors périmètre) ; repartitionner **range tous les 969 CWE** dans une famille, et le hors-périmètre disparaît par construction - **15,5 % -> 0,3 %**.

Trois configurations, budget et graine identiques, même population de test :

| Configuration | Précision | Couverture | CVE/an |
|---|---|---|---|
| **A.** 71 classes, cible fine | 78,1 % | 55,7 % | 24 063 |
| **B.** 71 classes, **relue** en famille | 89,2 % | 86,6 % | 37 420 |
| **C.** 11 familles, **entraînée** | 89,6 % | 87,1 % | 37 671 |

**+13 400 CVE nommées par an**, avec la précision qui monte de 11 points et la couverture de 31 **en même temps**. Ce n'est pas un déplacement sur la courbe précision/couverture - c'est la courbe entière qui se déplace.

**B contre C n'est pas tranchable**, et c'est instructif. L'écart bascule de signe selon le budget d'entraînement (−933 CVE à 10 époques, +251 à 20), avec une amplitude quatre fois supérieure aux intervalles de confiance publiés. Ces intervalles, calculés par bootstrap sur le jeu de test, bornaient la variance du **tirage** alors que la variance dominante était celle de l'**entraînement**. Un intervalle de confiance ne protège que contre la source d'incertitude qu'il modélise.

> **Dérogation à une règle annoncée d'avance.** Nous avions fixé : "si C gagne à budget supérieur, entraîner aussi le modèle à 71 classes sur 20 époques". C a gagné (+251). **Nous ne l'avons pas fait**, parce que la mesure établit que la question est sous-déterminée à cette précision - la trancher demanderait plusieurs graines par configuration, pour un arbitrage qui ne change aucune conclusion. **Cette dérogation a été décidée après avoir vu le résultat, et nous l'assumons comme telle.**

---

## 7ter. Mais une famille ne suffit pas pour agir

Ce test était motivé par une hypothèse que nous avions posée sans la vérifier : *"si l'action aval est le choix d'un contrôle compensatoire, une famille suffit probablement"*. **Elle est fausse, et MITRE le dit.**

| Niveau d'abstraction | CWE | Texte de mitigation | **Stratégie nommée** |
|---|---|---|---|
| `Pillar` (nos familles) | 10 | 30 % | **20 %** |
| `Class` | 114 | 57 % | 26 % |
| `Base` | 539 | 66 % | 19 % |
| `Variant` | 299 | **81 %** | 33 % |
| **Catégories** | **422** | **0 %** | **0 %** |

*(Source : catalogue MITRE complet `cwec_v4.20.xml`, et non un export de vue - `1000.csv` ne contient aucune catégorie.)*

Répondre "famille `resource-control`" ne désigne aucun correctif : la documentation qui permettrait d'agir est attachée au CWE précis.

**Mais aller vers le fin ne débloque pas la remédiation pour autant.** Sur les **stratégies nommées** - le vocabulaire fermé de 14 valeurs qu'un contrôle automatisé sait vérifier - la couverture tombe à 19-33 % et **le gradient disparaît** : les `Base` (19 %) sont moins bien pourvues que les `Class` (26 %). Un CWE plus précis fait exister une documentation en prose, pas une stratégie vérifiable.

➜ **Les deux contrats servent deux usages distincts :**

| | Chiffre | Usage |
|---|---|---|
| CWE précis | **81,5 % / 57,9 %** | chaîne vers la **remédiation** |
| Famille | **89,2 % / 86,6 %** | **tri** et **reporting** |

**Un dernier croisement** : les familles que le modèle ne prédit jamais sont aussi celles que MITRE documente le moins - 56 % de CWE documentés contre 72 % ailleurs (hors bucket résiduel). Les deux défaillances se composent, mais sur **1,1 % du flux** seulement.

---

## 8. Limites

**L'usage visé ne tient pas** - voir §8bis, c'est la limite principale : le CWE n'apporte rien au PEP.

**Il n'y a pas de trou de couverture à combler, et nous l'avons cherché.** Trois formes de l'argument ont été testées et **les trois sont tombées** :

- *La latence du NVD.* Fausse. Le CWE arrive **avec** la CVE, fourni par le CNA, au même horodatage que la description - vérifié dans l'API `cvehistory/2.0`. Les CVE de 0-7 jours ont 1,3 % de CWE manquant, celles de plus de 5 ans en ont 14,5 %.
- *La concentration sur certains publieurs.* Fausse. Le CNA du noyau Linux, soupçonné d'être le pire, est **le mieux couvert** (0,3 % sur 12 566 CVE). Le manque désigne des enregistrements vides : **38 caractères** de description médiane, sans vendeur résolu.
- *Un CWE mis en cache puis révisé.* Faux problème. 18,6 % des CVE voient leur CWE révisé, mais **20 révisions sur 21 le jour même** - une plateforme qui ingère à J+1 reçoit la valeur corrigée. Révision postérieure : 1 sur 113, IC 95 % [0,2 % ; 4,8 %].

➜ **Ce projet ne se défend donc pas sur un trou de couverture qu'il comblerait. Il se défend sur le typage et le reporting.**

**Le bruit d'étiquetage est réel, mesuré par deux voies indépendantes.** 51 % des erreurs du modèle se produisent entre types **parent et enfant** de la taxonomie MITRE (contre 3 % attendus au hasard) et seulement 0,5 % sans lien taxonomique (contre 14 %). Indépendamment, l'historique NVD montre que 18,6 % des CWE sont révisés après publication, dont un tiers change de famille. Une part de notre taux d'erreur est du désaccord d'annotation.

**Une seule graine par configuration, et nous avons mesuré ce que ça coûte.** Le comparatif de §7bis change de signe selon le budget d'entraînement, avec une amplitude quatre fois supérieure à nos intervalles de confiance. Des moyennes sur plusieurs graines n'auraient pas été un raffinement : elles étaient nécessaires pour tout écart inférieur à quelques pour cent.

**La transportabilité - la limite la plus importante pour qui voudrait réutiliser ce modèle.** Le contrat vaut pour des descriptions de **type NVD** (304 caractères médians). La seule population où il n'existe aujourd'hui **aucun typage** est celle des findings qui ne viennent pas d'une CVE : résultats SAST/DAST, défauts de configuration, règles propriétaires de scanners. Ils n'ont ni CWE ni NVD dont hériter, et ils sont bien plus nombreux dans une plateforme de gestion des vulnérabilités que les CVE non étiquetées.

**Nous avons cherché à savoir si le modèle y survivrait, et notre première réponse était fausse.** Une mesure par tranche de longueur sur les CVE réelles suggérait un effondrement de la couverture - mais elle est **confondue** : la longueur sélectionne un publieur autant qu'une difficulté. Une seconde mesure, par troncature, semblait confirmer ; elle était **mal spécifiée**, car tronquer depuis le début supprime le terme de type dans 83 % des cas - cela mesure l'absence d'information, pas sa concision.

Le bon test isole la **densité** : une fenêtre de 60 caractères prise *autour* du terme de type, soit le profil exact d'un finding de scanner, qui mène par le type.

| Sur les mêmes CVE | Longueur | Couverture | Précision |
|---|---|---|---|
| Texte complet | 653 car. | 98,7 % | 97,9 % |
| Tronqué depuis le début | 60 car. | 68,7 % | 82,5 % |
| **Fenêtre autour du type** | **58 car.** | **98,5 %** | **97,4 %** |

➜ **La longueur n'est pas le problème : la densité l'est.** Un texte de 58 caractères qui énonce le type donne le même résultat qu'une description de 653. Le modèle encaisse le format court.

**Ce qu'il faut mesurer en premier sur des findings réels n'est donc pas la couverture, mais la correspondance de vocabulaire.** Le risque n'est pas que le texte soit court - c'est qu'un scanner dise "Missing X-Frame-Options header" là où le modèle n'a appris que "Improper Neutralization of Input During Web Page Generation". Un problème de lexique, pas de format.

*Deux réserves* : la fenêtre contient les formulations **exactes** du NVD, celles sur lesquelles le modèle a été entraîné ; et les 50 % de CVE au type explicite sont les plus faciles, d'où une référence à 98,7 % contre 89,3 % sur l'ensemble.

**La dérive de description : bornée, mais non testée là où ça compterait.** 15 % des CVE voient leur texte modifié après publication ; sur les 17 cas observés, la prédiction est identique 17 fois sur 17 (borne haute 16,2 %, soit un effet plafonné à ~2,4 % des CVE). Mais ces 17 cas sont tous des retouches modérées - similarité minimale 0,775, aucun cas sous 0,60. Le cas d'une description squelettique substantiellement réécrite **n'est pas testé**. C'est "non testé", pas "réfuté".

**Une réserve qui joue contre nos propres conclusions.** Depuis 2024 le NVD diffère massivement son analyse (17 019 CVE de 2025 en statut `Deferred`). Moins d'analyses NVD signifie moins de révisions **et** moins d'enrichissements de description observés : les deux mesures précédentes sous-estiment probablement ce qu'elles vaudraient avec un NVD à jour.

**Deux comparaisons ne sont pas parfaitement isolées.** SecureBERT tourne à batch 8 contre 16 pour DistilBERT (contrainte des 6 Go). Les configurations à 17 classes proviennent de sessions antérieures sous un protocole moins rigoureux - elles sont marquées comme telles dans le registre.

**Le sous-échantillonnage à 40 000 exemples pénalise les classes rares** : une classe à 500 CVE n'en garde qu'une centaine, et le macro-F1 la compte à poids égal avec CWE-79. Le modèle entraîné sur les 218 000 CVE corrige en partie ce défaut, et c'est cohérent avec le fait qu'il domine (§7).

---

## 8bis. L'usage visé, vérifié - et réfuté

Le projet devait alimenter le **PEP**, un modèle d'exploitabilité développé par ailleurs, en lui fournissant le type de faille. Cette justification n'avait jamais été vérifiée. Elle l'a été en fin de projet, en vingt minutes.

### Le poids du CWE dans le PEP

Le PEP est une régression logistique sur trois blocs : catégorielles en one-hot (dont `primary_cwe`), numériques, et **la description en unigrammes binaires**. Somme des coefficients absolus par bloc :

| Bloc | nb de variables | part du poids |
|---|---|---|
| **Texte (unigrammes)** | 2 442 | **83,5 %** |
| Autres catégorielles | 195 | 11,2 % |
| **`primary_cwe`** | 121 | **5,0 %** |
| Flags mots-clés | 19 | 0,1 % |

### L'ablation : le vrai CWE dégrade le PEP

Entraînement identique, une seule variable retirée. Test sur 2025 (45 275 CVE) :

| Variante du PEP | ROC | PR |
|---|---|---|
| **A.** avec `primary_cwe` - production actuelle | 0.9209 | 0.8369 |
| **B.** sans CWE du tout | **0.9251** | **0.8492** |
| **C.** sans CWE + booléen "CWE manquant" | 0.9232 | 0.8439 |

**Le meilleur modèle n'a aucun CWE.** Et c'est avec le CWE *réel* du NVD - information parfaite. Aucun classifieur, si bon soit-il, ne peut donc servir cet usage.

### Pourquoi

**L'information est déjà là.** Le PEP lit la description complète, qui pèse 83,5 % de son poids. Notre classifieur prend **cette même description** en entrée : il produit une fonction de ce que le PEP lit déjà. Un résumé n'ajoute rien à qui possède l'original.

**Et une variable redondante coûte quand même.** Les 121 colonnes du CWE ont chacune un coefficient à estimer. Sans signal nouveau à capter, ils s'ajustent sur du bruit qui ne se reproduit pas en 2025 - surapprentissage classique.

**Le signal "CWE manquant" était un mirage.** Il est réel dans les données brutes (5 % d'exploits contre 33 %) et c'était la modalité la plus influente du bloc, à −1.407. Mais il **duplique `nvd_status`**, déjà présent : parmi les CVE `Rejected`, 94,6 % n'ont pas de CWE ; parmi les `Awaiting Analysis`, 66,6 %. "CWE manquant" signifie "CVE rejetée ou pas encore analysée", et le PEP le sait déjà par une variable plus précise. D'où la variante C, qui n'aide pas.

### Ce que ça change, et ce que ça ne change pas

**Mort** : le CWE comme variable d'entrée du PEP.

**Ouvert** : les usages humains - triage, reporting, filtrage par famille de faille. Jamais cadrés, jamais mesurés. Le contrat livrable reste réel (≈ 25 000 CVE/an à 81,5 % de précision, dont ~20 400 correctement typées) ; il ne sert simplement pas la finalité qu'on lui avait assignée.

**Gagné au passage** : retirer `primary_cwe` du PEP lui fait gagner +0.012 de PR et lui retire 121 dimensions. Un modèle plus simple et plus robuste - livrable concret de ce travail, quoique pas celui qu'on visait.

⚠️ **Réserve** : un seul découpage temporel, un seul tirage de 150 000, et des écarts faibles (0.012 de PR). La direction est cohérente sur quatre conditions et l'entraînement d'une régression logistique est déterministe, mais une validation croisée temporelle serait la confirmation propre.

### La leçon, et elle est inconfortable

**Cette mesure coûte vingt minutes et était faisable avant le premier entraînement.** Elle a été faite en dernier, après onze pistes d'optimisation.

C'est exactement la faute que ce rapport dénonce par ailleurs : optimiser une métrique sans avoir établi la décision qu'elle alimente. Le projet a passé l'essentiel de son effort à améliorer un chiffre dont personne n'avait vérifié l'utilité.

---

## 8ter. L'usage qui reste : unifier un axe, pas combler un trou

Le §8bis a réfuté l'usage visé (alimenter le PEP) et le §8 a montré qu'il n'y a **pas de trou de couverture** à combler sur les CVE. Reste un usage, un seul, et ce n'est pas celui du cadrage initial.

### La population qui n'a réellement aucun typage

Dans une plateforme de gestion des vulnérabilités, une grande partie des findings **ne vient pas d'une CVE** : résultats SAST/DAST, défauts de configuration, findings propriétaires de scanners, règles maison. Ceux-là n'ont **aucun CWE et aucun NVD dont hériter**. C'est la seule population où il n'existe rien - par opposition aux CVE, où le CNA fournit déjà le CWE à la publication.

### La valeur n'est pas de combler, c'est d'unifier

Aujourd'hui un finding de scanner ne peut pas figurer dans un rapport "par type de faille" à côté des findings CVE : il n'a pas d'étiquette. Les typer **à la famille** met les deux populations **sur le même axe** de regroupement et de reporting.

➜ **C'est ça le produit - pas le CWE précis.**

### Deux sorties, deux contrats, qu'on ne mélange pas

| Sortie | Contrat mesuré | Usage | Réserve |
|---|---|---|---|
| **Famille** (10 `Pillar`) | **89,2 % / 86,6 %** | tri, regroupement, reporting | - |
| **CWE fin** (71 classes) | **81,5 % / 57,9 %** | choisir une remédiation | MITRE ne nomme une stratégie vérifiable que sur **19-33 %** du catalogue, **sans gradient** (§7ter) - un CWE plus fin ne débloque pas la remédiation automatisable |

### Ce qu'il faut mesurer avant de promettre quoi que ce soit

Nous avons testé si le modèle survit au format d'un finding, et **notre première prédiction était fausse**. Le résultat corrigé (§8) : **la longueur n'est pas le problème, la densité l'est**. Un texte de 58 caractères qui énonce le type donne le même contrat qu'une description de 653.

➜ **Le premier test n'est donc pas la couverture, c'est la correspondance de vocabulaire.** Il faut vérifier qu'un scanner disant "Missing X-Frame-Options header" atterrit dans la bonne famille, alors que le modèle n'a appris que des formulations NVD.

Protocole minimal, par ordre de coût :

1. **Sans étiquettes** : passer un échantillon de findings réels et mesurer le **taux d'abstention** et la **distribution des familles prédites**. Une distribution aberrante - tout dans une seule famille - suffit à disqualifier la piste.
2. **Avec étiquettes** : un jeu de quelques centaines de findings typés à la main permet de mesurer le contrat réel.

La piste est **peu coûteuse à tester**, et un échec y serait informatif plutôt que dangereux - à condition de respecter les contraintes ci-dessous.

### Contraintes d'intégration

- **Ne jamais écraser le CWE du feed.** Champ distinct, avec **provenance**, **confiance** et **version de modèle**. Un utilisateur doit pouvoir distinguer "le CNA l'a déclaré" de "un modèle l'a estimé".
- **Feature flag**, comme pour le PEP.

### Un acquis qui n'est pas un produit, mais qui vaut d'être remonté

Le bruit d'étiquetage, mesuré par deux voies indépendantes (§8) : **51 %** des erreurs du modèle tombent entre types parent et enfant, et **18,6 %** des CWE sont révisés après publication, dont **un tiers change de famille**.

➜ **Avertissement pour la plateforme, indépendamment de ce modèle : ne pas bâtir de logique fine ni de règle automatique sur la valeur exacte d'un CWE - même quand elle vient du CNA.** Elle est instable, et l'instabilité porte souvent sur le niveau de détail plutôt que sur la nature de la faille.

---

## 9. Suites possibles

**Avant toute chose : cadrer un usage réel.** Le §8bis a montré que l'usage visé n'existe pas. Toute suite technique est prématurée tant qu'on n'a pas identifié quelqu'un qui a besoin du champ CWE - et pour quoi faire. C'est une conversation, pas une expérience.

**Applicable immédiatement, côté PEP** : retirer `primary_cwe` (+0.012 de PR, 121 dimensions en moins), après validation croisée temporelle.

Si un usage se confirme, par ordre de rentabilité :

1. ~~Entraîner sur le jeu complet~~ **fait, et c'est le levier qui rapporte** : +3,2 points de précision et +2,5 de couverture au protocole honnête, soit ~1 700 CVE justes de plus par an (§7). Le premier bilan concluait l'inverse, à tort.
2. **Améliorer le classement de confiance** - le facteur limitant identifié. Un ensemble des trois modèles déjà entraînés coûte une heure d'inférence et attaque directement ce point.
3. **Un vrai schedule de learning rate** (warmup puis décroissance linéaire, la recette canonique de BERT) plutôt qu'un 2e-5 constant avec réduction réactive. Sur ce projet, chaque baisse déclenchée par `ReduceLROnPlateau` a débloqué un gain - signe qu'un schedule anticipé ferait mieux.
4. ~~Une métrique hiérarchique exploitant l'arbre CWE officiel~~ **fait** (§7bis) : la granularité famille, tirée des 10 `Pillar` officiels, apporte +13 400 CVE nommées par an. Reste à trancher si l'usage aval s'en contente (§7ter).
5. **Annoter manuellement une centaine de CVE enrichies** pour valider - ou invalider - la valeur métier annoncée.
6. **Mesurer sur des findings de scanner** (SAST/DAST, défauts de configuration), la seule population où ce modèle aurait un usage non couvert par le NVD. Y mesurer la **couverture** d'abord (§8).
7. **Le protocole "CVE fraîches"** : prédire depuis la description telle qu'elle était à la publication, comparer au CWE finalement retenu. L'API NVD `cvehistory/2.0` fournit les deux - vérifié et utilisé ponctuellement ici, mais pas à l'échelle du jeu de test (plusieurs heures de collecte).

---

## 10. Reproduire

```bash
uv sync

# Données : la configuration de production
uv run python dataset_cwe.py --min-count 500 --out-dir data/cwe71

# Le plancher de comparaison (33 s de CPU)
uv run python baseline_tfidf.py --data-dir data/cwe71

# Le notebook (~2 h de GPU)
uv run python build_cwe_nb.py
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 07_cwe_finetuning.ipynb

# L'évaluation en conditions réelles
uv run python contrat_precision.py --data-dir data/cwe71 \
    --model distilbert-base-uncased \
    --weights best_distilbert-base-uncased_finetune_71cl.weights.h5 \
    --run-id distilbert-base-uncased_finetune_71cl
uv run python seuils_par_classe.py --data-dir data/cwe71 \
    --model distilbert-base-uncased \
    --weights best_distilbert-base-uncased_finetune_71cl.weights.h5 \
    --run-id distilbert-base-uncased_finetune_71cl

# Le bilan des pistes, sous LES DEUX protocoles (§7)
uv run python bilan_pistes.py

# La douzieme piste : granularite famille (§7bis)
uv run python dataset_famille.py
uv run python entrainement_famille.py --data-dir data/cwe_famille --n-train 40000 --epochs 20
uv run python comparaison_granularite.py

# L'actionnabilite : mitigations MITRE par niveau (§7ter)
uv run python actionnabilite_cwe.py

# La derive de la verite terrain, via l'API NVD (§8) -- lent, limite de debit
uv run python derive_etiquette.py --n 120
uv run python derive_description.py

# Le notebook de soutenance (lecture seule des artefacts, quelques secondes)
uv run python build_soutenance_nb.py
uv run jupyter nbconvert --to notebook --execute --inplace llm_fine_tuning_cwe.ipynb
```

Toutes les valeurs citées dans ce rapport proviennent de `runs/*.json`, produits par ces commandes. Le champ `provenance` de chaque fichier distingue ce qui a été mesuré sous le protocole final de ce qui a été transcrit de sessions antérieures.

**Les deux notebooks, et leurs rôles distincts :**

| | |
|---|---|
| `07_cwe_finetuning.ipynb` | la **preuve** - le code d'entraînement, exécuté sur GPU, avec ses sorties |
| `llm_fine_tuning_cwe.ipynb` | l'**argument** - 76 cellules, 11 figures, en lecture seule des artefacts, s'exécute en quelques secondes |

**Documentation complémentaire** : `docs/cwe-cadrage.md` (décisions et résultats détaillés), `docs/cwe-etapes.md` (parcours pédagogique), `docs/etat-avancement.md` (état du projet).
