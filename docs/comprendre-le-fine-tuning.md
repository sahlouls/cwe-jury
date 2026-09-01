# Comprendre le fine-tuning — de zéro, sans rien supposer

> Ce document ne dit pas *comment lancer* le projet (c'est `cwe-etapes.md`). Il explique **ce qui se
> passe réellement** à chaque étape, et **pourquoi** chaque choix a été fait. Écrit pour être lu
> d'un bout à l'autre avant une soutenance, y compris sur les points qu'on croit avoir compris.
>
> À la fin : la liste des questions qu'un jury pose, avec la réponse.

---

## 1. Le problème, en une phrase

On a la **description textuelle** d'une vulnérabilité, écrite en anglais par un humain :

> *« A buffer overflow in the HTTP parser of Acme Router 3.2 allows a remote attacker to execute
> arbitrary code via a crafted Content-Length header. »*

On veut en déduire son **type de faille** : ici, un débordement de tampon. C'est une
**classification multi-classe** — une entrée (du texte), une sortie parmi N étiquettes possibles.

Rien de plus. Tout le reste du document explique comment on y arrive.

---

## 2. Ce qu'est un modèle « pré-entraîné » — la seule chose à vraiment comprendre

C'est le concept central, et c'est celui qu'on explique le plus mal.

### Le problème qu'on n'a pas les moyens de résoudre

Notre jeu d'entraînement contient **40 000 descriptions**. Pour qu'un réseau apprenne à classer ces
textes en partant de rien, il devrait apprendre **simultanément** :

1. l'anglais — la grammaire, le vocabulaire, le fait que « overflow » et « débordement » relèvent du
   même champ, que « attacker » désigne un agent, que « crafted » qualifie une entrée malveillante ;
2. **et** la tâche — traduire cette compréhension en un choix parmi 71 types.

Avec 40 000 exemples, il échouerait aux deux. Apprendre l'anglais demande des **milliards** de mots.

### Ce que quelqu'un d'autre a payé pour nous

**DistilBERT** a été entraîné en amont sur Wikipedia et BookCorpus, sur une tâche qui n'a rien à voir
avec la nôtre : *deviner un mot masqué dans une phrase*.

```
"The attacker sent a [MASK] request to the server."
                       ↑ le modèle doit deviner : "malicious" ? "crafted" ? "HTTP" ?
```

Répétée des milliards de fois, cette tâche stupide force le modèle à construire une représentation
interne du langage — parce qu'on **ne peut pas** deviner le mot masqué sans avoir capté la syntaxe,
le sens, et le contexte.

> **C'est ça, le pré-entraînement** : un modèle qui ne sait rien de notre problème, mais qui sait
> lire. On récupère la lecture, et on n'apprend que le problème.

C'est **le transfer learning** : le principe de réutiliser ce savoir. Le **fine-tuning** est la
*méthode* pour l'adapter — on y vient au §6.

### L'ordre de grandeur, qui rend la chose concrète

| | Paramètres | Origine |
|---|---|---|
| Le corps de DistilBERT (« backbone ») | **66 362 880** | pré-entraîné, réutilisé tel quel |
| Notre couche de sortie (« tête ») | **54 599** | ajoutée par nous, apprise de zéro |

**99,9 % du réseau ne vient pas de nous.** Notre contribution propre représente moins de 55 000
paramètres — et nos 40 000 exemples suffisent largement à les apprendre.

---

## 3. Le tokenizer — pourquoi le texte doit devenir des nombres

Un réseau de neurones ne fait que des multiplications de matrices. Il ne peut pas manger du texte.
Le **tokenizer** est le pont.

### Ce qu'il fait, concrètement

Il découpe le texte en **sous-mots** (« tokens ») puis remplace chacun par son numéro dans un
dictionnaire de ~30 000 entrées :

```
"A buffer overflow in the HTTP parser"
   ↓  découpage en sous-mots
["[CLS]", "a", "buffer", "over", "##flow", "in", "the", "http", "par", "##ser", "[SEP]"]
   ↓  conversion en identifiants
[101, 1037, 17698, 2058, 12314, 1999, 1996, 8299, 11968, 8043, 102]
```

**Pourquoi des sous-mots et pas des mots ?** Parce qu'un dictionnaire de mots entiers serait
infini — chaque nom de produit, chaque version, chaque faute de frappe créerait un mot inconnu.
Avec des sous-mots, « overflow » inconnu se décompose en « over » + « ##flow », tous deux connus.
Le `##` signifie « ce morceau se colle au précédent ».

### Deux tokens spéciaux qu'il faut connaître

| Token | Rôle |
|---|---|
| `[CLS]` | ajouté **au début** de chaque texte. C'est un emplacement réservé, qu'on va utiliser au §4. |
| `[SEP]` | marque la fin du texte. |

### Deux paramètres du projet, et leur conséquence

**`max_length = 192`** — on tronque à 192 tokens. Pourquoi ? La mémoire du GPU. Le coût de
l'attention (le mécanisme interne du transformer) croît **au carré** de la longueur : passer de 192
à 384 tokens quadruple le coût. 192 couvre la grande majorité de nos descriptions ; les plus longues
sont coupées, et on l'assume.

**`padding='max_length'`** — tous les textes sont complétés à exactement 192 tokens, avec un token
de remplissage. Pourquoi pas juste la longueur du plus long de chaque lot ? Voir le §10, c'est un
piège qui a coûté du temps.

### `attention_mask` — le détail qu'on oublie d'expliquer

Le tokenizer renvoie **deux** tableaux :

```
input_ids      = [101, 1037, 17698, ..., 102,  0,  0,  0]   ← les identifiants, complétés par des 0
attention_mask = [  1,    1,     1, ...,   1,  0,  0,  0]   ← 1 = vrai token, 0 = remplissage
```

Le masque dit au modèle : *« ignore les positions à 0, ce n'est pas du texte »*. Sans lui, le modèle
traiterait le remplissage comme du contenu et le sens serait dilué.

---

## 4. Ce que le transformer produit, et pourquoi on prend `[CLS]`

Le backbone reçoit 192 tokens et renvoie **192 vecteurs de 768 nombres** — un par position.

```
entrée  :  192 identifiants
          ↓  6 couches de transformer
sortie  :  192 vecteurs × 768 dimensions
```

Chaque vecteur est une représentation de son token **en contexte** : le vecteur du mot « parser »
n'est pas le même dans « HTTP parser » et dans « SQL parser ». C'est là toute la valeur du
transformer par rapport aux méthodes plus anciennes.

### Mais nous, on veut UN vecteur, pas 192

On doit classer la description **entière**. Il faut donc résumer les 192 vecteurs en un seul.

**On prend celui de la position 0 — le vecteur du token `[CLS]`.**

Pourquoi celui-là ? Parce que `[CLS]` n'est pas un vrai mot : il n'a pas de sens propre. Pendant le
pré-entraînement, le modèle a appris à s'en servir comme d'un **emplacement d'agrégation** — un
endroit où déposer un résumé de toute la séquence. Le mécanisme d'attention lui permet de regarder
tous les autres tokens, donc son vecteur final « voit » la phrase entière.

```python
cls = backbone(ids, attention_mask=mask).last_hidden_state[:, 0, :]
#                                                          │  │  └── les 768 dimensions
#                                                          │  └───── position 0 = [CLS]
#                                                          └──────── tous les exemples du lot
```

> **Question de jury fréquente** : « pourquoi pas la moyenne des 192 vecteurs ? » Réponse honnête :
> ça marche aussi, c'est une alternative légitime (*mean pooling*), parfois même meilleure. On a
> pris `[CLS]` parce que c'est la convention BERT et le choix du cours. **Nous ne l'avons pas
> comparé** — c'est une ablation qu'on n'a pas faite, et il vaut mieux le dire que l'inventer.

---

## 5. La tête — les 55 000 paramètres qui sont à nous

Sur le vecteur `[CLS]` de 768 dimensions, on branche deux couches :

```
vecteur [CLS] (768)
      ↓
  Dropout(0.3)          ← régularisation
      ↓
  Dense(71, softmax)    ← la couche de décision
      ↓
71 probabilités qui somment à 1
```

### `Dense(71)` — ce que ça calcule

Une simple multiplication matricielle : `768 → 71`. Soit `768 × 71 + 71 = 54 599` paramètres. Chaque
sortie est une combinaison pondérée des 768 dimensions. On apprend ces poids.

### `softmax` — et sa propriété qui décide de tout le projet

Le softmax transforme 71 nombres quelconques en 71 probabilités **qui somment à 1**.

**Retenez cette contrainte, c'est le résultat central du projet** (§11) : puisque la somme vaut
toujours 1, le modèle **ne peut jamais dire « aucune des 71 »**. Il doit répartir sa masse. Face à
une vulnérabilité d'un type qu'il n'a jamais appris, il désignera quand même un gagnant.

### `Dropout(0.3)` — pourquoi éteindre 30 % du réseau

À l'entraînement, le dropout met à zéro 30 % des valeurs du vecteur, **au hasard, à chaque passage**.
Cela paraît absurde et c'est pourtant utile : le réseau ne peut plus se reposer sur une dimension en
particulier, donc il apprend des représentations plus redondantes, donc il **généralise mieux**.
Le dropout est désactivé automatiquement à l'inférence.

C'est notre **seul** garde-fou contre le surapprentissage, avec l'arrêt anticipé (§7).

---

## 6. Fine-tuning contre gel — l'ablation demandée par le sujet

Une fois le backbone récupéré, il reste **une** décision, et c'est celle que le sujet Alyra demande
de comparer.

| | **Fine-tuning complet** | **Extraction de caractéristiques** (gel) |
|---|---|---|
| Le backbone | est **réentraîné** avec la tête | est **gelé**, ses poids ne bougent plus |
| Ce qui apprend | 66 M de paramètres | la tête seule (55 k) |
| Taux d'apprentissage | **2 × 10⁻⁵** | **1 × 10⁻³** |
| Coût par époque | élevé | faible |
| **Résultat mesuré (macro-F1)** | **0,467** | **0,170** |

### Pourquoi deux taux d'apprentissage différents ? (question de jury quasi certaine)

Ce n'est **pas** un réglage empirique, c'est une conséquence logique du mode.

**En fine-tuning**, les poids de départ sont *déjà bons* — ils encodent une compréhension de
l'anglais acquise sur des milliards de mots. Un taux élevé les modifierait brutalement dès les
premiers lots et **détruirait ce savoir avant qu'il ne serve**. C'est l'**oubli catastrophique**
(*catastrophic forgetting*). On avance donc à très petits pas. `2e-5` est la valeur de référence de
l'article BERT original, et c'est celle recommandée par le cours.

**Backbone gelé**, la situation est inverse : la tête part de poids **aléatoires**, elle n'a rien à
préserver, et elle est la seule à apprendre. Un taux 50 fois plus élevé est approprié — avec `2e-5`
elle n'aurait pas convergé dans le budget d'époques imparti.

### Pourquoi le fine-tuning gagne aussi largement (facteur 2,8)

Le backbone gelé produit des représentations **génériques**, apprises sur Wikipedia. Elles séparent
bien les grands thèmes du langage courant, mais pas les nuances qui nous intéressent : distinguer un
débordement de tampon *en écriture* d'un débordement *en lecture* demande une sensibilité que rien,
dans le pré-entraînement généraliste, n'a eu de raison de développer.

Le fine-tuning autorise le backbone à **réorganiser** ses représentations pour notre tâche. C'est
tout l'écart.

---

## 7. L'entraînement — ce qui se passe réellement

### La boucle, à chaque lot de 16 exemples

1. **Passe avant** : les 16 descriptions traversent le réseau, on obtient 16 × 71 probabilités.
2. **Perte** : on compare aux bonnes réponses avec l'**entropie croisée**. Elle vaut ~0 si le modèle
   met toute sa probabilité sur la bonne classe, et grimpe vite sinon.
3. **Passe arrière** (rétropropagation) : on calcule, pour chaque paramètre, dans quel sens le
   modifier pour faire baisser la perte.
4. **Mise à jour** : l'optimiseur Adam applique la correction, dosée par le taux d'apprentissage.

Une **époque** = un passage sur tout le jeu d'entraînement. On en fait jusqu'à 10.

### Les trois rappels (« callbacks ») et leur rôle

| Rappel | Ce qu'il fait | Pourquoi c'est indispensable |
|---|---|---|
| `ModelCheckpoint` | sauvegarde les poids de la **meilleure** époque | la dernière époque n'est pas la meilleure — voir plus bas |
| `EarlyStopping` (patience 3) | arrête si le macro-F1 de validation ne progresse plus pendant 3 époques | évite de brûler du GPU à surapprendre |
| `ReduceLROnPlateau` (patience 2) | divise le taux d'apprentissage par 2 quand ça stagne | des pas plus fins quand on approche d'un optimum |

### Le surapprentissage, vu dans les chiffres du projet

À la fin de l'entraînement, la perte d'**entraînement** continue de descendre (0,80 → 0,17) tandis
que la perte de **validation** remonte (1,57 → 1,68). C'est la signature classique : le modèle
**mémorise** ses exemples au lieu d'apprendre des règles générales.

C'est exactement pour ça qu'on garde les poids de la meilleure époque et non de la dernière.

---

## 8. Le déséquilibre des classes — poids et macro-F1

### Le problème, chiffré

Dans notre jeu d'entraînement :

| | Classe | Exemples |
|---|---|---|
| La plus fréquente | CWE-79 (XSS) | **29 705** |
| La plus rare | CWE-98 | **78** |

**Un rapport de 381 pour 1.**

### Pourquoi un modèle non corrigé abandonne les classes rares

L'optimiseur cherche à faire baisser la perte **moyenne**. Or ignorer complètement CWE-98 (78
exemples sur 40 000) coûte presque rien à cette moyenne. Le modèle a donc **intérêt** à ne jamais la
prédire.

### Les poids de classe

On donne à chaque exemple un poids **inversement proportionnel** à la fréquence de sa classe : une
erreur sur une classe rare coûte plus cher. Concrètement, `compute_class_weight('balanced')`.

**Avec un plafond, et c'est un choix délibéré.** Le rapport brut de 381 appliqué tel quel rendrait la
mise à jour instable : quelques exemples rares domineraient chaque lot. On plafonne à **10**, ce qui
échange un peu de rappel sur les classes très rares contre un entraînement qui converge.

### Le macro-F1 — et pourquoi pas l'exactitude

**L'exactitude** (part de bonnes réponses) est dominée par les classes fréquentes. Un modèle qui
prédirait toujours CWE-79 obtiendrait déjà 17,5 % d'exactitude sans rien avoir appris.

**Le macro-F1** calcule le F1 de chaque classe **séparément**, puis en fait la moyenne simple.
Chaque classe pèse pareil, qu'elle ait 29 705 ou 78 exemples. Un modèle qui abandonne les rares le
paie immédiatement.

> **Rappel de vocabulaire, souvent demandé.** *Précision* = parmi mes réponses « CWE-89 », combien
> sont justes ? *Rappel* = parmi les vrais CWE-89, combien ai-je trouvés ? Le *F1* est leur moyenne
> harmonique — elle punit le déséquilibre entre les deux.

C'est le macro-F1 qui pilote `ModelCheckpoint` et `EarlyStopping`. **Un piège s'y cache**, voir §10.

---

## 9. Le découpage temporel — un choix qui dégrade nos chiffres, assumé

| Rôle | Période | Sert à |
|---|---|---|
| Entraînement | ≤ 2023 | apprendre |
| Validation | 2024 | choisir l'époque, calibrer le seuil |
| Test | 2025 | mesurer, **une seule fois** |

### Pourquoi pas un tirage aléatoire ?

Parce que **les CVE arrivent par grappes**. Une même campagne de recherche publie des dizaines de
vulnérabilités quasi identiques, dans le même produit, décrites dans les mêmes termes, à quelques
jours d'intervalle.

Un tirage aléatoire éclate ces grappes entre entraînement et test. Le modèle a alors vu, à
l'entraînement, des **quasi-jumelles** des CVE sur lesquelles on l'évalue. Le score ne mesure plus
sa capacité à généraliser mais sa capacité à reconnaître ce qu'il a déjà lu.

> **C'est une fuite de données** (*data leakage*), et elle est **invisible** : rien dans le
> protocole ne signale l'anomalie. Les chiffres sont simplement trop beaux.

### Le coût, assumé

Nos scores sont **plus bas** qu'ils ne le seraient avec un tirage aléatoire, parce qu'ils intègrent
la dérive du vocabulaire entre 2023 et 2025. C'est un choix : un chiffre plus bas et vrai plutôt
qu'un chiffre plus haut et trompeur.

---

## 10. Les quatre pièges techniques du projet

Ceux-là échouent de façon peu lisible, et ils ont tous coûté du temps.

### 10.1 `TF_USE_LEGACY_KERAS=1` avant tout import

TensorFlow 2.16+ embarque Keras 3, mais les modèles TensorFlow de `transformers` ont été écrits pour
Keras 2. Sans cette variable, la construction du modèle échoue avec `Data of type KerasTensor is not
allowed`. Elle doit être posée **avant** `import tensorflow`, car c'est à l'import que TF choisit son
implémentation.

### 10.2 `dtype='float32'` sur la couche de sortie

L'entraînement tourne en **précision mixte** (`mixed_float16`) : les calculs internes se font en
16 bits, ce qui divise par deux la mémoire et rend le fine-tuning possible sur 6 Go.

Mais le **softmax en 16 bits déborde** dès que les logits s'écartent un peu : la perte devient `NaN`,
**sans message d'erreur**. Forcer la sortie en float32 corrige le problème sans annuler le gain
mémoire, qui vient des couches internes.

### 10.3 `padding='max_length'`, surtout pas `padding=True`

Avec `padding=True`, le tokenizer complète chaque lot à la longueur de **son** plus long élément —
donc à une longueur *variable*. Or notre modèle déclare `Input(shape=(192,))`, une forme **fixe**.

Résultat : ça fonctionne tant qu'un lot contient une description longue, et **échoue dès qu'un lot
n'en contient aucune**. Un bug qui ne se déclenche que sur certains lots — le pire genre.

### 10.4 Le piège qui a fait échouer le premier run à 71 classes

Celui-ci mérite d'être raconté, un jury apprécie.

La taille de l'échantillon de validation était fixée à **3 000**. Sur 10 classes, c'est confortable
(~300 par classe). Sur **71** classes, cela laissait **moins de 10 exemples** pour 24 d'entre elles.

Le macro-F1 calculé sur si peu d'exemples est **extrêmement bruité** — et ce macro-F1 pilotait
`ModelCheckpoint` **et** `EarlyStopping`. On sauvegardait donc les poids d'une époque choisie au
hasard, et on arrêtait l'entraînement sur du bruit.

**Correction** : la taille de validation est devenue `max(3000, 150 × nb_classes)`. Le macro-F1 est
passé de **0,33 à 0,467**.

> La leçon : une métrique de pilotage doit être calculée sur assez de données pour être stable.
> Sinon on n'optimise pas, on tire au sort.

---

## 11. Le résultat central du projet — et il est contre-intuitif

C'est ce qu'il faut savoir raconter jeudi, plus que n'importe quel détail technique.

### L'observation

| Configuration | Exactitude « labo » | Précision réelle |
|---|---|---|
| **10 classes** | **96,9 %** | **39,8 %** |
| 71 classes | 59,7 % | 59,5 % |

La configuration qui affiche 37 points d'exactitude **en plus** est celle qui a la **pire** valeur
réelle. Le classement s'inverse.

### Le mécanisme, en trois étapes

1. **Le jeu de test est filtré.** Il ne contient que des CVE dont le vrai type appartient aux classes
   du modèle. C'est cohérent — on ne peut pas reprocher au modèle de rater une classe qu'on ne lui a
   pas apprise.
2. **En production, ce filtre n'existe pas.** Le modèle reçoit *toutes* les CVE, y compris celles
   dont le vrai type est **hors de son périmètre**. Avec 10 classes, c'est 59 % du flux.
3. **Le softmax ne peut pas s'abstenir** (§5). Face à une CVE hors périmètre, il désigne quand même
   un gagnant. Il donne un nom. Un nom **faux**.

**Conséquence mesurée** : sur la configuration à 10 classes, le taux de faux nommage hors périmètre
est de **100 %**. Le modèle nomme un CWE sur *chaque* CVE qu'il ne connaît pas, à n'importe quel
niveau de confiance.

### La leçon, formulée pour un jury

> **L'exactitude sur un jeu de test filtré ne prédit pas la valeur en production — elle peut la
> prédire à l'envers.** Ce n'est pas un défaut de notre modèle, c'est un défaut du **protocole
> d'évaluation standard** : il répond à « le modèle classe-t-il bien ce qu'il a appris ? », alors que
> la production pose « que se passe-t-il face à ce qu'il n'a pas appris ? ».

---

## 12. La réponse honnête : un contrat, pas une exactitude

### Ce qu'un utilisateur peut réellement demander

Pas « quelle est ton exactitude ? », mais :

> **« Quand tu me donnes un type, à quelle fréquence as-tu raison — et sur quelle part des CVE
> acceptes-tu de te prononcer ? »**

**Deux nombres, pas un.** Et ils s'échangent : le modèle produit une probabilité, on ne répond
qu'au-dessus d'un **seuil de confiance**. Seuil haut, on répond rarement mais bien ; seuil bas, on
répond souvent et on se trompe davantage.

### Le piège méthodologique, en trois niveaux

C'est subtil et le projet s'y est fait prendre. Trois façons de mesurer, deux sont fausses :

| Niveau | Comment le seuil est choisi | Ce qu'on annonce |
|---|---|---|
| **1** | cherché **sur l'année de test** | 90 % — garanti d'avance, on a choisi la réponse |
| **2** | calibré sur 2024, mais la *cible* choisie en regardant 2025 | 91,4 % — la fuite est déplacée d'un cran, pas supprimée |
| **3** | cible **fixée d'avance**, calibrée sur 2024, appliquée telle quelle | **78,3 %** — le seul défendable |

Le niveau 3 est le moins flatteur et le seul honnête. L'écart entre le 90 % visé et le 78 % obtenu a
un nom : **la dérive temporelle**. Le seuil qui garantissait 90 % sur 2024 n'en garantit plus que
78 % sur 2025 — le vocabulaire des vulnérabilités a bougé en un an.

**Conséquence opérationnelle** : un système en production doit se **recalibrer périodiquement**.

### Le contrat livré

> Sur les CVE de 2025, avec un seuil fixé **avant** de les voir, le modèle se prononce sur **57,9 %**
> d'entre elles et a raison **81,5 %** du temps. Soit ~25 000 CVE par an typées automatiquement,
> dont ~20 400 correctement.
>
> *(chiffres du modèle entraîné sur les 218 000 CVE, celui qu'on livre)*

---

## 13. Les questions que le jury va poser

### « Pourquoi DistilBERT et pas BERT ? »

DistilBERT est une version **distillée** de BERT : 40 % plus léger, 60 % plus rapide, pour ~97 % des
performances. Sur un GPU de portable de 6 Go, c'est ce qui rend le fine-tuning possible. C'est une
contrainte matérielle assumée, pas un choix de performance.

### « Pourquoi pas un LLM avec la taxonomie dans le prompt ? »

Légitime, et non testé. Un LLM généraliste avec les 71 définitions CWE en contexte pourrait
fonctionner. Ce qu'on peut dire : notre modèle tourne **hors ligne**, coûte quelques millisecondes
par CVE, et son comportement est reproductible. Un appel LLM par CVE sur 45 000 CVE/an, c'est un
autre budget et une dépendance externe. Mais **nous ne l'avons pas comparé** — le dire.

### « Le transformer valait-il le coup face à TF-IDF ? »

Oui, et c'est mesuré : macro-F1 **0,467** contre **0,424** pour une régression logistique sur TF-IDF.
Soit **+10 %**. Mais TF-IDF s'entraîne en **33 secondes de processeur** contre plus d'une heure de
GPU. C'est un arbitrage à assumer explicitement, pas une victoire écrasante.

L'écart se creuse quand la tâche est difficile : +1 % sur 10 classes, +12 % sur 121. Cohérent avec
l'intérêt du pré-entraînement — il sert surtout quand les données de la tâche ne suffisent pas.

### « Vos erreurs sont-elles graves ? »

Non, et c'est mesuré. **51 %** de nos erreurs se produisent entre types **parent et enfant** de la
taxonomie MITRE (contre 3 % attendus au hasard), et seulement **0,5 %** entre types sans lien
(contre 14 % au hasard).

Le modèle hésite sur le **niveau de détail**, presque jamais sur la nature de la faille. Il ne prend
jamais une injection SQL pour un débordement de tampon. Exemple type : la vraie étiquette dit
CWE-74 (« injection »), le modèle répond CWE-89 (« injection SQL ») — qui en est un **cas
particulier**. Les deux réponses sont correctes ; une seule est celle de l'annotateur.

### « Combien de graines aléatoires ? »

**Une seule par configuration**, et c'est une limite réelle. On a mesuré ce qu'elle coûte : sur une
comparaison, l'écart a **changé de signe** selon le budget d'entraînement, avec une amplitude quatre
fois supérieure aux intervalles de confiance publiés. Des moyennes sur plusieurs graines n'auraient
pas été un raffinement — elles étaient nécessaires pour tout écart inférieur à quelques pour cent.

### « Qu'est-ce que vous referiez autrement ? »

Deux choses, et elles se ressemblent :

1. **Définir le contrat de service avant d'entraîner.** La première moitié du projet a optimisé le
   macro-F1, une métrique qui ne mesurait pas ce qui compte. Toutes les décisions de cette phase ont
   dû être réexaminées.
2. **Choisir la granularité de la cible à partir de l'usage, pas de la taxonomie.** On a retenu
   71 classes parce que c'est ce que les données offraient, jamais parce que quelqu'un en avait
   besoin.

---

## 14. Le schéma complet, à mémoriser

```
  description de la CVE (texte libre en anglais)
        │
        ▼
  ┌──────────────────────────────┐
  │ Tokenizer                    │  découpe en sous-mots, tronque/complète à 192
  └──────────────────────────────┘  → input_ids + attention_mask
        │
        ▼
  ┌──────────────────────────────┐
  │ DistilBERT pré-entraîné      │  66 M paramètres — RÉUTILISÉS
  │ 6 couches de transformer     │  (fine-tuning : ils bougent aussi, à 2e-5)
  └──────────────────────────────┘
        │  192 vecteurs × 768
        ▼   on ne garde que celui du token [CLS] (position 0)
  ┌──────────────────────────────┐
  │ Dropout(0,3)                 │  régularisation
  └──────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────┐
  │ Dense(71, softmax, float32)  │  la TÊTE — 55 k paramètres, apprise de zéro
  └──────────────────────────────┘
        │
        ▼
  71 probabilités qui somment à 1
        │
        ▼   seuil de confiance calibré sur l'année de validation
  réponse, ou ABSTENTION
```

---

## En une phrase, si on ne retient qu'une chose

> On n'a pas appris à un réseau à lire l'anglais — on a récupéré un réseau qui savait lire, et on lui
> a ajouté 55 000 paramètres pour qu'il choisisse parmi 71 types. Le plus dur n'a pas été de
> l'entraîner, mais de **mesurer honnêtement** ce qu'il valait.
