# NLP_classqiue

création de venv :

```powershell
py -m venv .venv
```

activé venv file

```powershell
.venv\Scripts\Activate.ps1
```

# **Partie Data**

Cette partie prépare les données du corpus **JNLPBA** pour l'entraînement d'un modèle de reconnaissance d'entités nommées biomédicales (**NER**), en suivant la méthode de l'article *Tuning Support Vector Machines for Biomedical Named Entity Recognition* (Kazama et al., 2002).

Elle produit :

- les **caractéristiques (features)** de chaque mot, sous forme de dictionnaires ;
- les **étiquettes BIO** avec *class splitting* (la classe `O` est éclatée selon la POS) ;
- la **fonction d'extraction** qui permet de transformer une nouvelle phrase en ces mêmes features.

---

### 1. Corpus

| Élément               | Valeur                                                        |
| ----------------------- | ------------------------------------------------------------- |
| Corpus                  | [bigbio/jnlpba](https://huggingface.co/datasets/bigbio/jnlpba) |
| Configuration utilisée | `jnlpba_source` (mots + étiquettes BIO déjà alignés)    |
| Splits                  | `train` et `validation`                                   |
| Types d'entités        | `protein`, `DNA`, `RNA`, `cell_type`, `cell_line`   |
| Taille du train         | 492 551 mots                                                  |
| Taille de la validation | 101 039 mots                                                  |

> La configuration `jnlpba_source` est utilisée à la place de `jnlpba_bigbio_kb`, car cette dernière ne contient que des positions de caractères (offsets) et obligerait à retokeniser et réaligner les entités à la main.

---

### 2. Structure des fichiers

```text
NLP_classique/
└── Data/
    ├── __init__.py        # expose build_features_data, prepare_split, extract_fn, extract_word_features
    ├── features.py        # extraction des features (extract_word_features, extract_fn)
    ├── prepare_data.py    # chargement du corpus, POS tagging, class splitting, construction de X et y
    └── test_data.py       # tests de cohérence (hors-ligne et en ligne)
```

| Fichier             | Rôle                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| `features.py`     | Définit `extract_word_features` (features d'un mot) et `extract_fn` (features de toute une phrase). |
| `prepare_data.py` | Charge JNLPBA, ajoute la POS à chaque mot, applique le class splitting et construit `X` / `y`.      |
| `test_data.py`    | Vérifie le format des données et la cohérence des features.                                           |

---

### 3. Installation

```powershell
pip install datasets nltk
```

Le modèle POS de NLTK (`averaged_perceptron_tagger_eng`) est téléchargé automatiquement au premier lancement. Le corpus JNLPBA est téléchargé depuis Hugging Face au premier appel de `build_features_data()`.

---

### 4. Utilisation

```python
from Data import build_features_data

(
    X_train_dict, y_train,
    X_validation_dict, y_validation,
    extract_fn,
) = build_features_data()
```

> Le premier appel peut durer plusieurs minutes (téléchargement du corpus, puis POS tagging de chaque phrase).

#### Contrat de sortie

| Variable              | Type           | Description                                                  |
| --------------------- | -------------- | ------------------------------------------------------------ |
| `X_train_dict`      | `list[dict]` | Un dictionnaire de features par mot du train.                |
| `y_train`           | `list[str]`  | Une classe par mot du train (avec class splitting).          |
| `X_validation_dict` | `list[dict]` | Idem pour la validation.                                     |
| `y_validation`      | `list[str]`  | Idem pour la validation.                                     |
| `extract_fn`        | `function`   | Transforme une phrase en liste de dictionnaires de features. |

`X_*` et `y_*` sont **aplatis** : une ligne par mot, toutes phrases confondues. `len(X) == len(y)` est toujours vérifié.

---

### 5. Format des données

#### Exemple pour la phrase « Number of glucocorticoid receptors in lymphocytes »

```python
X_train_dict = [
    ...,
    {
        "bias": 1.0,
        "w[0].word": "glucocorticoid",
        "w[0].pref3": "glu",
        "w[0].suff3": "oid",
        "w[0].pos": "NN",
        "w[-2].word": "Number",
        "w[-2].pos": "NN",
        "w[-1].word": "of",
        "w[-1].pos": "IN",
        "w[1].word": "receptors",
        "w[1].pos": "NNS",
        "w[2].word": "in",
        "w[2].pos": "IN",
        "prev_class": "O-IN",
    },
    ...
]

y_train = ["O-NN", "O-IN", "B-protein", "I-protein", "O-IN", "B-cell_type"]
```

#### Features extraites pour chaque mot

| Feature                                                      | Signification                                  |
| ------------------------------------------------------------ | ---------------------------------------------- |
| `bias`                                                     | Constante `1.0` (terme de biais).            |
| `w[0].word`                                                | Mot courant.                                   |
| `w[0].pref3`                                               | 3 premières lettres du mot courant.           |
| `w[0].suff3`                                               | 3 dernières lettres du mot courant.           |
| `w[0].pos`                                                 | Étiquette POS (Penn Treebank) du mot courant. |
| `w[-2].word`, `w[-1].word`, `w[1].word`, `w[2].word` | Mots dans une fenêtre de ±2.                 |
| `w[-2].pos`, `w[-1].pos`, `w[1].pos`, `w[2].pos`     | POS des mots de la fenêtre.                   |
| `prev_class`                                               | Classe du mot précédent.                     |

#### Valeurs spéciales

| Valeur      | Utilisation                                     |
| ----------- | ----------------------------------------------- |
| `<BOS>`   | Mot ou POS situé avant le début de la phrase. |
| `<EOS>`   | Mot ou POS situé après la fin de la phrase.   |
| `<START>` | `prev_class` du premier mot d'une phrase.     |

---

### 6. Étiquetage BIO et class splitting

Le corpus utilise la représentation **BIO** :

- `B-xxx` : début d'une entité de type `xxx` ;
- `I-xxx` : intérieur d'une entité de type `xxx` ;
- `O` : mot hors entité.

Pour réduire le coût d'entraînement et améliorer la précision (article, section 5.1), la classe `O` est **éclatée selon la POS du mot** :

| Mot            | POS | Classe d'origine | Classe finale |
| -------------- | --- | ---------------- | ------------- |
| Number         | NN  | `O`            | `O-NN`      |
| of             | IN  | `O`            | `O-IN`      |
| glucocorticoid | NN  | `B-protein`    | `B-protein` |
| receptors      | NNS | `I-protein`    | `I-protein` |

Les classes d'entités (`B-xxx`, `I-xxx`) ne sont **pas** modifiées. Il y a environ 45 sous-classes `O-POS` (une par étiquette Penn Treebank).

---

### 7. Fonction d'extraction

Deux fonctions sont disponibles :

```python
from Data import extract_fn, extract_word_features
```

**`extract_word_features(tokens, pos_tags, i, prev_class) -> dict`**
Features d'un seul mot (position `i`) avec la classe précédente connue.

**`extract_fn(tokens, pos_tags=None, previous_classes=None) -> list[dict]`**
Features de tous les mots d'une phrase. `previous_classes[i]` est la classe du mot qui précède le mot `i` (donc `previous_classes[0] = "<START>"`).

#### Utilisation en entraînement et en validation

Les classes précédentes sont connues (classes annotées) :

```python
tokens = ["Number", "of", "glucocorticoid", "receptors"]
pos    = ["NN", "IN", "NN", "NNS"]
prev   = ["<START>", "O-NN", "O-IN", "B-protein"]

features = extract_fn(tokens, pos, prev)
```

#### Utilisation en prédiction

La classe précédente dépend de la prédiction du modèle. Il faut donc avancer **mot par mot** :

```python
prev_class = "<START>"
predictions = []

for i in range(len(tokens)):
    feats = extract_word_features(tokens, pos, i, prev_class)
    prev_class = model.predict(vectorizer.transform([feats]))[0]
    predictions.append(prev_class)
```

Les POS d'une nouvelle phrase s'obtiennent avec :

```python
from Data.prepare_data import get_pos_tags
pos = get_pos_tags(tokens)
```

#### Évaluation

Pour calculer précision, rappel et F-score sur les entités, on ramène les sous-classes `O-xx` vers `O` :

```python
def merge_o(label):
    return "O" if label.startswith("O-") else label
```

---

### 8. Tests

À lancer depuis le dossier `NLP_classique` :

```powershell
python -m Data.test_data            # test hors-ligne (exemple de l'article)
python -m Data.test_data --online   # charge le vrai corpus JNLPBA
```

Le test hors-ligne vérifie le class splitting, les features du mot « glucocorticoid » et la valeur `<START>`. Le test en ligne vérifie les tailles, les types et la présence des clés obligatoires dans `X_train_dict` et `X_validation_dict`.

Résultat attendu du test en ligne :

```text
Train : 492551 492551
Validation : 101039 101039
Toutes les vérifications de base sont réussies.
```

---

### 9. Remarques

- Les données sont **aplaties** : les frontières de phrases ne sont conservées que par `<START>` (`prev_class` du premier mot) et par `<BOS>` / `<EOS>`.
- En entraînement, `prev_class` est la **vraie** classe du mot précédent (*teacher forcing*). En prédiction, c'est la classe prédite.
- Les features sont des dictionnaires : elles peuvent être transformées en matrice avec `sklearn.feature_extraction.DictVectorizer`.
- Le POS tagging utilise NLTK (étiquettes Penn Treebank, comme dans l'article).

### Référence

J. Kazama, T. Makino, Y. Ohta, J. Tsujii. *Tuning Support Vector Machines for Biomedical Named Entity Recognition*. Proceedings of the Workshop on Natural Language Processing in the Biomedical Domain, ACL, Philadelphia, 2002.

# Partie model

Cette partie entraîne le classifieur SVM de reconnaissance d'entités nommées biomédicales (NER) à partir des données préparées par la partie Data, puis l'expose sous la forme d'une fonction Python (`predict_entities`) et d'une API REST (FastAPI) consommée par la partie web. Elle suit la méthode de l'article *Tuning Support Vector Machines for Biomedical Named Entity Recognition* (Kazama et al., 2002).

Elle produit :

- un modèle SVM entraîné et son vectoriseur de features (`Model/artifacts/`) ;
- un protocole d'évaluation au niveau des entités (précision, rappel, F-score) ;
- la fonction `predict_entities(text)`, qui transforme un texte brut en liste `{token, tag}` ;
- l'API `POST /api/predict`, conforme au contrat convenu avec la partie web.

## 1. Place dans le projet

```
texte brut
   │  tokenisation (Penn Treebank) + POS tagging (NLTK)          ← Model/predict.py
   ▼
features de chaque mot (extract_word_features)                   ← Data/features.py
   ▼
SVM pairwise, décodage gauche → droite                           ← Model/decoding.py
   ▼
fusion O-POS → O, retrait de B-/I-                               ← Model/predict.py
   ▼
[{"token": "...", "tag": "PROTEIN"}, ...]                        ← API (Model/api.py)
```

| Partie | Entrée | Sortie |
|---|---|---|
| Data | corpus JNLPBA (Hugging Face) | `X_*_dict`, `y_*`, `extract_fn` |
| **Model** | `X_*_dict`, `y_*` de la partie Data | modèle entraîné + API `POST /api/predict` |
| Web | texte saisi par l'utilisateur | affichage des entités (appelle l'API) |

La partie model **réutilise** le code de la partie Data (`extract_word_features`, `get_pos_tags`, `START_CLASS`) à l'entraînement comme à la prédiction : les features d'une nouvelle phrase sont donc construites exactement comme celles du corpus.

## 2. Rappel de la méthode (article)

| Élément | Choix |
|---|---|
| Formulation | classification de chaque mot, avec son contexte, dans une classe BIO (`B-xxx`, `I-xxx`, `O`) |
| Class splitting | la classe `O` est éclatée selon la POS (`O-NN`, `O-IN`, ...), fait par la partie Data |
| Classifieur | SVM binaire à marge souple (paramètre `C`) |
| Noyau | polynomial `(s · x·x' + r)^d` |
| Multi-classe | **pairwise** (Kreßel) : `K(K-1)/2` SVM binaires, classe finale = vote majoritaire |
| Décodage | gauche → droite, **sans Viterbi** : la feature « classe précédente » est la classe prédite |
| Évaluation | précision / rappel / F-score au niveau des entités, et F-score d'identification (frontières seules) |

## 3. Structure des fichiers

```
NLP_classique/
├── Data/                     # partie Data (inchangée)
├── Model/
│   ├── __init__.py           # expose predict_entities
│   ├── config.py             # chemins des artefacts
│   ├── decoding.py           # reconstruction des phrases, décodage, fusion O-POS, réparation BIO
│   ├── evaluate.py           # métriques au niveau des entités
│   ├── train.py              # entraînement, évaluation, sauvegarde
│   ├── predict.py            # predict_entities(text) : texte brut -> entités
│   ├── api.py                # API FastAPI (POST /api/predict)
│   ├── mock_api.py           # API factice de test pour la partie web
│   └── artifacts/            # modèle et vectoriseur (non versionnés)
└── Web/                      # partie web
```

| Fichier | Rôle |
|---|---|
| `config.py` | Définit les chemins `svm_model.joblib`, `vectorizer.joblib` et `data_cache.joblib`. |
| `decoding.py` | `rebuild_sentences` (retrouve les phrases dans les données aplaties), `decode` (prédiction gauche → droite par lots), `merge_o` (`O-xx` → `O`), `repair_bio` (règle de l'article), `as_int32` (compatibilité scikit-learn). |
| `evaluate.py` | `evaluate` calcule précision, rappel, F-score, F-score d'identification et le rapport par classe. |
| `train.py` | Charge les données de la partie Data, entraîne le SVM, évalue, sauvegarde les artefacts. |
| `predict.py` | `predict_entities` : tokenisation, POS, décodage, fusion des classes, format de sortie. |
| `api.py` | Expose la fonction via FastAPI, avec CORS et un point `/health`. |

## 4. Installation

Python 3.11 ou 3.12 recommandé (`uvicorn 0.41.0` exige Python 3.10 ou plus).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip check
```

| Bibliothèque | Version | Usage |
|---|---|---|
| fastapi | 0.135.1 (fixée) | API |
| uvicorn | 0.41.0 (fixée) | serveur de l'API |
| pydantic | 2.12.5 (fixée) | validation des requêtes |
| scikit-learn | >=1.5,<2 | `DictVectorizer`, `SVC`, `LinearSVC` |
| numpy | >=1.26 | calcul numérique |
| joblib | >=1.4 | sauvegarde du modèle et du cache |
| seqeval | >=1.2.2 | métriques au niveau des entités |
| nltk | >=3.9 | tokenisation, POS tagging |
| datasets | >=4.0,<5 | chargement du corpus (partie Data) |

Toutes les commandes se lancent **depuis la racine du dépôt** (`NLP_classique/`), avec `python -m` : `Data` et `Model` sont des paquets de premier niveau.

## 5. Utilisation

### 5.1 Entraîner

```powershell
python -m Model.train --kernel linear                      # baseline rapide (one-vs-rest)
python -m Model.train --kernel poly --max-sentences 1000   # SVM pairwise, sous-ensemble
python -m Model.train --kernel poly --degree 2 --final     # modèle final
```

| Option | Défaut | Signification |
|---|---|---|
| `--kernel` | `linear` | `linear` : `LinearSVC` (one-vs-rest). `poly` : `SVC` à noyau polynomial (**pairwise**, comme l'article). |
| `--max-sentences` | tout | limite le nombre de phrases d'entraînement (mesure du temps, tests rapides). |
| `--C` | 1.0 | pénalité de la marge souple (`C` de l'article). |
| `--gamma` | 0.01 | `s` de l'article. |
| `--coef0` | 1.0 | `r` de l'article. |
| `--degree` | 2 | `d` de l'article (1, 2 ou 3 testés). |
| `--final` | désactivé | entraîne sur tout le train, évalue **une seule fois** sur `validation`, sauvegarde les artefacts. |

Le premier lancement télécharge le corpus et calcule les POS (plusieurs minutes). Le résultat des données est mis en cache dans `Model/artifacts/data_cache.joblib` : les lancements suivants démarrent immédiatement. **Si `Data/` change, supprimer ce cache et réentraîner.**

### 5.2 Prédire en Python

```python
from Model import predict_entities

predict_entities("Number of glucocorticoid receptors in lymphocytes.")
# -> [{"token": "Number", "tag": "O"}, ..., {"token": "lymphocytes", "tag": "CELL_TYPE"}, ...]
```

### 5.3 Lancer l'API

```powershell
uvicorn Model.api:app --host 127.0.0.1 --port 8000
```

Documentation interactive (Swagger) : `http://localhost:8000/docs`.

## 6. Fonctionnement détaillé

### 6.1 Entraînement (`train.py`)

1. **Chargement.** `build_features_data()` (partie Data) renvoie `X_train_dict`, `y_train`, `X_validation_dict`, `y_validation`. Le résultat est mis en cache.
2. **Reconstruction des phrases.** Les données sont aplaties (un mot par ligne). `rebuild_sentences` retrouve les phrases grâce à `prev_class == "<START>"` (premier mot d'une phrase) et lit le mot et sa POS dans `w[0].word` et `w[0].pos`.
3. **Découpage.** Sans `--final`, les 10 % dernières phrases du train forment le jeu de développement (découpage contigu, pour que les phrases d'un même résumé restent ensemble). Avec `--final`, tout le train sert à l'entraînement.
4. **Vectorisation.** `DictVectorizer` transforme les dictionnaires de features en matrice creuse binaire (une colonne par couple feature/valeur : le « bit-vector » de l'article). Les indices sont convertis en `int32` (`as_int32`), car les SVM de scikit-learn refusent l'`int64`.
5. **Entraînement** du SVM.
6. **Évaluation** par décodage gauche → droite (voir 6.3).
7. **Sauvegarde** (avec `--final`) de `vectorizer.joblib` et `svm_model.joblib`.

### 6.2 SVM pairwise

Avec `--kernel poly`, `sklearn.svm.SVC` (libsvm) entraîne un SVM binaire pour **chaque paire de classes** et prédit par vote majoritaire. Avec environ 55 classes (10 classes d'entités B-/I- + environ 45 classes `O-POS`), cela représente environ 1 485 SVM binaires. Le class splitting rend ces problèmes beaucoup plus petits que si la classe `O` restait unique : c'est l'apport principal de l'article.

Correspondance des hyperparamètres avec l'article :

| Article | scikit-learn | Valeur de départ |
|---|---|---|
| `C` (marge souple) | `C` | 1.0 |
| `s` | `gamma` | 0.01 |
| `r` | `coef0` | 1.0 |
| `d` | `degree` | 2 (l'article explore 1, 2, 3) |

`--kernel linear` utilise `LinearSVC` (one-vs-rest) : c'est une **baseline rapide, pas une version fidèle à l'article**. Pour un pairwise linéaire fidèle, utiliser `--kernel poly --degree 1`.

Le cache du noyau de libsvm est fixé à 2 000 Mo : prévoir cette mémoire en plus de celle des données. L'entraînement du SVM à noyau est super-linéaire en nombre d'échantillons : mesurer le temps avec `--max-sentences` croissant avant de lancer le jeu complet.

### 6.3 Décodage gauche → droite (`decode`)

La feature `prev_class` dépend de la prédiction précédente, donc un mot ne peut être classé qu'après son voisin de gauche. Pour rester rapide, `decode` traite **toutes les phrases en parallèle, pas à pas** : au pas `i`, il prédit le `i`-ème mot de chaque phrase en un seul appel au modèle. La classe précédente reste la **classe éclatée** (`O-NN`, ...), comme à l'entraînement.

À l'entraînement, `prev_class` est la vraie classe du mot précédent. À la prédiction, c'est la classe prédite : une erreur peut donc se propager au mot suivant. C'est la limite de l'absence de décodage de Viterbi, signalée par l'article.

### 6.4 Post-traitement

1. **Fusion** : `merge_o` remplace toute classe `O-xx` par `O`.
2. **Réparation BIO** : `repair_bio` applique la règle de l'article. Une entité commençant par `B-C` ne se termine que lorsqu'on rencontre un autre `B-` ou un `O`. Ainsi `O B-DNA I-RNA O` donne une seule entité de classe DNA.
3. **Format de sortie** : le préfixe `B-`/`I-` est retiré et le type est mis en majuscules : `PROTEIN`, `DNA`, `RNA`, `CELL_LINE`, `CELL_TYPE`, ou `O`.

## 7. Évaluation

| Élément | Valeur |
|---|---|
| Jeu de développement | derniers 10 % des phrases du train |
| Jeu final | `validation` de bigbio/jnlpba. D'après ses tailles (101 039 mots), il correspond au jeu d'évaluation officiel de JNLPBA 2004. **À utiliser une seule fois** (`--final`), jamais pour régler les hyperparamètres. |
| Niveau de mesure | entité complète (frontières **et** type corrects), via `seqeval` |
| Mesures | précision, rappel, F-score, rapport par classe |
| F-score d'identification | frontières seules, sans le type (équivalent de la ligne « All (id) » de l'article) |

Les classes `O-xx` sont ramenées à `O` avant l'évaluation, puis les séquences prédites sont réparées (voir 6.4).

### Résultats

À compléter après les entraînements (jeu de développement, puis `--final`) :

| Configuration | Précision | Rappel | F-score | F identification | Temps d'entraînement |
|---|---|---|---|---|---|
| Linéaire (`LinearSVC`, one-vs-rest) | … | … | … | … | … |
| Pairwise, noyau `d = 1` | … | … | … | … | … |
| Pairwise, noyau `d = 2` | … | … | … | … | … |
| Pairwise, noyau `d = 3` | … | … | … | … | … |
| Pairwise `d = 2`, sans class splitting | … | … | … | … | … |

Les scores ne sont pas comparables à ceux de l'article : celui-ci utilise GENIA 1.1 avec 24 classes, alors que ce projet utilise JNLPBA avec 5 types d'entités.

## 8. API REST

| Élément | Valeur |
|---|---|
| Méthode | `POST` |
| URL | `http://localhost:8000/api/predict` |
| En-tête | `Content-Type: application/json` |

Requête :

```json
{
  "text": "Number of glucocorticoid receptors in lymphocytes and NF-kappa B activation through CD28 requires 5-lipoxygenase."
}
```

Réponse (format) :

```json
{
  "entities": [
    {"token": "Number", "tag": "O"},
    {"token": "glucocorticoid", "tag": "PROTEIN"},
    {"token": "receptors", "tag": "PROTEIN"},
    {"token": "lymphocytes", "tag": "CELL_TYPE"}
  ]
}
```

| Champ | Signification |
|---|---|
| `entities` | un élément par token, dans l'ordre du texte, toutes phrases confondues |
| `token` | le mot, tel que découpé par le tokenizer Penn Treebank |
| `tag` | `PROTEIN`, `DNA`, `RNA`, `CELL_LINE`, `CELL_TYPE` ou `O` (hors entité) |

| Cas | Comportement |
|---|---|
| Texte vide ou espaces | `{"entities": []}` |
| Champ `text` absent | erreur `422` (validation FastAPI) |
| Modèle non entraîné | erreur `500` : lancer `python -m Model.train --kernel poly --final` |

Autres points : `GET /health` renvoie `{"status": "ok"}`, et le CORS est ouvert pour le développement. Exemple de test sous PowerShell :

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/predict `
  -ContentType "application/json" `
  -Body '{"text": "Number of glucocorticoid receptors in lymphocytes."}'
```

Le contrat est identique à celui de `mock_api.py` : la partie web peut passer de l'un à l'autre sans modification. Ne pas lancer les deux en même temps (même port 8000).

## 9. Artefacts

| Fichier | Contenu |
|---|---|
| `Model/artifacts/vectorizer.joblib` | `DictVectorizer` ajusté (vocabulaire des features) |
| `Model/artifacts/svm_model.joblib` | SVM entraîné |
| `Model/artifacts/data_cache.joblib` | données de la partie Data (évite de refaire le POS tagging) |

Ces fichiers sont volumineux et **non versionnés** (`Model/artifacts/` est dans `.gitignore`) ; ils se partagent par un lien de téléchargement.

Le vectoriseur et le modèle forment un tout : ils doivent venir du **même** entraînement et de la **même version** de `Data/features.py`. Si les features changent, supprimer le cache et réentraîner.

## 10. Écarts avec l'article

| Élément | Article | Ce projet |
|---|---|---|
| Corpus | GENIA 1.1 (670 résumés, 24 classes) | JNLPBA (2 000 résumés d'entraînement, 5 types) |
| Features | mot, POS, préfixe, suffixe, sous-chaîne, états HMM, word cache | version réduite (voir partie Data) : pas de HMM, pas de word cache |
| POS | tagger maison | NLTK |
| SVM | implémentation maison de SMO, entraînement parallèle, arrêt anticipé du vote | libsvm (scikit-learn), un seul cœur, vecteurs de support partagés |
| Multi-classe | pairwise | pairwise avec `--kernel poly` (le mode `linear` est one-vs-rest) |
| Décodage | gauche → droite, sans Viterbi | identique |
| Tokenisation à la prédiction | non applicable | Punkt + Penn Treebank (NLTK) |

## 11. Limites et pistes d'amélioration

- **Propagation d'erreurs** : pas de décodage global ; une mauvaise classe précédente peut entraîner l'erreur suivante.
- **Frontières B-/I- perdues dans l'API** : deux entités du même type qui se suivent apparaissent fusionnées côté web.
- **Entités imbriquées** : non gérées (JNLPBA est à plat).
- **Tokenisation** : à la prédiction, le tokenizer Penn Treebank peut découper différemment du corpus (parenthèses, slashes, tirets) ; à vérifier sur des textes réels.
- **Temps d'entraînement** du SVM à noyau sur le corpus complet.

Pistes : ajouter les états HMM et le word cache dans la partie Data ; seuil de fréquence sur les features (`DictVectorizer.restrict`) pour réduire le bruit ; comparaison avec un CRF (décodage global).

## 12. Dépannage

| Erreur | Cause | Solution |
|---|---|---|
| `RuntimeError: Dataset scripts are no longer supported` | `datasets >= 4` n'exécute plus les scripts de chargement | la partie Data charge la branche parquet ; voir sa documentation |
| `ValueError: Only sparse matrices with 32-bit integer indices` | indices `int64` renvoyés par `DictVectorizer` | corrigé par `as_int32` ; vérifier que `decoding.py` et `train.py` sont à jour |
| `FileNotFoundError ... svm_model.joblib` | modèle non entraîné | `python -m Model.train --kernel poly --final` |
| `LookupError` (NLTK) | ressources NLTK absentes | `python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng` |
| `ModuleNotFoundError: No module named 'Data'` | lancement hors de la racine du dépôt | se placer dans `NLP_classique/` et utiliser `python -m` |
| Port 8000 occupé | `mock_api.py` ou autre serveur actif | l'arrêter, ou lancer uvicorn avec `--port 8001` |
| Entraînement très long | noyau polynomial sur tout le corpus | réduire `--max-sentences`, essayer `--degree 1` |
| Activation du venv refusée (PowerShell) | politique d'exécution | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

## Référence

J. Kazama, T. Makino, Y. Ohta, J. Tsujii. *Tuning Support Vector Machines for Biomedical Named Entity Recognition*. Proceedings of the Workshop on Natural Language Processing in the Biomedical Domain, ACL, Philadelphia, 2002.

# **Partie web**

## Architecture et Déploiement : Implémentation du Pattern BFF (Backend For Frontend)

Pour garantir la sécurité, la robustesse et une séparation stricte des responsabilités (Separation of Concerns), ce projet adopte une architecture de type Backend For Frontend (BFF). Le serveur Spring Boot agit comme une passerelle d'API (API Gateway) et un bouclier exclusif entre le client (l'interface utilisateur web) et le service de traitement asynchrone (le modèle d'Intelligence Artificielle exposé via FastAPI en Python).

Cette approche empêche l'exposition directe du modèle d'apprentissage automatique sur le réseau public, protégeant ainsi les ressources matérielles de calcul et masquant la complexité de l'infrastructure sous-jacente.

### 1. Flux de Données et Contrats d'Interface (DTOs)

La communication inter-processus est strictement typée et standardisée via des objets de transfert de données (DTOs). L'utilisation des `records` (introduits dans les versions récentes de Java) garantit l'immuabilité des données et la sécurité des threads (thread-safety) lors du traitement simultané de plusieurs requêtes.

La bibliothèque Jackson, intégrée nativement dans Spring Boot, gère la sérialisation et la désérialisation du format JSON de manière transparente selon les contrats suivants :

* **NerRequest** : Encapsule la charge utile (payload). Il contient la chaîne de caractères soumise par l'utilisateur ainsi qu'un identifiant de session unique (`sessionId`) généré côté client pour le traçage et l'observabilité.
* **NerResponse** : Agit comme le conteneur de la réponse du modèle d'IA. Il attend une structure JSON précise contenant une liste d'entités.
* **EntityToken** : Représente l'unité d'analyse atomique. Il associe le texte brut extrait (`token`) à sa classification biomédicale issue du corpus GENIA (`tag` tel que PROTEIN, DNA, CELL_TYPE).

### 2. Logique Métier et Communication Inter-Services (Couche Service)

L'orchestration de la communication réseau est encapsulée dans la classe `NerService`. Cette couche a trois responsabilités critiques :

* **Validation et Sanitarisation** : Les données entrantes subissent une validation en amont (vérification de la nullité, limitation stricte de la longueur des chaînes). Cette étape prévient l'épuisement des ressources (attaques DoS ou dépassement de la mémoire allouée) avant même que le modèle de machine learning ne soit sollicité.
* **Client HTTP Moderne** : L'interface `RestClient` (l'API fluide et moderne de Spring remplaçant `RestTemplate`) est utilisée pour forger et exécuter les requêtes HTTP POST de manière synchrone vers l'instance FastAPI locale (port 8000).
* **Résilience et Tolérance aux Pannes** : La méthode d'appel intègre un mécanisme de capture d'exceptions. Si le microservice Python est hors ligne ou subit une surcharge temporelle, l'erreur réseau est interceptée. Le service renvoie alors une exception métier claire, évitant la propagation d'une erreur critique (Cascading Failure) qui ferait s'effondrer l'application principale.

### 3. Exposition de l'API (Couche Contrôleur)

La classe `NerController` expose l'endpoint RESTful principal sur la route `/api/ner/analyze`. Annotée de l'annotation `@RestController`, elle gère la traduction du protocole HTTP vers les objets Java. Elle réceptionne le corps de la requête (`@RequestBody`), délègue le traitement séquentiel au `NerService`, et encapsule le résultat final dans un objet `ResponseEntity`. Ce mécanisme assure le renvoi des codes de statut HTTP standardisés (ex: HTTP 200 OK en cas de succès).

---

## Interface Utilisateur et Ingénierie Front-end

L'interface graphique est conçue comme une Single Page Application (SPA) légère, s'appuyant sur des technologies web natives (Vanilla JavaScript, HTML5) et le framework utilitaire Tailwind CSS, sans nécessiter de bibliothèque lourde comme React ou Angular.

### Design System : L'approche Glassmorphism

L'interface rompt avec les paradigmes visuels habituels des outils d'intelligence artificielle. Elle adopte un style "Morpho Glaces" (Glassmorphism) qui confère un aspect académique, éditorial et minimaliste.

* **Technique CSS** : L'effet de verre dépoli est obtenu grâce aux propriétés CSS avancées `backdrop-filter: blur()` combinées à des arrière-plans semi-transparents (`rgba`).
* **DOM Dynamique** : L'affichage des résultats ne nécessite aucun rechargement de page. Le moteur JavaScript parse le tableau JSON reçu et manipule le Document Object Model (DOM) en créant dynamiquement des éléments `<span>`. Les balises générées intègrent des classes CSS spécifiques (ex: `tag-PROTEIN`), appliquant une typographie colorée stricte (Vert Olive, Noir, Blanc) basée sur la classification du modèle.
* **Responsive Design** : L'utilisation d'unités relatives et d'une grille CSS (CSS Grid) permet à l'interface de s'adapter dynamiquement aux contraintes d'affichage (smartphones, tablettes, moniteurs haute résolution).

### Gestion de l'État Client et Observabilité

L'application est optimisée pour des démonstrations publiques massives et simultanées.

* **Isolation des Sessions** : L'état de l'application (l'historique des requêtes) est géré de manière éphémère dans la mémoire vive (RAM) du navigateur de chaque utilisateur via des tableaux JavaScript. Cette architecture "Stateless" côté serveur garantit qu'aucune donnée utilisateur ne se croise.
* **Mise en Cache Visuelle** : La sélection d'un élément dans l'historique latéral recharge instantanément les données précédemment analysées dans la vue principale. Cette opération se fait intégralement côté client, évitant une nouvelle requête réseau redondante vers le serveur d'IA.
* **Traçage des Requêtes** : Dès l'initialisation de l'interface, une fonction mathématique génère un identifiant de session unique (UUID allégé). Ce jeton est injecté dans chaque payload HTTP, permettant à l'administrateur de surveiller l'activité et le multiplexage des utilisateurs en temps réel via les logs de la console Spring Boot.

---

## Stratégie de Déploiement et d'Intégration (Monorepo)

Le dépôt de code source adopte une structure Monorepo, hébergeant conjointement le code d'infrastructure (Java) et la logique scientifique (Python/SVM). Cette organisation garantit que les contrats d'interface (le format JSON attendu) évoluent de manière synchronisée.

### Transition Isomorphe (Du Mock à la Production)

Le découplage fort induit par le serveur Spring Boot permet une transition technique transparente lors de la finalisation du modèle d'apprentissage automatique.
L'adresse du service Python cible est externalisée dans le fichier de configuration `application.properties`. Ainsi, le remplacement de l'API de simulation (Mock) par le modèle SVM réel basé sur le corpus GENIA ne nécessite aucune recompilation du code Java ni aucune altération du code Front-end. L'intégrité du système repose exclusivement sur le respect du contrat de nommage des clés JSON.

### Déploiement Dynamique et Reverse Proxy (Ngrok)

Pour contourner les restrictions des réseaux locaux (NAT/Firewalls) lors de sessions de démonstration académique, le système s'appuie sur le logiciel de tunnelisation Ngrok.

* Le client Ngrok établit une connexion TCP sortante sécurisée depuis la machine hôte vers les serveurs Edge de Ngrok.
* Le trafic entrant sur un domaine statique public (ex: `https://projet-nlp.ngrok-free.app`) est instantanément redirigé vers le port local du serveur Spring Boot.
* Cette architecture permet de transformer l'application de développement local en un service SaaS (Software as a Service) temporaire, distribuable à l'audience via un simple code QR, avec des temps de latence minimisés.

