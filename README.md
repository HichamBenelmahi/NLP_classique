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

# **Partie model**

# **Partie web**
