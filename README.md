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

