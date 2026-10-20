START_CLASS = "<START>"
BOS = "<BOS>"
EOS = "<EOS>"

# Fenêtre de contexte (article : positions -2..2)
CONTEXT_OFFSETS = (-2, -1, 1, 2)


def _word_at(tokens, j):
    if j < 0:
        return BOS
    if j >= len(tokens):
        return EOS
    return tokens[j]


def _pos_at(pos_tags, j):
    if j < 0:
        return BOS
    if j >= len(pos_tags):
        return EOS
    return pos_tags[j]


def extract_word_features(tokens, pos_tags, i, prev_class):
    """
    tokens : liste des mots ['Number', 'of', 'IL-2', ...]
    pos_tags : liste des POS ['NN', 'IN', 'NNP', ...]
    i : position du mot actuel
    prev_class : classe assignée au mot précédent (ex: 'O-IN', 'B-protein')
    """
    word = tokens[i]

    features = {
        "bias": 1.0,
        "w[0].word": word,
        "w[0].pref3": word[:3],
        "w[0].suff3": word[-3:],
        "w[0].pos": pos_tags[i],
    }

    for k in CONTEXT_OFFSETS:
        features[f"w[{k}].word"] = _word_at(tokens, i + k)
        features[f"w[{k}].pos"] = _pos_at(pos_tags, i + k)

    features["prev_class"] = prev_class
    return features


def extract_fn(tokens, pos_tags=None, previous_classes=None):
    """
    Extrait les caractéristiques de chaque token d'une phrase.

    tokens : liste de mots
    pos_tags : liste des étiquettes POS associées aux mots
    previous_classes : previous_classes[i] = classe du mot PRÉCÉDENT le mot i
                       (donc previous_classes[0] = "<START>").
                       Utile en entraînement / validation. En prédiction,
                       la classe précédente dépend de la prédiction : on
                       appelle alors extract_word_features mot par mot.
    """
    if pos_tags is None:
        pos_tags = ["UNK"] * len(tokens)

    if previous_classes is None:
        previous_classes = [START_CLASS] * len(tokens)

    if len(tokens) != len(pos_tags):
        raise ValueError("tokens et pos_tags doivent avoir la même longueur.")

    if len(tokens) != len(previous_classes):
        raise ValueError(
            "tokens et previous_classes doivent avoir la même longueur."
        )

    return [
        extract_word_features(
            tokens=tokens,
            pos_tags=pos_tags,
            i=i,
            prev_class=previous_classes[i],   # <- correction : 1 classe, pas la liste
        )
        for i in range(len(tokens))
    ]