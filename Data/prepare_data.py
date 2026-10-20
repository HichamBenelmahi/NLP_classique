from .features import START_CLASS, extract_fn


def get_pos_tags(tokens):
    """Attribue un POS (Penn Treebank) à chaque token."""
    import nltk
    try:
        return [tag for _, tag in nltk.pos_tag(tokens)]
    except LookupError:
        for resource in ("averaged_perceptron_tagger_eng",
                         "averaged_perceptron_tagger"):
            nltk.download(resource, quiet=True)
        return [tag for _, tag in nltk.pos_tag(tokens)]


def get_label_names(split):
    """Récupère les noms des classes du corpus (None si absents)."""
    features = getattr(split, "features", None)
    if not features or "ner_tags" not in features:
        return None
    feature = getattr(features["ner_tags"], "feature", None)
    names = getattr(feature, "names", None)
    return names or None


def convert_tags(raw_tags, label_names):
    """Convertit les IDs en classes lisibles si nécessaire."""
    tags = []
    for tag in raw_tags:
        if isinstance(tag, int):
            if label_names is None:
                raise ValueError("Les noms des classes NER sont introuvables.")
            tag = label_names[tag]
        tags.append(str(tag))
    return tags


def split_class(tag, pos):
    """O devient O-POS ; les classes d'entités restent inchangées."""
    if tag == "O":
        return f"O-{pos}"
    return tag


def prepare_split(split, pos_tagger=get_pos_tags):
    X_dict = []
    y = []

    label_names = get_label_names(split)

    for example in split:
        tokens = example["tokens"]
        tags = convert_tags(example["ner_tags"], label_names)

        if len(tokens) != len(tags):
            raise ValueError("Tokens et étiquettes non alignés.")

        pos_tags = pos_tagger(tokens)

        if len(tokens) != len(pos_tags):
            raise ValueError("Tokens et POS non alignés.")

        # Classes éclatées de la phrase (O -> O-POS)
        classes = [split_class(t, p) for t, p in zip(tags, pos_tags)]

        # Classe du mot précédent (teacher forcing : classe annotée)
        previous_classes = [START_CLASS] + classes[:-1]

        X_dict.extend(extract_fn(tokens, pos_tags, previous_classes))
        y.extend(classes)

    return X_dict, y


def build_features_data():
    from datasets import load_dataset

    base_url = (
        "hf://datasets/bigbio/jnlpba@refs%2Fconvert%2Fparquet/"
        "jnlpba_source/"
    )
    try:
        dataset = load_dataset(
            "parquet",
            data_files={
                "train": base_url + "train/*.parquet",
                "validation": base_url + "validation/*.parquet",
            },
        )
    except Exception:
        # Repli : chargement classique du dépôt
        dataset = load_dataset("bigbio/jnlpba", name="jnlpba_source")

    X_train_dict, y_train = prepare_split(dataset["train"])
    X_validation_dict, y_validation = prepare_split(dataset["validation"])

    return (
        X_train_dict,
        y_train,
        X_validation_dict,
        y_validation,
        extract_fn,
    )