"""
Lancer depuis le dossier NLP_classique (parent de Data/) :

    python -m Data.test_data            # test hors-ligne (exemple de l'article)
    python -m Data.test_data --online   # charge vraiment JNLPBA
"""
import sys

from Data import prepare_split, extract_word_features

REQUIRED_KEYS = {
    "bias", "w[0].word", "w[0].pref3", "w[0].suff3", "w[0].pos",
    "w[-1].word", "w[1].word", "prev_class",
}


def test_offline():
    split = [{
        "tokens": ["Number", "of", "glucocorticoid", "receptors", "in", "lymphocytes"],
        "ner_tags": ["O", "O", "B-protein", "I-protein", "O", "B-cell_type"],
    }]
    pos = ["NN", "IN", "NN", "NNS", "IN", "NNS"]

    X, y = prepare_split(split, pos_tagger=lambda t: pos)

    assert y == ["O-NN", "O-IN", "B-protein", "I-protein", "O-IN", "B-cell_type"], y
    assert REQUIRED_KEYS.issubset(X[2].keys())
    assert X[2]["w[0].word"] == "glucocorticoid"
    assert X[2]["w[0].pref3"] == "glu" and X[2]["w[0].suff3"] == "oid"
    assert X[2]["w[-1].word"] == "of" and X[2]["w[1].word"] == "receptors"
    assert X[2]["prev_class"] == "O-IN"
    assert X[3]["prev_class"] == "B-protein"
    assert X[0]["prev_class"] == "<START>"
    # même résultat à l'inférence, mot par mot
    assert extract_word_features(split[0]["tokens"], pos, 2, "O-IN") == X[2]
    print("OK - test hors-ligne")
    print(X[2])


def test_online():
    from Data import build_features_data
    (X_train_dict, y_train, X_validation_dict,
     y_validation, extract_fn) = build_features_data()

    assert len(X_train_dict) == len(y_train)
    assert len(X_validation_dict) == len(y_validation)
    assert all(isinstance(x, dict) for x in X_train_dict)
    assert all(isinstance(c, str) for c in y_train)
    assert REQUIRED_KEYS.issubset(X_train_dict[0].keys())
    assert REQUIRED_KEYS.issubset(X_validation_dict[0].keys())

    print("Train :", len(X_train_dict), len(y_train))
    print("Validation :", len(X_validation_dict), len(y_validation))
    print("Exemple de features :", X_train_dict[0])
    print("Exemple de classes :", y_train[:10])
    print("Fonction d'extraction :", extract_fn.__name__)
    print("Toutes les vérifications de base sont réussies.")


if __name__ == "__main__":
    test_offline()
    if "--online" in sys.argv:
        test_online()