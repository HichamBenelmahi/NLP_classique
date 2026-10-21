import numpy as np
from Data.features import START_CLASS, extract_word_features


def as_int32(X):
    """Les SVM de scikit-learn n'acceptent que des indices int32 ;
    DictVectorizer peut renvoyer de l'int64 avec les versions récentes."""
    X = X.tocsr()
    if X.indices.dtype != np.int32:
        X = X.copy()
        X.indices = X.indices.astype(np.int32)
        X.indptr = X.indptr.astype(np.int32)
    return X


def merge_o(label):
    """O-NN, O-IN... -> O"""
    return "O" if label.startswith("O-") else label


def repair_bio(tags):
    """une entité B-C ne se termine qu'à un autre B- ou à un O."""
    out, cur = [], None
    for t in tags:
        if t == "O":
            out.append("O"); cur = None
        elif t.startswith("B-"):
            cur = t[2:]; out.append(t)
        else:                                  # I-xxx
            if cur is None:
                cur = t[2:]; out.append("B-" + cur)
            else:
                out.append("I-" + cur)
    return out


def rebuild_sentences(X_dicts, y):
    """Données aplaties -> liste de (tokens, pos, classes) par phrase."""
    sents, toks, pos, cls = [], [], [], []
    for x, c in zip(X_dicts, y):
        if x["prev_class"] == START_CLASS and toks:
            sents.append((toks, pos, cls)); toks, pos, cls = [], [], []
        toks.append(x["w[0].word"]); pos.append(x["w[0].pos"]); cls.append(c)
    if toks:
        sents.append((toks, pos, cls))
    return sents


def decode(sent_tokens, sent_pos, vec, clf):
    """Décodage gauche→droite par lots : au pas i, on prédit le i-ème mot de
    toutes les phrases d'un coup, avec leur classe précédente prédite."""
    preds = [[] for _ in sent_tokens]
    for i in range(max(len(t) for t in sent_tokens)):
        active = [k for k, t in enumerate(sent_tokens) if len(t) > i]
        feats = [extract_word_features(sent_tokens[k], sent_pos[k], i,
                                       preds[k][-1] if preds[k] else START_CLASS)
                 for k in active]
        for k, c in zip(active, clf.predict(as_int32(vec.transform(feats)))):
            preds[k].append(c)
    return preds