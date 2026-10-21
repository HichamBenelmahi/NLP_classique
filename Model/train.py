
import argparse, time
import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC, SVC

from Data import build_features_data
from .config import DATA_CACHE, MODEL_PATH, VECTORIZER_PATH
from .decoding import as_int32, rebuild_sentences
from .evaluate import evaluate


def load_data():
    if DATA_CACHE.exists():
        return joblib.load(DATA_CACHE)
    data = build_features_data()[:4]
    joblib.dump(data, DATA_CACHE)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=["linear", "poly"], default="linear")
    ap.add_argument("--max-sentences", type=int, default=None)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.01)    # « s » de l'article
    ap.add_argument("--coef0", type=float, default=1.0)     # « r »
    ap.add_argument("--degree", type=int, default=2)        # « d »
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()

    X_dict, y, X_val_dict, y_val = load_data()
    train_s = rebuild_sentences(X_dict, y)
    print("phrases train :", len(train_s))

    if a.final:
        X_tr, y_tr = X_dict, y
        eval_s = rebuild_sentences(X_val_dict, y_val)
        print("phrases validation :", len(eval_s))
    else:                                    # dev = derniers 10 % des phrases du train
        n_dev = len(train_s) // 10
        cut = sum(len(s[0]) for s in train_s[:-n_dev])
        X_tr, y_tr, eval_s = X_dict[:cut], y[:cut], train_s[-n_dev:]

    if a.max_sentences:
        cut = sum(len(s[0]) for s in train_s[:a.max_sentences])
        X_tr, y_tr = X_tr[:cut], y_tr[:cut]

    vec = DictVectorizer()
    X = as_int32(vec.fit_transform(X_tr))
    print("échantillons :", X.shape[0], "| features :", X.shape[1], "| classes :", len(set(y_tr)))

    if a.kernel == "linear":
        clf = LinearSVC(C=a.C)
    else:                                    # pairwise (libsvm), noyau (s·x·x' + r)^d
        clf = SVC(kernel="poly", degree=a.degree, gamma=a.gamma, coef0=a.coef0,
                  C=a.C, cache_size=2000)

    t = time.time()
    clf.fit(X, y_tr)
    print(f"entraînement : {time.time() - t:.0f} s")
    evaluate(eval_s, vec, clf)

    if a.final:
        joblib.dump(vec, VECTORIZER_PATH)
        joblib.dump(clf, MODEL_PATH)
        print("artefacts sauvegardés")


if __name__ == "__main__":
    main()