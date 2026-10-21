from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from .decoding import decode, merge_o, repair_bio


def evaluate(sents, vec, clf, verbose=True):
    toks = [s[0] for s in sents]
    pos = [s[1] for s in sents]
    gold = [[merge_o(c) for c in s[2]] for s in sents]
    pred = [repair_bio([merge_o(c) for c in p]) for p in decode(toks, pos, vec, clf)]
    ident = lambda seqs: [[t if t == "O" else t[:2] + "X" for t in s] for s in seqs]
    res = {"precision": float(precision_score(gold, pred)), "recall": float(recall_score(gold, pred)),
           "f1": float(f1_score(gold, pred)),
           "f1_identification": float(f1_score(ident(gold), ident(pred)))}
    if verbose:
        print(classification_report(gold, pred, digits=3))
        print(res)
    return res