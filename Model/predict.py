from functools import lru_cache
import joblib, nltk
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize

from Data.prepare_data import get_pos_tags
from .config import MODEL_PATH, VECTORIZER_PATH
from .decoding import decode, merge_o, repair_bio

_word_tok = TreebankWordTokenizer()          # tokenizer Penn Treebank, comme l'article


def _sentences(text):
    try:
        return sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True); nltk.download("punkt", quiet=True)
        return sent_tokenize(text)


@lru_cache(maxsize=1)
def _load():
    return joblib.load(VECTORIZER_PATH), joblib.load(MODEL_PATH)


def predict_entities(text: str) -> list[dict]:
    """texte brut -> [{"token": "...", "tag": "PROTEIN" | "DNA" | ... | "O"}, ...]"""
    sentences = [_word_tok.tokenize(s) for s in _sentences(text) if s.strip()]
    sentences = [s for s in sentences if s]
    if not sentences:
        return []
    vec, clf = _load()
    pos = [get_pos_tags(s) for s in sentences]
    raw = decode(sentences, pos, vec, clf)            # classes éclatées (O-NN...)
    out = []
    for toks, labels in zip(sentences, raw):
        tags = repair_bio([merge_o(c) for c in labels])
        for tok, tag in zip(toks, tags):
            out.append({"token": tok, "tag": "O" if tag == "O" else tag[2:].upper()})
    return out