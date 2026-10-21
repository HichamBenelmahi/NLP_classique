def predict_entities(text: str):
    """Point d'entrée unique : texte brut -> [{"token": ..., "tag": ...}, ...]."""
    from .predict import predict_entities as _predict
    return _predict(text)
