from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS_DIR / "svm_model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"
DATA_CACHE = ARTIFACTS_DIR / "data_cache.joblib"   # évite de refaire le POS tagging