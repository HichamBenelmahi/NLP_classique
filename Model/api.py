from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .predict import predict_entities

app = FastAPI(title="Bio-NER SVM")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class TextRequest(BaseModel):
    text: str


@app.post("/api/predict")
def predict(req: TextRequest):
    return {"entities": predict_entities(req.text)}


@app.get("/health")
def health():
    return {"status": "ok"}