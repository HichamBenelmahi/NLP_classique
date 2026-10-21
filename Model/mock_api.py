from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def mock_ner(request: TextRequest):
    print(f"Requête reçue depuis Spring Boot : {request.text}")

    # On renvoie le contrat exact que tu as négocié
    return {
        "entities": [
            {"token": "Number", "tag": "O"},
            {"token": "of", "tag": "O"},
            {"token": "glucocorticoid", "tag": "PROTEIN"},
            {"token": "receptors", "tag": "PROTEIN"},
            {"token": "in", "tag": "O"},
            {"token": "lymphocytes", "tag": "CELL_TYPE"}
        ]
    }