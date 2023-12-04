from fastapi import FastAPI
from pydantic import BaseModel
from accid_predictor import predict_accid

app = FastAPI()

class DateTimeIn(BaseModel):
    year: str
    month: str

class PredictionOut(BaseModel):
    prediction: int

@app.get("/")
def index():
    return {"Heathcheck": "OK"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: DateTimeIn):
    predicted_accid = predict_accid(year=payload.year, month=payload.month)
    return {"prediction": predicted_accid}