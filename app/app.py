import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from accid_predictor import predict_accid

app = FastAPI()

class DateTimeIn(BaseModel):
    year: int
    month: int

class PredictionOut(BaseModel):
    prediction: int

@app.get("/")
def index():
    return {"Heathcheck": "OK"}

@app.post("/predict", response_model=PredictionOut)
async def predict_endpoint(payload: DateTimeIn):
    predicted_accid = predict_accid(year=payload.year, month=payload.month)
    return {"prediction": predicted_accid}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4343)
#uvicorn app:app --reload