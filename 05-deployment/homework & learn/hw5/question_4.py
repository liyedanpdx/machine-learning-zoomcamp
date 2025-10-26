import pickle
from fastapi import FastAPI
from pydantic import BaseModel

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict")
def predict(lead: Lead):
    X = lead.dict()
    prob = model.predict_proba([X])[0, 1]
    return {"conversion_probability": prob}
