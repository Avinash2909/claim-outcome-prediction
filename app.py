import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Insurance Claim Outcome Prediction API",
    description="Predicts insurance claim outcomes using Machine Learning",
    version="1.0"
)

pipeline = joblib.load("models/claim_model.pkl")
metadata = joblib.load("models/feature_metadata.pkl")

payer_denial_rate = metadata["payer_denial_rate"]
provider_avg_amount = metadata["provider_avg_amount"]
default_provider_avg = metadata["default_provider_avg"]


class ClaimInput(BaseModel):
    payer: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    cpt_code: str = Field(min_length=1)
    diagnosis_code: str = Field(min_length=1)
    billed_amount: float = Field(gt=0)
    patient_age: int = Field(ge=1, le=120)


@app.get("/")
def home():
    return {
        "message": "Insurance Claim Prediction API is running"
    }


@app.post("/predict")
def predict_claim(claim: ClaimInput):
    input_data = pd.DataFrame([claim.model_dump()])

    input_data["payer_denial_rate"] = (
        input_data["payer"]
        .map(payer_denial_rate)
        .fillna(0)
    )

    input_data["provider_avg_billed_amount"] = (
        input_data["provider"]
        .map(provider_avg_amount)
        .fillna(default_provider_avg)
    )

    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]

    probability_dict = {
        class_name: round(float(probability), 4)
        for class_name, probability
        in zip(pipeline.classes_, probabilities)
    }

    return {
        "predicted_status": prediction,
        "probabilities": probability_dict
    }