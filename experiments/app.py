from fastapi import FastAPI
from pydantic import BaseModel

from predict_utils import predict_sms

app = FastAPI(
    title="SMS Spam Classifier",
    description="RNN-based SMS spam detection API",
    version="1.0",
)


class SMSRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    spam_probability: float


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: SMSRequest):
    return predict_sms(request.text)
