import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(
    title="Production Model Serving & Drift Monitoring Service",
    description="Real-time inference API equipped with automated model version routing",
    version="1.0.0"
)

# Active Model Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
active_model_path = os.path.join(project_root, "models", "active_model.joblib")

class PredictionPayload(BaseModel):
    credit_score_scaled: float
    debt_to_income_ratio: float
    monthly_income_log: float
    revolving_utilization: float
    num_credit_inquiries_6m: float
    delinquency_history_count: float
    loan_amount_requested: float
    employment_duration_years: float
    savings_balance_scaled: float
    transaction_velocity_score: float

class BatchPayload(BaseModel):
    records: List[PredictionPayload]

@app.get("/health")
def health_check():
    is_ready = os.path.exists(active_model_path)
    return {
        "status": "healthy" if is_ready else "degraded",
        "active_model_ready": is_ready,
        "model_file": active_model_path
    }

@app.post("/predict")
def predict(payload: PredictionPayload):
    if not os.path.exists(active_model_path):
        raise HTTPException(status_code=503, detail="Active model is not trained yet.")
        
    model = joblib.load(active_model_path)
    data = pd.DataFrame([payload.model_dump()])
    
    pred = int(model.predict(data)[0])
    prob = float(model.predict_proba(data)[0, 1])
    
    return {
        "prediction": pred,
        "default_probability": round(prob, 4),
        "decision": "REJECT" if pred == 1 else "APPROVE"
    }

@app.post("/predict_batch")
def predict_batch(payload: BatchPayload):
    if not os.path.exists(active_model_path):
        raise HTTPException(status_code=503, detail="Active model is not trained yet.")
        
    model = joblib.load(active_model_path)
    records = [r.model_dump() for r in payload.records]
    data = pd.DataFrame(records)
    
    preds = model.predict(data).tolist()
    probs = model.predict_proba(data)[:, 1].tolist()
    
    return {
        "batch_size": len(preds),
        "predictions": preds,
        "default_probabilities": [round(p, 4) for p in probs]
    }
