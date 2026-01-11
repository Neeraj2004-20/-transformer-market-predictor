from fastapi import FastAPI, Query
from pydantic import BaseModel
import torch
import numpy as np
from model import TimeSeriesTransformer
from data_loader import preprocess_data
import json

app = FastAPI()

class PredictRequest(BaseModel):
    data: list
    config: dict = None

@app.on_event("startup")
def load_model():
    global model, config
    with open("config.json") as f:
        config = json.load(f)
    model = TimeSeriesTransformer(input_dim=5, model_dim=config["MODEL_DIM"], num_heads=config["NUM_HEADS"], num_layers=config["NUM_LAYERS"])
    model.load_state_dict(torch.load(f"model_{config['symbol']}.pt", map_location="cpu"))
    model.eval()

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.data, dtype=np.float32)
    X = torch.tensor(X).unsqueeze(0)  # (1, seq_len, input_dim)
    with torch.no_grad():
        pred = model(X).item()
    return {"prediction": pred}

@app.get("/")
def root():
    return {"message": "Transformer Market Movement API"}
