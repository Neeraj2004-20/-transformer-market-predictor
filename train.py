import torch
import torch.nn as nn
import numpy as np
from model import TimeSeriesTransformer
from data_loader import download_data, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

import json
import os

from utils import plot_predictions

symbol = "AAPL"

# Configurable parameters
CONFIG_PATH = "config.json"

DEFAULT_CONFIG = {
    "symbol": "AAPL",
    "start": "2015-01-01",
    "end": "2024-01-01",
    "SEQ_LEN": 30,
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "MODEL_DIM": 64,
    "NUM_HEADS": 4,
    "NUM_LAYERS": 2,
    "LR": 1e-3
}
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    else:
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len][3])  # Close price
    return np.array(xs), np.array(ys)

def train(config=None):
    if config is None:
        config = load_config()
    df = download_data(config["symbol"], config["start"], config["end"])
    df_norm = preprocess_data(df).values
    X, y = create_sequences(df_norm, config["SEQ_LEN"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = TimeSeriesTransformer(input_dim=5, model_dim=config["MODEL_DIM"], num_heads=config["NUM_HEADS"], num_layers=config["NUM_LAYERS"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])
    loss_fn = nn.MSELoss()

    for epoch in range(config["EPOCHS"]):
        model.train()
        losses = []
        for i in tqdm(range(0, len(X_train), config["BATCH_SIZE"])):
            xb = X_train[i:i+config["BATCH_SIZE"]]
            yb = y_train[i:i+config["BATCH_SIZE"]]
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {np.mean(losses):.4f}")

    # Save model
    torch.save(model.state_dict(), f"model_{config['symbol']}.pt")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        preds_np = preds.numpy()
        y_test_np = y_test.numpy()
        mse = mean_squared_error(y_test_np, preds_np)
        mae = mean_absolute_error(y_test_np, preds_np)
        r2 = r2_score(y_test_np, preds_np)
        print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Visualization
        plot_predictions(y_test_np, preds_np, title=f"{config['symbol']} Test Set: Predictions vs Actual")

        # Log experiment
        log = {
            "symbol": config["symbol"],
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "params": config
        }
        with open("experiment_log.json", "a") as f:
            f.write(json.dumps(log) + "\n")

    if __name__ == "__main__":
        config = load_config()
        train(config)
