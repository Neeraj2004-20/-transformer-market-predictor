import torch
import itertools
from train import train, create_sequences
from data_loader import download_data, preprocess_data
from model import TimeSeriesTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Hyperparameter grid
param_grid = {
    'MODEL_DIM': [32, 64],
    'NUM_HEADS': [2, 4],
    'NUM_LAYERS': [1, 2],
    'LR': [1e-3, 5e-4],
}

symbol = "AAPL"
start = "2015-01-01"
end = "2024-01-01"
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 5

def grid_search():
    df = download_data(symbol, start, end)
    df_norm = preprocess_data(df).values
    X, y = create_sequences(df_norm, SEQ_LEN)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    best_mse = float('inf')
    best_params = None
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), values))
        model = TimeSeriesTransformer(input_dim=5, model_dim=params['MODEL_DIM'], num_heads=params['NUM_HEADS'], num_layers=params['NUM_LAYERS'])
        optimizer = torch.optim.Adam(model.parameters(), lr=params['LR'])
        loss_fn = torch.nn.MSELoss()
        for epoch in range(EPOCHS):
            model.train()
            for i in range(0, len(X_train), BATCH_SIZE):
                xb = X_train[i:i+BATCH_SIZE]
                yb = y_train[i:i+BATCH_SIZE]
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_test)
            mse = mean_squared_error(y_test.numpy(), preds.numpy())
            print(f"Params: {params}, Test MSE: {mse:.4f}")
            if mse < best_mse:
                best_mse = mse
                best_params = params
    print(f"Best Params: {best_params}, Best Test MSE: {best_mse:.4f}")

if __name__ == "__main__":
    grid_search()
