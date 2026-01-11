# Transformer-Based Market Movement Prediction (AIML)


This project uses transformer-based deep learning models to predict market movements (e.g., stock prices) using historical data. It includes data download, preprocessing, model training, evaluation, visualization, and advanced features for research and experimentation.


## Features
- Download historical stock data (Yahoo Finance)
- Data preprocessing and visualization
- Transformer-based time series forecasting (PyTorch)
- Model evaluation and prediction visualization
- Hyperparameter search (see hyperparameter_search.py)
- Model saving/loading for reuse
- Support for multiple assets (edit config.json)
- Advanced evaluation metrics (MSE, MAE, R²)
- Configurable via config.json
- Experiment logging (experiment_log.json)
- Easy to extend for other markets (crypto, forex, etc.)

## Requirements
- Python 3.8+
- See requirements.txt for dependencies


## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Edit `config.json` to set symbol, dates, and model parameters
3. Run the main script: `python main.py` or `python train.py`
4. For hyperparameter search: `python hyperparameter_search.py`
5. Prediction plots are saved as `prediction_vs_actual.png`
6. Model weights are saved as `model_<symbol>.pt`
7. Experiment results are logged in `experiment_log.json`


## Project Structure
- main.py — Entry point, runs the pipeline
- data_loader.py — Download and preprocess data
- model.py — Transformer model definition
- train.py — Training and evaluation logic
- hyperparameter_search.py — Grid search for best model parameters
- utils.py — Helper functions (plotting, etc.)
- requirements.txt — Python dependencies
- config.json — Configuration for model/data
- experiment_log.json — Experiment results log
- README.md — Project documentation


## Notes
- Default: Predicts next-day closing price for AAPL (Apple Inc.)
- To use other assets, edit `config.json` and re-run
- All results and plots are saved in the project directory
