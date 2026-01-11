import yfinance as yf
import pandas as pd

def download_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.dropna()
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize features
    return (df - df.mean()) / df.std()

if __name__ == "__main__":
    symbol = "AAPL"
    start = "2015-01-01"
    end = "2024-01-01"
    df = download_data(symbol, start, end)
    df_norm = preprocess_data(df)
    print(df_norm.head())
