"""
Data Preprocessing Pipeline
VIT Bhopal — Group 142
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def create_sequences(data: np.ndarray, look_back: int = 60):
    """
    Create sliding window sequences for time-series prediction.

    Args:
        data: Normalized 1D array of prices
        look_back: Number of past days to use as input features

    Returns:
        X: (samples, look_back) input sequences
        y: (samples,) target values
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def preprocess(df: pd.DataFrame, look_back: int = 60, train_ratio: float = 0.8):
    """
    Full preprocessing pipeline:
    1. Extract Close price
    2. MinMax normalize to [0, 1]
    3. Create train/test sequences

    Returns dict with X_train, y_train, X_test, y_test, scaler
    """
    close = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)

    train_size = int(len(scaled) * train_ratio)
    train_data = scaled[:train_size]
    test_data  = scaled[train_size:]

    X_train, y_train = create_sequences(train_data, look_back)
    X_test, y_test   = create_sequences(
        np.concatenate([train_data[-look_back:], test_data]), look_back
    )

    # Reshape for RNN: (samples, timesteps, features)
    X_train = X_train.reshape(-1, look_back, 1)
    X_test  = X_test.reshape(-1, look_back, 1)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test':  X_test,  'y_test':  y_test,
        'scaler':  scaler,
    }
