"""
Technical Indicators
RSI, MACD, Bollinger Bands, Moving Averages
VIT Bhopal — Group 142
"""

import pandas as pd
import numpy as np


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a dataframe with OHLCV columns."""
    df = df.copy()
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_macd(df)
    df = add_volume_ratio(df)
    return df


def add_moving_averages(df, windows=(20, 50, 200)):
    for w in windows:
        df[f'MA{w}'] = df['Close'].rolling(w).mean()
    return df


def add_rsi(df, window=14):
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def add_bollinger_bands(df, window=20, num_std=2):
    df['BB_mid'] = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    df['BB_up'] = df['BB_mid'] + num_std * std
    df['BB_dn'] = df['BB_mid'] - num_std * std
    df['BB_width'] = df['BB_up'] - df['BB_dn']
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    df['MACD']        = df['Close'].ewm(span=fast).mean() - df['Close'].ewm(span=slow).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=signal).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']
    return df


def add_volume_ratio(df, window=20):
    df['Vol_MA']    = df['Volume'].rolling(window).mean()
    df['Vol_ratio'] = df['Volume'] / df['Vol_MA']
    return df


def generate_signal(rsi: float, macd: float, macd_signal: float, pred_change: float) -> dict:
    """
    Generate Buy/Sell/Hold signal from technical indicators.

    Scoring:
    +2: predicted change > 1% or RSI < 30 (oversold)
    +1: predicted change > 0 or RSI < 45
    -2: predicted change < -1% or RSI > 70 (overbought)
    -1: predicted change < 0 or RSI > 60
    ±1: MACD above/below signal line
    """
    score = 0
    if pred_change > 1:   score += 2
    elif pred_change > 0: score += 1
    elif pred_change < -1: score -= 2
    else: score -= 1

    if rsi < 30:   score += 2
    elif rsi < 45: score += 1
    elif rsi > 70: score -= 2
    elif rsi > 60: score -= 1

    if macd > macd_signal: score += 1
    else: score -= 1

    if score >= 3:    action = "BUY"
    elif score <= -2: action = "SELL"
    else:             action = "HOLD"

    confidence = min(100, abs(score) * 20 + 40)

    return {'action': action, 'score': score, 'confidence': confidence}
