"""
Model Architectures
BiLSTM, LSTM, GRU, BiGRU for stock price prediction
VIT Bhopal — Group 142
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm(input_shape, units=(128, 64), dropout=(0.3, 0.2), lr=0.0001):
    model = Sequential([
        LSTM(units[0], return_sequences=True, input_shape=input_shape),
        Dropout(dropout[0]),
        LSTM(units[1]),
        Dropout(dropout[1]),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr), loss='mse', metrics=['mae'])
    return model


def build_gru(input_shape, units=(128, 64), dropout=(0.3, 0.2), lr=0.0001):
    model = Sequential([
        GRU(units[0], return_sequences=True, input_shape=input_shape),
        Dropout(dropout[0]),
        GRU(units[1]),
        Dropout(dropout[1]),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr), loss='mse', metrics=['mae'])
    return model


def build_bilstm(input_shape, units=(128, 64), dropout=(0.3, 0.2), lr=0.0001):
    """
    Bidirectional LSTM — best performing model.
    Processes sequences in both forward and backward directions,
    capturing richer temporal dependencies.
    """
    model = Sequential([
        Bidirectional(LSTM(units[0], return_sequences=True), input_shape=input_shape),
        Dropout(dropout[0]),
        Bidirectional(LSTM(units[1])),
        Dropout(dropout[1]),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr), loss='mse', metrics=['mae'])
    return model


def build_bigru(input_shape, units=(128, 64), dropout=(0.3, 0.2), lr=0.0001):
    model = Sequential([
        Bidirectional(GRU(units[0], return_sequences=True), input_shape=input_shape),
        Dropout(dropout[0]),
        Bidirectional(GRU(units[1])),
        Dropout(dropout[1]),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr), loss='mse', metrics=['mae'])
    return model


MODEL_REGISTRY = {
    'lstm':   build_lstm,
    'gru':    build_gru,
    'bilstm': build_bilstm,
    'bigru':  build_bigru,
}


def get_model(name: str, input_shape: tuple, **kwargs):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](input_shape, **kwargs)
