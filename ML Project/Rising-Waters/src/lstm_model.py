import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from .data_loader import load_datasets
from .feature_engineering import _find_month_columns, _standardize_year

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
LSTM_PATH = MODELS_DIR / "flood_lstm.h5"


def build_sequences(rain_df: pd.DataFrame, flood_df: pd.DataFrame, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    rain_df = _standardize_year(rain_df)
    flood_df = _standardize_year(flood_df)
    month_cols = _find_month_columns(rain_df)
    rain_df["annual_rainfall"] = rain_df[month_cols].sum(axis=1) if month_cols else rain_df.select_dtypes(include=[np.number]).iloc[:, 0]

    # aggregate rainfall by year (and state if available)
    group_keys = ["state"] if "state" in rain_df.columns else []
    rainfall_series = rain_df.groupby(group_keys + ["year"]) ["annual_rainfall"].mean().reset_index()
    flood_series = flood_df.groupby(group_keys + ["year"]).sum().reset_index()

    merged = pd.merge(rainfall_series, flood_series, on=group_keys + ["year"], how="inner")
    target_col = [c for c in merged.columns if "flood" in c.lower()][0]
    merged = merged.sort_values(group_keys + ["year"]) if group_keys else merged.sort_values("year")

    sequences = []
    labels = []
    for _, group in merged.groupby(group_keys) if group_keys else [(None, merged)]:
        arr = group["annual_rainfall"].values
        target = group[target_col].values
        for i in range(len(arr) - window):
            sequences.append(arr[i : i + window])
            labels.append(1 if target[i + window] > 0 else 0)
    return np.array(sequences)[..., np.newaxis], np.array(labels)


def build_lstm(input_shape):
    model = Sequential(
        [
            LSTM(64, return_sequences=False, input_shape=input_shape),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm(window: int = 5, epochs: int = 20, batch_size: int = 32):
    flood_df, rain_df = load_datasets()
    X, y = build_sequences(rain_df, flood_df, window=window)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_lstm((window, 1))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(LSTM_PATH)
    logger.info("Saved LSTM model to %s", LSTM_PATH)
    return history


if __name__ == "__main__":
    train_lstm()
