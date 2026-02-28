import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

from .preprocessing import PREPROCESSOR_PATH

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "flood_model.pkl"


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Preprocessor file not found. Train the model first.")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def predict_flood(payload: Dict) -> Tuple[str, float]:
    model, preprocessor = load_artifacts()
    # ensure all expected columns exist (fit-time columns are in preprocessor.feature_names_in_)
    expected_cols = list(getattr(preprocessor, "feature_names_in_", []))
    input_df = pd.DataFrame([payload])
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_cols] if expected_cols else input_df
    processed = preprocessor.transform(input_df)
    prob = model.predict_proba(processed)[0][1]
    risk = "High" if prob >= 0.5 else "Low"
    return risk, prob


if __name__ == "__main__":
    sample = {
        "state": "Maharashtra",
        "year": 2015,
        "annual_rainfall": 1200,
        "monthly_rainfall_avg": 100,
        "previous_year_rainfall": 1150,
        "rolling_mean_rainfall_3y": 1100,
        "rainfall_dev_from_mean": 50,
    }
    risk, prob = predict_flood(sample)
    print(f"Flood Risk: {risk} ({prob:.2%})")
