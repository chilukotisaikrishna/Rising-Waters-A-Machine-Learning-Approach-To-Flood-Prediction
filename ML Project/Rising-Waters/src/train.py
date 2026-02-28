import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from .data_loader import load_datasets
from .feature_engineering import prepare_dataset
from .preprocessing import preprocess_data

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "flood_model.pkl"
RESULTS_JSON = MODELS_DIR / "training_results.json"


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def train_models():
    flood_df, rain_df = load_datasets()
    dataset = prepare_dataset(flood_df, rain_df)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(dataset)

    models = {
        "log_reg": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    results = []
    best_model_name = None
    best_f1 = -1

    for name, model in models.items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results.append({"model": name, **metrics})
        logger.info(
            "%s -> accuracy: %.3f, f1: %.3f, roc_auc: %.3f",
            name,
            metrics["accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = name

    # select and save best model
    best_model = models[best_model_name]
    joblib.dump(best_model, MODEL_PATH)
    logger.info("Saved best model '%s' to %s", best_model_name, MODEL_PATH)

    # persist metrics to JSON
    payload = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "best_model": best_model_name,
        "metrics": results,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    logger.info("Saved training metrics to %s", RESULTS_JSON)

    return results, best_model_name


if __name__ == "__main__":
    train_models()
