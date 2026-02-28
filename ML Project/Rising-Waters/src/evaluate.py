import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from .feature_engineering import prepare_dataset
from .data_loader import load_datasets
from .preprocessing import PREPROCESSOR_PATH, split_features_target

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "flood_model.pkl"


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}. Train the model first.")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def evaluate_model(test_size: float = 0.2, random_state: int = 42):
    model, preprocessor = load_artifacts()
    flood_df, rain_df = load_datasets()
    df = prepare_dataset(flood_df, rain_df)
    X, y = split_features_target(df, target_col="flood")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_test_transformed = preprocessor.transform(X_test)

    y_pred = model.predict(X_test_transformed)
    y_prob = model.predict_proba(X_test_transformed)[:, 1] if hasattr(model, "predict_proba") else None

    # reports
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))
    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
        logger.info("ROC AUC: %.3f", auc)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    cm_path = MODELS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    # roc curve
    if y_prob is not None:
        roc_disp = RocCurveDisplay.from_predictions(y_test, y_prob)
        roc_disp.figure_.savefig(MODELS_DIR / "roc_curve.png", bbox_inches="tight")
        plt.close(roc_disp.figure_)

    # feature importance
    feature_names = preprocessor.get_feature_names_out()
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        importance = np.abs(coef)

    if importance is not None:
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
            by="importance", ascending=False
        )
        top = fi_df.head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"], top["importance"])
        ax.invert_yaxis()
        ax.set_title("Top Feature Importances")
        fig.tight_layout()
        fi_path = MODELS_DIR / "feature_importance.png"
        fig.savefig(fi_path, bbox_inches="tight")
        plt.close(fig)
        top.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    logger.info("Saved evaluation artifacts to %s", MODELS_DIR)


if __name__ == "__main__":
    evaluate_model()
