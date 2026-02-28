import logging
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
SCALER_PATH = MODELS_DIR / "scaler.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"


def split_features_target(df: pd.DataFrame, target_col: str = "flood") -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def save_preprocessor(preprocessor: ColumnTransformer) -> None:
    """Persist the fitted transformers for inference."""
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    # also persist the numeric scaler component for direct access if needed
    if hasattr(preprocessor, "named_transformers_") and "num" in preprocessor.named_transformers_:
        joblib.dump(preprocessor.named_transformers_["num"], SCALER_PATH)
    logger.info("Saved preprocessor to %s and scaler to %s", PREPROCESSOR_PATH, SCALER_PATH)


def preprocess_data(
    df: pd.DataFrame, target_col: str = "flood", test_size: float = 0.2, random_state: int = 42
):
    X, y = split_features_target(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    save_preprocessor(preprocessor)
    logger.info("Train shape after preprocessing: %s", X_train_processed.shape)
    logger.info("Test shape after preprocessing: %s", X_test_processed.shape)
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
