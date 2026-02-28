import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"
FLOOD_FILE = DATA_DIR / "flood dataset.xlsx"
RAIN_FILE = DATA_DIR / "rainfall in india 1901-2015.xlsx"


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected data file missing: {path}")


def _parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine year/month columns into datetime if present."""
    year_cols = [c for c in df.columns if c.lower() in {"year", "yr"}]
    month_cols = [c for c in df.columns if c.lower() in {"month", "mn"}]
    if year_cols and month_cols:
        year_col = year_cols[0]
        month_col = month_cols[0]
        df["date"] = pd.to_datetime(
            dict(year=df[year_col], month=df[month_col], day=1), errors="coerce"
        )
    elif year_cols:
        year_col = year_cols[0]
        df["date"] = pd.to_datetime(df[year_col].astype(int), format="%Y", errors="coerce")
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(axis=0, how="all", inplace=True)
    df = _parse_datetime(df)
    # forward/backward fill for small gaps
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def load_flood_data(path: Path = FLOOD_FILE) -> pd.DataFrame:
    _ensure_exists(path)
    df = pd.read_excel(path)
    df = _clean(df)
    logger.info("Flood dataset loaded: %s", path)
    logger.info("Flood shape: %s", df.shape)
    logger.info("Flood columns: %s", list(df.columns))
    return df


def load_rainfall_data(path: Path = RAIN_FILE) -> pd.DataFrame:
    _ensure_exists(path)
    df = pd.read_excel(path)
    df = _clean(df)
    logger.info("Rainfall dataset loaded: %s", path)
    logger.info("Rainfall shape: %s", df.shape)
    logger.info("Rainfall columns: %s", list(df.columns))
    return df


def load_datasets(
    flood_path: Path = FLOOD_FILE, rainfall_path: Path = RAIN_FILE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    flood_df = load_flood_data(flood_path)
    rain_df = load_rainfall_data(rainfall_path)
    logger.info("Flood summary:\n%s", flood_df.describe(include="all"))
    logger.info("Rainfall summary:\n%s", rain_df.describe(include="all"))
    return flood_df, rain_df


if __name__ == "__main__":
    load_datasets()
