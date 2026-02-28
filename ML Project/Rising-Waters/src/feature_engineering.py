import logging
import re
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

MONTH_ALIASES = {
    "jan": "jan",
    "feb": "feb",
    "mar": "mar",
    "apr": "apr",
    "may": "may",
    "jun": "jun",
    "jul": "jul",
    "aug": "aug",
    "sep": "sep",
    "sept": "sep",
    "oct": "oct",
    "nov": "nov",
    "dec": "dec",
}


def _standardize_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    year_cols = [c for c in df.columns if c.lower() in {"year", "yr"}]
    if year_cols:
        df["year"] = df[year_cols[0]].astype(int)
    elif "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
    return df


def _impute_year_from_rainfall(flood_df: pd.DataFrame, rain_df: pd.DataFrame) -> pd.DataFrame:
    """
    If the flood dataset lacks a year column but its row count matches the number
    of unique rainfall years, assign the sorted rainfall years in order.
    This is a pragmatic fallback for yearly flood summaries without explicit year field.
    """
    if "year" in flood_df.columns:
        return flood_df
    rain_years = sorted(pd.unique(_standardize_year(rain_df)["year"].dropna()))
    if len(flood_df) == len(rain_years):
        flood_df = flood_df.copy()
        flood_df["year"] = rain_years
        logger.info("Imputed year column into flood data using rainfall years range (%s-%s)", rain_years[0], rain_years[-1])
    return flood_df


def _standardize_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    state_cols = [c for c in df.columns if c.lower() in {"state", "region", "state/ut"}]
    if state_cols:
        df["state"] = df[state_cols[0]].astype(str).str.strip()
    return df


def _standardize_district(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    district_cols = [c for c in df.columns if c.lower() in {"district", "dist"}]
    if district_cols:
        df["district"] = df[district_cols[0]].astype(str).str.strip()
    return df


def _find_month_columns(df: pd.DataFrame) -> List[str]:
    month_cols = []
    for col in df.columns:
        key = re.sub(r"[^a-z]", "", col.lower())
        if key in MONTH_ALIASES:
            month_cols.append(col)
    return month_cols


def _detect_target_column(df: pd.DataFrame) -> str:
    # prefer columns that are exactly named flood or end with '_flood'
    lower_map = {col.lower(): col for col in df.columns}
    if "flood" in lower_map:
        return lower_map["flood"]
    for col in df.columns:
        if col.lower().endswith("_flood"):
            return col
    for col in df.columns:
        if "flood" in col.lower():
            return col
    raise KeyError("No flood target column found. Please ensure a flood indicator column exists.")


def merge_datasets(flood_df: pd.DataFrame, rain_df: pd.DataFrame) -> pd.DataFrame:
    flood_df = _impute_year_from_rainfall(
        _standardize_year(_standardize_state(_standardize_district(flood_df))), rain_df
    )
    rain_df = _standardize_year(_standardize_state(_standardize_district(rain_df)))

    # Determine merge keys by availability priority
    keys = []
    if all(col in flood_df.columns for col in ["state", "district", "year"]) and all(
        col in rain_df.columns for col in ["state", "district", "year"]
    ):
        keys = ["state", "district", "year"]
    elif all(col in flood_df.columns for col in ["state", "year"]) and all(
        col in rain_df.columns for col in ["state", "year"]
    ):
        keys = ["state", "year"]
    elif "year" in flood_df.columns and "year" in rain_df.columns:
        keys = ["year"]
    else:
        raise KeyError("Cannot find common merge keys. Ensure year/state fields exist in both datasets.")

    merged = pd.merge(rain_df, flood_df, on=keys, how="inner", suffixes=("_rain", "_flood"))
    logger.info("Merged dataset shape: %s using keys %s", merged.shape, keys)
    return merged


def build_features(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()

    # Drop raw datetime columns if present
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz", "datetime64[ns]"]).columns.tolist()
    df = df.drop(columns=datetime_cols, errors="ignore")

    month_cols = _find_month_columns(df)
    if month_cols:
        df["annual_rainfall"] = df[month_cols].sum(axis=1)
        df["monthly_rainfall_avg"] = df[month_cols].mean(axis=1)
    else:
        # fallback: assume a single rainfall column exists
        rain_cols = [c for c in df.columns if "rain" in c.lower()]
        rain_col = rain_cols[0] if rain_cols else None
        if rain_col:
            df["annual_rainfall"] = df[rain_col]
            df["monthly_rainfall_avg"] = df[rain_col]
        else:
            raise KeyError("No rainfall columns found to compute features.")

    # deviation from long-term mean per state if state available
    if "state" in df.columns:
        df["rainfall_dev_from_mean"] = df["annual_rainfall"] - df.groupby("state")["annual_rainfall"].transform("mean")
    else:
        df["rainfall_dev_from_mean"] = df["annual_rainfall"] - df["annual_rainfall"].mean()

    # lag and rolling features by state if present, else overall
    group_keys = ["state"] if "state" in df.columns else []
    df = df.sort_values(group_keys + ["year"] if group_keys else ["year"])
    df["previous_year_rainfall"] = (
        df.groupby(group_keys)["annual_rainfall"].shift(1) if group_keys else df["annual_rainfall"].shift(1)
    )
    df["rolling_mean_rainfall_3y"] = (
        df.groupby(group_keys)["annual_rainfall"].rolling(window=3, min_periods=1).mean().reset_index(level=group_keys, drop=True)
        if group_keys
        else df["annual_rainfall"].rolling(window=3, min_periods=1).mean()
    )

    # target
    target_col = _detect_target_column(df)
    df["flood"] = df[target_col].apply(
        lambda x: 1
        if str(x).lower() in {"1", "yes", "true", "y"}
        else int(float(x)) if str(x).replace(".", "", 1).isdigit() else 0
    )

    # remove obvious irrelevant columns
    drop_cols = {"date_rain", "date_flood"}
    if target_col != "flood":
        drop_cols.add(target_col)
    drop_cols |= {c for c in df.columns if c.startswith("Unnamed")}
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df = df.dropna(subset=["flood"]).reset_index(drop=True)
    logger.info("Feature matrix shape: %s", df.shape)
    # Keep a compact, inference-friendly feature set
    keep_cols = [
        col
        for col in [
            "state",
            "district",
            "year",
            "annual_rainfall",
            "monthly_rainfall_avg",
            "rainfall_dev_from_mean",
            "previous_year_rainfall",
            "rolling_mean_rainfall_3y",
            # optional climate covariates if present
            "Temp",
            "Humidity",
            "Cloud Cover",
            "avgjune",
            "sub",
            "flood",
        ]
        if col in df.columns
    ]
    df = df[keep_cols]
    logger.info("Selected feature columns: %s", keep_cols)
    return df


def prepare_dataset(flood_df: pd.DataFrame, rain_df: pd.DataFrame) -> pd.DataFrame:
    merged = merge_datasets(flood_df, rain_df)
    return build_features(merged)
