from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from ml_signal.core_compat import calculate_features, load_data
from ml_signal.features.candidate import add_candidate_features
from ml_signal.features.registry import get_feature_sets as registry_get_feature_sets
from ml_signal.features.valuation import (
    get_extended_valuation_columns,
    get_valuation_feature_columns,
)
from ml_signal.labels.tp_sl import calculate_targets


logger = logging.getLogger(__name__)

PROJECT_ROOT = getattr(cfg, "PROJECT_ROOT", Path(__file__).resolve().parents[3])
PROCESSED_DATA_DIR = getattr(
    cfg,
    "PROCESSED_DATA_DIR",
    PROJECT_ROOT / "data" / "processed_data",
)
VNINDEX_FILE = getattr(
    cfg,
    "VNINDEX_FILE",
    PROCESSED_DATA_DIR / "VNINDEX_standardized.csv",
)
STOCK_FILES = getattr(cfg, "STOCK_FILES", [])
VALUATION_DATA_DIR = PROCESSED_DATA_DIR / "valuation"


def get_stock_file_map() -> dict[str, Path]:
    file_map: dict[str, Path] = {}

    for file_path in STOCK_FILES:
        path = Path(file_path)
        ticker = path.stem.replace("_standardized", "").upper()
        file_map[ticker] = path

    return file_map


def resolve_stock_file(ticker: str) -> Path:
    ticker = ticker.upper()
    file_map = get_stock_file_map()

    if ticker in file_map:
        return file_map[ticker]

    fallback_path = PROCESSED_DATA_DIR / f"{ticker}_standardized.csv"

    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        f"Cannot find standardized data file for ticker: {ticker}"
    )


def load_vnindex_data() -> pd.DataFrame:
    df_vnindex = load_data(VNINDEX_FILE, is_vnindex=True)

    if df_vnindex is None:
        raise ValueError(f"VNINDEX data could not be loaded: {VNINDEX_FILE}")

    return df_vnindex


def load_valuation_features(ticker: str) -> pd.DataFrame | None:
    ticker = ticker.upper()
    valuation_path = VALUATION_DATA_DIR / f"{ticker}_valuation_standardized.csv"

    if not valuation_path.exists():
        logger.warning(
            "Valuation file not found | ticker=%s | path=%s",
            ticker,
            valuation_path,
        )
        return None

    df_valuation = pd.read_csv(valuation_path, encoding="utf-8-sig")
    df_valuation.columns = df_valuation.columns.astype(str).str.strip()

    if "Date" not in df_valuation.columns:
        raise ValueError(f"Valuation file must contain Date column: {valuation_path}")

    df_valuation["Date"] = pd.to_datetime(df_valuation["Date"], errors="coerce")
    df_valuation = df_valuation.dropna(subset=["Date"])
    df_valuation = df_valuation.sort_values("Date")
    df_valuation = df_valuation.drop_duplicates(subset=["Date"], keep="last")

    candidate_cols = get_valuation_feature_columns() + get_extended_valuation_columns()
    available_cols = [col for col in candidate_cols if col in df_valuation.columns]

    for col in available_cols:
        df_valuation[col] = pd.to_numeric(df_valuation[col], errors="coerce")

    df_valuation = df_valuation[["Date"] + available_cols].copy()
    df_valuation = df_valuation.set_index("Date").sort_index()

    logger.info(
        "Loaded valuation features | ticker=%s | rows=%d | columns=%s",
        ticker,
        len(df_valuation),
        available_cols,
    )

    return df_valuation


def merge_valuation_features(df_features: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # Valuation data is expected to be point-in-time standardized.
    # Exact-date join avoids forward-filling PB/market cap across missing dates.
    df_features = df_features.copy().sort_index()
    valuation_df = load_valuation_features(ticker)

    required_cols = get_valuation_feature_columns()

    if valuation_df is None:
        for col in required_cols:
            if col not in df_features.columns:
                df_features[col] = np.nan
        return df_features

    df_features = df_features.join(valuation_df, how="left")

    for col in required_cols:
        if col not in df_features.columns:
            df_features[col] = np.nan

    return df_features


def build_feature_dataframe(
    ticker: str,
    df_vnindex: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if df_vnindex is None:
        df_vnindex = load_vnindex_data()

    stock_file = resolve_stock_file(ticker)
    df_stock = load_data(stock_file, is_vnindex=False)

    if df_stock is None:
        raise ValueError(f"Stock data could not be loaded: {ticker}")

    df = pd.merge(
        df_stock,
        df_vnindex,
        left_index=True,
        right_index=True,
        how="inner",
    ).sort_index()

    df_features = calculate_features(df)
    df_features = add_candidate_features(df_features)
    df_features = merge_valuation_features(df_features, ticker)

    return df_features.replace([np.inf, -np.inf], np.nan)


def get_feature_sets() -> dict[str, list[str]]:
    return registry_get_feature_sets()


def validate_feature_columns(
    df_features: pd.DataFrame,
    requested_columns: list[str],
) -> list[str]:
    missing_cols = [
        feature for feature in requested_columns
        if feature not in df_features.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Feature columns are missing from dataframe: {missing_cols}"
        )

    return requested_columns


def get_feature_columns(
    df_features: pd.DataFrame,
    feature_set: str,
) -> list[str]:
    feature_sets = get_feature_sets()

    if feature_set not in feature_sets:
        available = ", ".join(sorted(feature_sets.keys()))
        raise ValueError(
            f"Unknown feature_set={feature_set}. Available feature sets: {available}"
        )

    return validate_feature_columns(df_features, feature_sets[feature_set])


__all__ = [
    "PROJECT_ROOT",
    "PROCESSED_DATA_DIR",
    "VNINDEX_FILE",
    "STOCK_FILES",
    "VALUATION_DATA_DIR",
    "build_feature_dataframe",
    "calculate_targets",
    "get_feature_columns",
    "get_feature_sets",
    "get_stock_file_map",
    "load_valuation_features",
    "load_vnindex_data",
    "merge_valuation_features",
    "resolve_stock_file",
    "validate_feature_columns",
]
