from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def parse_number(value):
    """
    Parse numeric values from standardized or semi-standardized market data.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    if text in ["", "-", "--", "nan", "NaN", "None"]:
        return np.nan

    text = text.replace(" ", "").replace(",", "")

    multiplier = 1.0

    if text.upper().endswith("K"):
        multiplier = 1_000
        text = text[:-1]
    elif text.upper().endswith("M"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.upper().endswith("B"):
        multiplier = 1_000_000_000
        text = text[:-1]

    if "%" in text:
        text = text.replace("%", "")
        multiplier = multiplier / 100

    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def load_data(file_path: Path, is_vnindex: bool = False) -> pd.DataFrame | None:
    """
    Load standardized stock or VNINDEX data.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error("File not found | path=%s", file_path)
        return None

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception as exc:
        logger.error("Failed to read file | path=%s | error=%s", file_path, exc)
        return None

    df.columns = df.columns.astype(str).str.strip()

    if is_vnindex:
        if "VN_Date" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"VN_Date": "Date"})
        required_cols = ["Date", "VN_Close"]
    else:
        required_cols = ["Date", "Close", "Volume"]
        if "Net_Volume_Foreign" in df.columns:
            required_cols.append("Net_Volume_Foreign")

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.error(
            "Missing required columns | file=%s | columns=%s",
            file_path.name,
            missing_cols,
        )
        return None

    df = df[required_cols].copy()

    for col in required_cols:
        if col != "Date":
            df[col] = df[col].apply(parse_number)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date").sort_index()

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Data is not sorted in ascending time order: {file_path}")

    logger.info(
        "Loaded data | file=%s | rows=%d | range=%s to %s",
        file_path.name,
        len(df),
        df.index.min().date(),
        df.index.max().date(),
    )

    return df
