from __future__ import annotations

import numpy as np
import pandas as pd


LIQUIDITY_COST_COLUMNS = [
    "Dollar_Volume",
    "Log_Dollar_Volume",
    "Dollar_Volume_Z_60",
    "Volume_Dry_Up_20",
    "Amihud_Illiquidity_20",
    "Liquidity_Shock_20",
]


def add_liquidity_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["Dollar_Volume"] = df["Close"] * df["Volume"]
    df["Log_Dollar_Volume"] = np.log(df["Dollar_Volume"] + 1e-9)

    dollar_volume_mean_60 = df["Dollar_Volume"].rolling(60).mean()
    dollar_volume_std_60 = df["Dollar_Volume"].rolling(60).std()

    df["Dollar_Volume_Z_60"] = (
        (df["Dollar_Volume"] - dollar_volume_mean_60)
        / (dollar_volume_std_60 + 1e-9)
    )

    df["Volume_Dry_Up_20"] = (
        df["Volume"]
        / (df["Volume"].rolling(20).mean() + 1e-9)
    )

    raw_amihud = df["Log_Return"].abs() / (df["Dollar_Volume"] + 1e-9)
    df["Amihud_Illiquidity_20"] = raw_amihud.rolling(20).mean()

    df["Liquidity_Shock_20"] = (
        df["Dollar_Volume"]
        / (df["Dollar_Volume"].rolling(20).mean() + 1e-9)
    )

    return df
