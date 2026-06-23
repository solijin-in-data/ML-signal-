from __future__ import annotations

import numpy as np
import pandas as pd


MOMENTUM_COLUMNS = [
    "Abs_Momentum_14",
    "Log_Momentum_14",
    "EWM_Return_10",
    "Momentum_Dropoff_14",
    "Momentum_Dropoff_Z_14",
]


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["Abs_Momentum_14"] = df["Close"] - df["Close"].shift(14)

    df["Log_Momentum_14"] = np.log(
        df["Close"] / (df["Close"].shift(14) + 1e-9)
    )

    df["EWM_Return_10"] = df["Log_Return"].ewm(
        span=10,
        adjust=False,
    ).mean()

    df["Momentum_Dropoff_14"] = df["Log_Return"].shift(14)

    df["Momentum_Dropoff_Z_14"] = (
        df["Momentum_Dropoff_14"].abs()
        / (df["Log_Return"].rolling(60).std() + 1e-9)
    )

    return df
