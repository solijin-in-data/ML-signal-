from __future__ import annotations

import pandas as pd


RECOVERY_COLUMNS = [
    "Drawdown_60",
    "Distance_52W_High",
    "Distance_52W_Low",
]


def add_recovery_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    rolling_high_60 = df["Close"].rolling(60).max()
    rolling_high_252 = df["Close"].rolling(252).max()
    rolling_low_252 = df["Close"].rolling(252).min()

    df["Drawdown_60"] = (
        df["Close"] / (rolling_high_60 + 1e-9)
        - 1
    )

    df["Distance_52W_High"] = (
        df["Close"] / (rolling_high_252 + 1e-9)
        - 1
    )

    df["Distance_52W_Low"] = (
        df["Close"] / (rolling_low_252 + 1e-9)
        - 1
    )

    return df
