from __future__ import annotations

import numpy as np
import pandas as pd


VOLUME_COLUMNS = [
    "Volume_Ratio_20",
    "Log_Volume_Change",
    "Foreign_Net_5D_Ratio",
]


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    df["Volume_Ratio_20"] = (
        df["Volume"]
        / (df["Volume"].rolling(20).mean() + 1e-9)
    )

    df["Log_Volume_Change"] = np.log(
        (df["Volume"] + 1e-9)
        / (df["Volume"].shift(1) + 1e-9)
    )

    if "Net_Volume_Foreign" in df.columns:
        df["Foreign_Net_5D_Ratio"] = (
            df["Net_Volume_Foreign"].rolling(5).sum()
            / (df["Volume"].rolling(5).sum() + 1e-9)
        )
    else:
        df["Foreign_Net_5D_Ratio"] = 0.0

    return df
