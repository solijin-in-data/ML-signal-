from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate baseline price, volume, foreign flow, and market-relative features.
    """
    df = df.copy().sort_index()

    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["Momentum_14"] = df["Close"] - df["Close"].shift(14)

    df["EMA_13"] = df["Close"].ewm(span=13, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_13_21_Cross"] = (df["EMA_13"] - df["EMA_21"]) / (df["EMA_21"] + 1e-9)
    df["Dist_EMA_13"] = (df["Close"] - df["EMA_13"]) / (df["EMA_13"] + 1e-9)

    ema_10 = df["Close"].ewm(span=10, adjust=False).mean()
    ema_50 = df["Close"].ewm(span=50, adjust=False).mean()
    df["MACD_10_50"] = ema_10 - ema_50
    df["MACD_Signal_100"] = df["MACD_10_50"].ewm(span=100, adjust=False).mean()
    df["MACD_Hist_Custom"] = df["MACD_10_50"] - df["MACD_Signal_100"]

    df["BB_Mid"] = df["Close"].rolling(window=20).mean()
    df["BB_Std"] = df["Close"].rolling(window=20).std(ddof=0)
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    df["Dist_BB_Upper"] = (df["Close"] - df["BB_Upper"]) / (df["BB_Upper"] + 1e-9)
    df["Dist_BB_Lower"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Lower"] + 1e-9)

    df["Volatility"] = df["Log_Return"].rolling(window=20).std()

    df["VN_Return"] = np.log(df["VN_Close"] / df["VN_Close"].shift(1))
    df["VN_Volatility"] = df["VN_Return"].rolling(window=20).std()
    df["Relative_Strength"] = df["Log_Return"] - df["VN_Return"]
    df["RS_Trend"] = df["Relative_Strength"].rolling(window=10).mean()
    df["VN_EMA20"] = df["VN_Close"].ewm(span=20, adjust=False).mean()
    df["Market_Distance_EMA"] = (df["VN_Close"] - df["VN_EMA20"]) / (df["VN_EMA20"] + 1e-9)

    if "Net_Volume_Foreign" in df.columns:
        df["Foreign_Net_5D"] = df["Net_Volume_Foreign"].rolling(window=5).sum()
        df["Foreign_Net_20D_Mean"] = df["Net_Volume_Foreign"].rolling(window=20).mean()
        df["Foreign_Net_20D_Std"] = df["Net_Volume_Foreign"].rolling(window=20).std()
        df["Foreign_Mutation"] = (
            (df["Net_Volume_Foreign"] - df["Foreign_Net_20D_Mean"])
            / (df["Foreign_Net_20D_Std"] + 1e-9)
        )
    else:
        df["Foreign_Net_5D"] = 0.0
        df["Foreign_Net_20D_Mean"] = 0.0
        df["Foreign_Mutation"] = 0.0

    return df


def get_feature_columns() -> list[str]:
    return [
        "Log_Return",
        "RSI_14",
        "Momentum_14",
        "Volatility",
        "Volume",
        "MACD_Hist_Custom",
        "EMA_13_21_Cross",
        "Dist_EMA_13",
        "Dist_BB_Upper",
        "Dist_BB_Lower",
        "VN_Return",
        "VN_Volatility",
        "Relative_Strength",
        "RS_Trend",
        "Market_Distance_EMA",
        "Foreign_Net_5D",
        "Foreign_Net_20D_Mean",
        "Foreign_Mutation",
    ]
