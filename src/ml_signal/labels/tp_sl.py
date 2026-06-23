from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_targets(df: pd.DataFrame, lookahead: int, tp: float, sl: float) -> pd.DataFrame:
    """
    Create TP/SL first-hit labels.
    """
    df = df.copy().sort_index()

    targets = []
    holding_days = []
    exit_returns = []
    exit_reasons = []
    timeouts = []

    close_prices = df["Close"].values
    n_samples = len(close_prices)

    for i in range(n_samples):
        if i + lookahead >= n_samples:
            targets.append(np.nan)
            holding_days.append(np.nan)
            exit_returns.append(np.nan)
            exit_reasons.append(np.nan)
            timeouts.append(np.nan)
            continue

        entry_price = close_prices[i]
        label = 0
        hold_days = lookahead
        exit_return = (close_prices[i + lookahead] - entry_price) / entry_price
        exit_reason = "timeout"
        timeout = 1

        for j in range(1, lookahead + 1):
            future_return = (close_prices[i + j] - entry_price) / entry_price

            if future_return <= sl:
                label = 0
                hold_days = j
                exit_return = future_return
                exit_reason = "sl"
                timeout = 0
                break

            if future_return >= tp:
                label = 1
                hold_days = j
                exit_return = future_return
                exit_reason = "tp"
                timeout = 0
                break

        targets.append(label)
        holding_days.append(hold_days)
        exit_returns.append(exit_return)
        exit_reasons.append(exit_reason)
        timeouts.append(timeout)

    df["Target"] = targets
    df["Holding_Days"] = holding_days
    df["Exit_Return"] = exit_returns
    df["Exit_Reason"] = exit_reasons
    df["Timeout"] = timeouts

    return df.replace([np.inf, -np.inf], np.nan).dropna()
