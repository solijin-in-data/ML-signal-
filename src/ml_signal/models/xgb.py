from __future__ import annotations

import numpy as np
import xgboost as xgb


try:
    from config import (
        LEARNING_RATE,
        MAX_DEPTH,
        N_ESTIMATORS,
        RANDOM_STATE,
    )
except Exception:
    N_ESTIMATORS = 150
    MAX_DEPTH = 4
    LEARNING_RATE = 0.05
    RANDOM_STATE = 42


def calculate_scale_pos_weight(y_train: np.ndarray) -> float:
    pos_cases = np.sum(y_train == 1)
    neg_cases = np.sum(y_train == 0)

    return 1.0 if pos_cases == 0 else neg_cases / pos_cases


def create_xgb_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
