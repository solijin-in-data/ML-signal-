from __future__ import annotations

from typing import Any

import numpy as np

from ml_signal.models.xgb import (
    calculate_scale_pos_weight as _calculate_scale_pos_weight,
    create_xgb_model as _create_xgb_model,
)


def calculate_scale_pos_weight(y_train: np.ndarray) -> float:
    return float(_calculate_scale_pos_weight(y_train))


def create_xgb_model(scale_pos_weight: float) -> Any:
    return _create_xgb_model(scale_pos_weight)


__all__ = [
    "calculate_scale_pos_weight",
    "create_xgb_model",
]
