from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


MODULE_PATH = Path(__file__).resolve()
PROJECT_ROOT = MODULE_PATH.parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

for path in [PROJECT_ROOT, SRC_ROOT]:
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


# Transitional bridge:
# XGBoost factory settings are still sourced from the mature research runner.
import feature_experiment_runner as exp  # noqa: E402


def calculate_scale_pos_weight(y_train: np.ndarray) -> float:
    return float(exp.radar.calculate_scale_pos_weight(y_train))


def create_xgb_model(scale_pos_weight: float) -> Any:
    return exp.radar.create_xgb_model(scale_pos_weight)


__all__ = [
    "calculate_scale_pos_weight",
    "create_xgb_model",
]
