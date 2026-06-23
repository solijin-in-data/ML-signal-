from __future__ import annotations

"""
Compatibility layer for the legacy research scripts.

`feature_experiment_runner.py` can import this module as `radar` and keep using
the old function names while the implementation gradually moves into src/.
"""

from ml_signal.data.loaders import load_data, parse_number
from ml_signal.features.base_price_volume import calculate_features, get_feature_columns
from ml_signal.labels.tp_sl import calculate_targets
from ml_signal.models.xgb import calculate_scale_pos_weight, create_xgb_model


__all__ = [
    "parse_number",
    "load_data",
    "calculate_features",
    "get_feature_columns",
    "calculate_targets",
    "calculate_scale_pos_weight",
    "create_xgb_model",
]
