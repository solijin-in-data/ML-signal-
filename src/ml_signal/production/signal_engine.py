from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_signal.evaluation.breakeven import calculate_breakeven_precision
from ml_signal.models.factory import calculate_scale_pos_weight, create_xgb_model
from ml_signal.pipelines.training_frame import build_latest_training_frame
from ml_signal.production.config_loader import ProductionSignalConfig
from ml_signal.production.signal_report import now_iso


SCRIPT_VERSION = "phase3_extract_research_utilities_v1"


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None

    if np.isnan(out) or np.isinf(out):
        return None

    return out


def load_research_evidence(
    ticker: str,
    profile: str,
    feature_set: str,
    lookahead: int,
    tp: float,
    sl: float,
    threshold: float,
    round_trip_cost: float,
) -> dict[str, Any]:
    """
    Load latest frozen-grid evidence when available.

    The signal engine can still run without this file. Evidence is included so
    daily signals remain connected to the research validation process.
    """
    from ml_signal.pipelines.feature_builder import PROJECT_ROOT

    evidence_path = (
        PROJECT_ROOT
        / "outputs"
        / "walkforward_frozen_grid"
        / f"{ticker}_{profile}_frozen_grid_stability_score.csv"
    )

    default = {
        "source": str(evidence_path) if evidence_path.exists() else None,
        "pass_periods": None,
        "min_edge": None,
        "avg_return": None,
        "round_trip_cost": round_trip_cost,
    }

    if not evidence_path.exists():
        return default

    try:
        df = pd.read_csv(evidence_path, encoding="utf-8-sig")
    except Exception:
        return default

    required = {"Feature_Set", "LOOKAHEAD", "TP", "SL", "THRESH"}

    if not required.issubset(df.columns):
        return default

    mask = (
        df["Feature_Set"].astype(str).eq(feature_set)
        & np.isclose(pd.to_numeric(df["LOOKAHEAD"], errors="coerce"), lookahead)
        & np.isclose(pd.to_numeric(df["TP"], errors="coerce"), tp)
        & np.isclose(pd.to_numeric(df["SL"], errors="coerce"), sl)
        & np.isclose(pd.to_numeric(df["THRESH"], errors="coerce"), threshold)
    )

    out = df[mask].copy()

    if out.empty:
        return default

    row = out.iloc[0]

    return {
        "source": str(evidence_path),
        "pass_periods": _safe_float(row.get("Pass_Periods")),
        "min_edge": _safe_float(row.get("Min_Edge")),
        "avg_return": _safe_float(row.get("Avg_Return")),
        "round_trip_cost": round_trip_cost,
    }


def get_top_model_features(
    model: Any,
    feature_cols: list[str],
    top: int = 10,
) -> list[dict[str, Any]]:
    importances = getattr(model, "feature_importances_", None)

    if importances is None:
        return []

    pairs = sorted(
        zip(feature_cols, importances),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [
        {
            "rank": idx,
            "feature": feature,
            "importance": float(importance),
        }
        for idx, (feature, importance) in enumerate(pairs[:top], start=1)
    ]


def decide_action(
    probability: float,
    threshold: float,
    breakeven_precision: float,
    min_edge: float,
    watch_margin: float,
) -> tuple[str, str]:
    min_required_precision = breakeven_precision + min_edge

    if probability >= threshold and probability >= min_required_precision:
        return "BUY", "Probability is above both the model threshold and the cost-adjusted minimum precision."

    if probability >= threshold:
        return "WATCHLIST", "Probability is above threshold but below the cost-adjusted minimum precision."

    if probability >= threshold - watch_margin:
        return "WATCHLIST", "Probability is near the threshold."

    return "HOLD", "Probability is below the threshold."


def run_latest_signal(config: ProductionSignalConfig) -> dict[str, Any]:
    df_features, df_train, feature_cols, latest_feature_date = build_latest_training_frame(
        ticker=config.ticker,
        feature_set=config.feature_set,
        lookahead=config.lookahead,
        tp=config.tp,
        sl=config.sl,
    )

    X_train_raw = df_train[feature_cols].values
    y_train = df_train["Target"].astype(int).values

    latest_row = df_features.loc[[latest_feature_date]].copy()
    latest_x_raw = latest_row[feature_cols].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    latest_x = scaler.transform(latest_x_raw)

    model = create_xgb_model(
        calculate_scale_pos_weight(y_train)
    )

    model.fit(X_train, y_train)
    probability = float(model.predict_proba(latest_x)[0, 1])

    breakeven_precision = calculate_breakeven_precision(
        tp=config.tp,
        sl=config.sl,
        round_trip_cost=config.round_trip_cost,
    )

    min_required_precision = breakeven_precision + config.min_edge_vs_breakeven

    action, reason = decide_action(
        probability=probability,
        threshold=config.threshold,
        breakeven_precision=breakeven_precision,
        min_edge=config.min_edge_vs_breakeven,
        watch_margin=config.watch_margin,
    )

    latest_data_date = df_features.index.max()
    data_lag_days = int((latest_data_date - latest_feature_date).days)

    evidence = load_research_evidence(
        ticker=config.ticker,
        profile=config.profile,
        feature_set=config.feature_set,
        lookahead=config.lookahead,
        tp=config.tp,
        sl=config.sl,
        threshold=config.threshold,
        round_trip_cost=config.round_trip_cost,
    )

    payload: dict[str, Any] = {
        "ticker": config.ticker,
        "profile": config.profile,
        "candidate_name": config.candidate_name,
        "setup": {
            "feature_set": config.feature_set,
            "lookahead": config.lookahead,
            "tp": config.tp,
            "sl": config.sl,
            "threshold": config.threshold,
            "round_trip_cost": config.round_trip_cost,
            "min_edge_vs_breakeven": config.min_edge_vs_breakeven,
        },
        "data": {
            "latest_feature_date": latest_feature_date.strftime("%Y-%m-%d"),
            "latest_data_date": latest_data_date.strftime("%Y-%m-%d"),
            "data_lag_days": data_lag_days,
            "latest_close": _safe_float(latest_row["Close"].iloc[0]),
            "training_rows": int(len(df_train)),
            "feature_columns": len(feature_cols),
        },
        "signal": {
            "probability": probability,
            "threshold": config.threshold,
            "breakeven_precision": breakeven_precision,
            "min_required_precision": min_required_precision,
            "action": action,
            "reason": reason,
        },
        "research_evidence": evidence,
        "top_model_features": get_top_model_features(model, feature_cols, top=10),
        "script_version": SCRIPT_VERSION,
        "generated_at": now_iso(),
    }

    return payload


__all__ = [
    "SCRIPT_VERSION",
    "decide_action",
    "get_top_model_features",
    "load_research_evidence",
    "run_latest_signal",
]
