from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

for path in [PROJECT_ROOT, SRC_ROOT]:
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


from ml_signal.production.config_loader import (  # noqa: E402
    apply_overrides,
    load_production_signal_config,
)
from ml_signal.production.signal_report import now_iso, write_signal_outputs  # noqa: E402

import feature_experiment_runner as exp  # noqa: E402


SCRIPT_VERSION = "phase3_production_signal_runner_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the latest production-candidate signal for one ticker/profile."
    )

    parser.add_argument("--ticker", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--config", default=None)

    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--lookahead", type=int, default=None)
    parser.add_argument("--tp", type=float, default=None)
    parser.add_argument("--sl", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--round-trip-cost", type=float, default=None)
    parser.add_argument("--min-edge-vs-breakeven", type=float, default=None)
    parser.add_argument("--watch-margin", type=float, default=0.05)

    parser.add_argument(
        "--output-dir",
        default="reports/signals",
        help="Directory where latest signal markdown/json files are written.",
    )

    return parser.parse_args()


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
    Load the latest frozen-grid evidence when available.

    The signal runner can still run without this file; the evidence block is
    included to keep daily signal output connected to the research process.
    """
    evidence_path = (
        PROJECT_ROOT
        / "outputs"
        / "walkforward_frozen_grid"
        / f"{ticker}_{profile}_frozen_grid_stability_score.csv"
    )

    if not evidence_path.exists():
        return {
            "source": None,
            "pass_periods": None,
            "min_edge": None,
            "avg_return": None,
            "round_trip_cost": round_trip_cost,
        }

    try:
        df = pd.read_csv(evidence_path, encoding="utf-8-sig")
    except Exception:
        return {
            "source": str(evidence_path),
            "pass_periods": None,
            "min_edge": None,
            "avg_return": None,
            "round_trip_cost": round_trip_cost,
        }

    required = {"Feature_Set", "LOOKAHEAD", "TP", "SL", "THRESH"}

    if not required.issubset(df.columns):
        return {
            "source": str(evidence_path),
            "pass_periods": None,
            "min_edge": None,
            "avg_return": None,
            "round_trip_cost": round_trip_cost,
        }

    mask = (
        df["Feature_Set"].astype(str).eq(feature_set)
        & np.isclose(pd.to_numeric(df["LOOKAHEAD"], errors="coerce"), lookahead)
        & np.isclose(pd.to_numeric(df["TP"], errors="coerce"), tp)
        & np.isclose(pd.to_numeric(df["SL"], errors="coerce"), sl)
        & np.isclose(pd.to_numeric(df["THRESH"], errors="coerce"), threshold)
    )

    out = df[mask].copy()

    if out.empty:
        return {
            "source": str(evidence_path),
            "pass_periods": None,
            "min_edge": None,
            "avg_return": None,
            "round_trip_cost": round_trip_cost,
        }

    row = out.iloc[0]

    return {
        "source": str(evidence_path),
        "pass_periods": _safe_float(row.get("Pass_Periods")),
        "min_edge": _safe_float(row.get("Min_Edge")),
        "avg_return": _safe_float(row.get("Avg_Return")),
        "round_trip_cost": round_trip_cost,
    }


def get_top_model_features(model: Any, feature_cols: list[str], top: int = 10) -> list[dict[str, Any]]:
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


def build_training_frame(
    ticker: str,
    feature_set: str,
    lookahead: int,
    tp: float,
    sl: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.Timestamp]:
    df_vnindex = exp.radar.load_data(exp.VNINDEX_FILE, is_vnindex=True)

    if df_vnindex is None:
        raise ValueError(f"VNINDEX data could not be loaded: {exp.VNINDEX_FILE}")

    df_features = exp.build_feature_dataframe(ticker, df_vnindex)

    feature_sets = exp.get_feature_sets()

    if feature_set not in feature_sets:
        available = ", ".join(sorted(feature_sets.keys()))
        raise ValueError(
            f"Unknown feature_set={feature_set}. Available feature sets: {available}"
        )

    feature_cols = exp.validate_feature_columns(df_features, feature_sets[feature_set])

    latest_candidates = (
        df_features
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=feature_cols + ["Close"])
    )

    if latest_candidates.empty:
        raise ValueError("No latest row has a complete feature vector.")

    latest_feature_date = latest_candidates.index.max()

    df_labeled = exp.radar.calculate_targets(
        df_features,
        lookahead=lookahead,
        tp=tp,
        sl=sl,
    )

    required_cols = feature_cols + [
        "Target",
        "Exit_Return",
        "Holding_Days",
        "Timeout",
    ]

    df_labeled = (
        df_labeled
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required_cols)
        .sort_index()
    )

    # Do not train on the row being scored, even when historical target values
    # exist for that date.
    df_train = df_labeled[df_labeled.index < latest_feature_date].copy()

    if len(df_train) < 100:
        raise ValueError(
            f"Insufficient labeled training rows before latest signal date: {len(df_train)}"
        )

    if df_train["Target"].nunique() < 2:
        raise ValueError("Training target has only one class.")

    return df_features, df_train, feature_cols, latest_feature_date


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


def run_latest_signal(args: argparse.Namespace) -> dict[str, Any]:
    config = load_production_signal_config(
        project_root=PROJECT_ROOT,
        ticker=args.ticker,
        profile=args.profile,
        config_path=args.config,
    )

    config = apply_overrides(
        config,
        candidate_name=args.candidate_name,
        feature_set=args.feature_set,
        lookahead=args.lookahead,
        tp=args.tp,
        sl=args.sl,
        threshold=args.threshold,
        round_trip_cost=args.round_trip_cost,
        min_edge_vs_breakeven=args.min_edge_vs_breakeven,
        watch_margin=args.watch_margin,
    )

    df_features, df_train, feature_cols, latest_feature_date = build_training_frame(
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

    model = exp.radar.create_xgb_model(
        exp.radar.calculate_scale_pos_weight(y_train)
    )

    model.fit(X_train, y_train)
    probability = float(model.predict_proba(latest_x)[0, 1])

    breakeven_precision = exp.calculate_breakeven_precision(
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


def main() -> None:
    args = parse_args()
    payload = run_latest_signal(args)

    md_path, json_path = write_signal_outputs(
        payload,
        output_dir=PROJECT_ROOT / args.output_dir,
    )

    print("=" * 100)
    print("Latest production-candidate signal generated")
    print("=" * 100)
    print(f"Ticker/Profile: {payload['ticker']} {payload['profile']}")
    print(f"Candidate:      {payload['candidate_name']}")
    print(f"Probability:    {payload['signal']['probability']:.4f}")
    print(f"Action:         {payload['signal']['action']}")
    print(f"Markdown:       {md_path}")
    print(f"JSON:           {json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
