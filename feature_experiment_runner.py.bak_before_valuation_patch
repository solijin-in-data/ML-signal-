from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import config as cfg
import main_signal_radar_v2 as radar


warnings.filterwarnings("ignore")


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# =============================================================================
# PATHS AND CONFIG FALLBACKS
# =============================================================================

PROJECT_ROOT = getattr(cfg, "PROJECT_ROOT", Path(__file__).resolve().parent)
OUTPUT_DIR = getattr(cfg, "OUTPUT_DIR", PROJECT_ROOT / "outputs")
PROCESSED_DATA_DIR = getattr(cfg, "PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed_data")
VNINDEX_FILE = getattr(cfg, "VNINDEX_FILE", PROCESSED_DATA_DIR / "VNINDEX_standardized.csv")
STOCK_TICKERS = getattr(cfg, "STOCK_TICKERS", [])
STOCK_FILES = getattr(cfg, "STOCK_FILES", [])
STRATEGY_PROFILES = getattr(cfg, "STRATEGY_PROFILES", {})

CV_SPLITS = getattr(cfg, "CV_SPLITS", 3)
RANDOM_STATE = getattr(cfg, "RANDOM_STATE", 42)

MIN_TRADES_REQUIRED = getattr(cfg, "MIN_TRADES_REQUIRED", 5)
MIN_VALID_FOLDS_REQUIRED = getattr(cfg, "MIN_VALID_FOLDS_REQUIRED", 1)

EXPECTANCY_PER_DAY_SCALE = getattr(cfg, "EXPECTANCY_PER_DAY_SCALE", 30.0)
PRECISION_STD_PENALTY_WEIGHT = getattr(cfg, "PRECISION_STD_PENALTY_WEIGHT", 0.30)
TIMEOUT_PENALTY_WEIGHT = getattr(cfg, "TIMEOUT_PENALTY_WEIGHT", 0.10)
HOLDING_PENALTY_WEIGHT = getattr(cfg, "HOLDING_PENALTY_WEIGHT", 0.05)
TRADE_BONUS_CAP = getattr(cfg, "TRADE_BONUS_CAP", 50)
TRADE_BONUS_PER_TRADE = getattr(cfg, "TRADE_BONUS_PER_TRADE", 0.001)

FEATURE_EXPERIMENT_DIR = OUTPUT_DIR / "feature_experiments"
FEATURE_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

BEST_PARAMS_FILE = OUTPUT_DIR / "best_params.csv"
SETUP_DIAGNOSTIC_DIR = OUTPUT_DIR / "setup_diagnostics"


# =============================================================================
# FEATURE SET DEFINITIONS
# =============================================================================

def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    output = []

    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)

    return output


def get_feature_sets() -> dict[str, list[str]]:
    """
    Define baseline and candidate feature sets.

    Baseline uses main_signal_radar_v2.get_feature_columns().
    Candidate sets add new features computed in add_candidate_features().
    """
    baseline = radar.get_feature_columns()

    feature_sets = {
        "baseline": baseline,

        "candidate_momentum_v1": baseline + [
            "Abs_Momentum_14",
            "Log_Momentum_14",
            "EWM_Return_10",
            "Momentum_Dropoff_Z_14",
        ],

        "candidate_trend_quality_v1": baseline + [
            "ER_10",
            "EMA_13_Slope",
            "EMA_21_55_Gap",
            "BB_Position",
        ],

        "candidate_volume_v1": baseline + [
            "Volume_Ratio_20",
            "Log_Volume_Change",
            "Foreign_Net_5D_Ratio",
        ],

        "candidate_recovery_v1": baseline + [
            "Drawdown_60",
            "Distance_52W_High",
            "Distance_52W_Low",
        ],

        "candidate_light_combo_v1": baseline + [
            "Log_Momentum_14",
            "EWM_Return_10",
            "Momentum_Dropoff_Z_14",
            "ER_10",
            "Volume_Ratio_20",
            "Drawdown_60",
            "Distance_52W_High",
        ],
    }

    return {
        name: unique_preserve_order(cols)
        for name, cols in feature_sets.items()
    }


# =============================================================================
# CANDIDATE FEATURE ENGINEERING
# =============================================================================

def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add experimental features on top of the baseline features.

    All features use only current and past data at time t.
    No future values, centered rolling windows, or global full-sample statistics are used.
    """
    df = df.copy().sort_index()

    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # -------------------------------------------------------------------------
    # Momentum and drop-off diagnostics
    # -------------------------------------------------------------------------
    df["Abs_Momentum_14"] = df["Close"] - df["Close"].shift(14)

    df["Log_Momentum_14"] = np.log(
        df["Close"] / (df["Close"].shift(14) + 1e-9)
    )

    df["EWM_Return_10"] = df["Log_Return"].ewm(
        span=10,
        adjust=False,
    ).mean()

    df["Momentum_Dropoff_14"] = df["Log_Return"].shift(14)

    df["Momentum_Dropoff_Z_14"] = (
        df["Momentum_Dropoff_14"].abs()
        / (df["Log_Return"].rolling(60).std() + 1e-9)
    )

    # -------------------------------------------------------------------------
    # Trend quality
    # -------------------------------------------------------------------------
    if "EMA_13" not in df.columns:
        df["EMA_13"] = df["Close"].ewm(span=13, adjust=False).mean()

    if "EMA_21" not in df.columns:
        df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()

    df["EMA_55"] = df["Close"].ewm(span=55, adjust=False).mean()

    df["EMA_13_Slope"] = df["EMA_13"].pct_change()

    df["EMA_21_55_Gap"] = (
        (df["EMA_21"] - df["EMA_55"])
        / (df["EMA_55"] + 1e-9)
    )

    df["ER_10"] = (
        (df["Close"] - df["Close"].shift(10)).abs()
        / (df["Close"].diff().abs().rolling(10).sum() + 1e-9)
    )

    if "BB_Mid" not in df.columns:
        df["BB_Mid"] = df["Close"].rolling(window=20).mean()

    if "BB_Std" not in df.columns:
        df["BB_Std"] = df["Close"].rolling(window=20).std(ddof=0)

    df["BB_Position"] = (
        (df["Close"] - df["BB_Mid"])
        / (2 * df["BB_Std"] + 1e-9)
    )

    # -------------------------------------------------------------------------
    # Volume and foreign flow
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Recovery and drawdown structure
    # -------------------------------------------------------------------------
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

    return df.replace([np.inf, -np.inf], np.nan)


# =============================================================================
# DATA LOADING
# =============================================================================

def get_stock_file_map() -> dict[str, Path]:
    file_map = {}

    for file_path in STOCK_FILES:
        ticker = Path(file_path).stem.replace("_standardized", "").upper()
        file_map[ticker] = Path(file_path)

    return file_map


def resolve_stock_file(ticker: str) -> Path:
    ticker = ticker.upper()
    file_map = get_stock_file_map()

    if ticker in file_map:
        return file_map[ticker]

    fallback_path = PROCESSED_DATA_DIR / f"{ticker}_standardized.csv"

    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        f"Cannot find standardized data file for ticker: {ticker}"
    )


def build_feature_dataframe(ticker: str, df_vnindex: pd.DataFrame) -> pd.DataFrame:
    stock_file = resolve_stock_file(ticker)
    df_stock = radar.load_data(stock_file, is_vnindex=False)

    if df_stock is None:
        raise ValueError(f"Stock data could not be loaded: {ticker}")

    df = pd.merge(
        df_stock,
        df_vnindex,
        left_index=True,
        right_index=True,
        how="inner",
    ).sort_index()

    df_features = radar.calculate_features(df)
    df_features = add_candidate_features(df_features)

    return df_features


# =============================================================================
# FEATURE HEALTH
# =============================================================================

def create_feature_health_report(
    ticker: str,
    feature_set_name: str,
    df_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    records = []

    for feature in feature_cols:
        if feature not in df_features.columns:
            records.append(
                {
                    "Ticker": ticker,
                    "Feature_Set": feature_set_name,
                    "Feature": feature,
                    "Missing_Pct": 1.0,
                    "Unique_Count": 0,
                    "Mean": np.nan,
                    "Std": np.nan,
                    "Zero_Variance": True,
                    "Status": "missing_column",
                }
            )
            continue

        series = df_features[feature].replace([np.inf, -np.inf], np.nan)

        missing_pct = float(series.isna().mean())
        unique_count = int(series.nunique(dropna=True))
        mean_value = float(series.mean()) if series.notna().any() else np.nan
        std_value = float(series.std()) if series.notna().any() else np.nan

        zero_variance = bool(
            unique_count <= 1
            or pd.isna(std_value)
            or std_value == 0
        )

        if zero_variance:
            status = "zero_variance"
        elif missing_pct > 0.30:
            status = "high_missing"
        else:
            status = "ok"

        records.append(
            {
                "Ticker": ticker,
                "Feature_Set": feature_set_name,
                "Feature": feature,
                "Missing_Pct": missing_pct,
                "Unique_Count": unique_count,
                "Mean": mean_value,
                "Std": std_value,
                "Zero_Variance": zero_variance,
                "Status": status,
            }
        )

    return pd.DataFrame(records)


def validate_feature_columns(df_features: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    missing_cols = [
        feature for feature in feature_cols
        if feature not in df_features.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Feature columns are missing from dataframe: {missing_cols}"
        )

    return feature_cols


# =============================================================================
# METRICS
# =============================================================================

def calculate_breakeven_precision(
    tp: float,
    sl: float,
    round_trip_cost: float = 0.0,
) -> float:
    """
    Breakeven hit rate based on TP/SL payoff.

    Example:
    TP = 0.10, SL = -0.07
    Breakeven = 0.07 / (0.10 + 0.07)
    """
    win_return = tp - round_trip_cost
    loss_return = abs(sl) + round_trip_cost

    if win_return <= 0:
        return 1.0

    return loss_return / (win_return + loss_return)


def create_empty_eval_stats() -> dict[str, Any]:
    return {
        "valid_folds": 0,
        "total_trades": 0,
        "total_correct": 0,
        "test_positive": 0,
        "test_count": 0,
        "precision_values": [],
        "return_sum": 0.0,
        "holding_sum": 0.0,
        "timeout_sum": 0.0,
    }


def update_eval_stats(
    stats: dict[str, Any],
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    predictions = (probabilities >= threshold).astype(int)
    selected_mask = predictions == 1
    num_trades = int(np.sum(selected_mask))

    stats["test_positive"] += int(np.sum(y_test == 1))
    stats["test_count"] += int(len(y_test))

    if num_trades == 0:
        return

    selected_trades = df_test.iloc[np.where(selected_mask)[0]]

    correct_trades = int(selected_trades["Target"].sum())
    precision = correct_trades / num_trades

    stats["valid_folds"] += 1
    stats["total_trades"] += num_trades
    stats["total_correct"] += correct_trades
    stats["precision_values"].append(precision)
    stats["return_sum"] += float(selected_trades["Exit_Return"].sum())
    stats["holding_sum"] += float(selected_trades["Holding_Days"].sum())
    stats["timeout_sum"] += float(selected_trades["Timeout"].sum())


def finalize_eval_stats(
    stats: dict[str, Any],
    profile_name: str,
    profile_config: dict,
    lookahead: int,
    tp: float,
    sl: float,
    threshold: float,
    round_trip_cost: float,
    min_edge_vs_breakeven: float,
) -> dict[str, Any]:
    total_trades = stats["total_trades"]

    breakeven_precision = calculate_breakeven_precision(
        tp=tp,
        sl=sl,
        round_trip_cost=round_trip_cost,
    )

    if total_trades == 0:
        return {
            "LOOKAHEAD": lookahead,
            "TP": tp,
            "SL": sl,
            "THRESH": threshold,
            "Score": np.nan,
            "Validation_Precision": 0.0,
            "Precision_Std": np.nan,
            "Base_Rate": np.nan,
            "Excess_Precision": np.nan,
            "Precision_Lift": np.nan,
            "Breakeven_Precision": breakeven_precision,
            "Edge_vs_Breakeven": np.nan,
            "Trades": 0,
            "Valid_Folds": 0,
            "Avg_Return": np.nan,
            "Avg_Holding_Days": np.nan,
            "Expectancy_Per_Day": np.nan,
            "Timeout_Rate": np.nan,
            "Research_Status": "NO_SIGNAL",
            "Pass_Tradeable_Rule": False,
        }

    avg_precision = stats["total_correct"] / total_trades
    base_rate = stats["test_positive"] / max(stats["test_count"], 1)

    excess_precision = avg_precision - base_rate
    precision_lift = avg_precision / (base_rate + 1e-9)
    edge_vs_breakeven = avg_precision - breakeven_precision

    avg_return = stats["return_sum"] / total_trades
    avg_holding_days = stats["holding_sum"] / total_trades
    expectancy_per_day = avg_return / (avg_holding_days + 1e-9)
    timeout_rate = stats["timeout_sum"] / total_trades

    precision_values = stats["precision_values"]
    precision_std = (
        float(np.std(precision_values, ddof=0))
        if len(precision_values) > 1
        else 0.0
    )

    max_avg_holding_days = profile_config.get("max_avg_holding_days", lookahead)
    max_timeout_rate = profile_config.get("max_timeout_rate", 1.0)

    trade_bonus = min(total_trades, TRADE_BONUS_CAP) * TRADE_BONUS_PER_TRADE

    holding_penalty = (
        avg_holding_days / max_avg_holding_days
    ) * HOLDING_PENALTY_WEIGHT

    score = (
        excess_precision
        + expectancy_per_day * EXPECTANCY_PER_DAY_SCALE
        + trade_bonus
        - precision_std * PRECISION_STD_PENALTY_WEIGHT
        - timeout_rate * TIMEOUT_PENALTY_WEIGHT
        - holding_penalty
    )

    pass_tradeable_rule = (
        stats["valid_folds"] >= MIN_VALID_FOLDS_REQUIRED
        and total_trades >= MIN_TRADES_REQUIRED
        and excess_precision > 0
        and edge_vs_breakeven >= min_edge_vs_breakeven
        and avg_return > 0
        and expectancy_per_day > 0
        and avg_holding_days <= max_avg_holding_days
        and timeout_rate <= max_timeout_rate
    )

    if pass_tradeable_rule:
        research_status = "TRADEABLE_EDGE"
    elif excess_precision > 0 and avg_return > 0 and expectancy_per_day > 0:
        research_status = "RESEARCH_EDGE"
    else:
        research_status = "NO_EDGE"

    return {
        "LOOKAHEAD": lookahead,
        "TP": tp,
        "SL": sl,
        "THRESH": threshold,
        "Score": score,
        "Validation_Precision": avg_precision,
        "Precision_Std": precision_std,
        "Base_Rate": base_rate,
        "Excess_Precision": excess_precision,
        "Precision_Lift": precision_lift,
        "Breakeven_Precision": breakeven_precision,
        "Edge_vs_Breakeven": edge_vs_breakeven,
        "Trades": total_trades,
        "Valid_Folds": stats["valid_folds"],
        "Avg_Return": avg_return,
        "Avg_Holding_Days": avg_holding_days,
        "Expectancy_Per_Day": expectancy_per_day,
        "Timeout_Rate": timeout_rate,
        "Research_Status": research_status,
        "Pass_Tradeable_Rule": pass_tradeable_rule,
    }


# =============================================================================
# WALK-FORWARD EVALUATION
# =============================================================================

def evaluate_fixed_setup(
    ticker: str,
    profile_name: str,
    profile_config: dict,
    feature_set_name: str,
    df_features: pd.DataFrame,
    feature_cols: list[str],
    lookahead: int,
    tp: float,
    sl: float,
    threshold: float,
    round_trip_cost: float = 0.0,
    min_edge_vs_breakeven: float = 0.0,
) -> dict[str, Any]:
    feature_cols = validate_feature_columns(df_features, feature_cols)

    df_labeled = radar.calculate_targets(
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
    )

    if len(df_labeled) < 100:
        metrics = finalize_eval_stats(
            create_empty_eval_stats(),
            profile_name=profile_name,
            profile_config=profile_config,
            lookahead=lookahead,
            tp=tp,
            sl=sl,
            threshold=threshold,
            round_trip_cost=round_trip_cost,
            min_edge_vs_breakeven=min_edge_vs_breakeven,
        )

        metrics.update(
            {
                "Ticker": ticker,
                "Profile": profile_name,
                "Feature_Set": feature_set_name,
                "Experiment_Mode": "fixed_setup",
                "Rows": len(df_labeled),
                "Reject_Note": "insufficient_labeled_rows",
            }
        )

        return metrics

    X = df_labeled[feature_cols].values
    y = df_labeled["Target"].values

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    stats = create_empty_eval_stats()

    for train_idx, test_idx in tscv.split(X):
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model = radar.create_xgb_model(
            radar.calculate_scale_pos_weight(y_train)
        )

        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            logger.warning(
                "Model training failed | ticker=%s | profile=%s | feature_set=%s | error=%s",
                ticker,
                profile_name,
                feature_set_name,
                exc,
            )
            continue

        probabilities = model.predict_proba(X_test)[:, 1]
        df_test = df_labeled.iloc[test_idx].copy()

        update_eval_stats(
            stats=stats,
            df_test=df_test,
            y_test=y_test,
            probabilities=probabilities,
            threshold=threshold,
        )

    metrics = finalize_eval_stats(
        stats,
        profile_name=profile_name,
        profile_config=profile_config,
        lookahead=lookahead,
        tp=tp,
        sl=sl,
        threshold=threshold,
        round_trip_cost=round_trip_cost,
        min_edge_vs_breakeven=min_edge_vs_breakeven,
    )

    metrics.update(
        {
            "Ticker": ticker,
            "Profile": profile_name,
            "Feature_Set": feature_set_name,
            "Experiment_Mode": "fixed_setup",
            "Rows": len(df_labeled),
            "Reject_Note": "",
        }
    )

    return metrics


def evaluate_reoptimized_setup(
    ticker: str,
    profile_name: str,
    profile_config: dict,
    feature_set_name: str,
    df_features: pd.DataFrame,
    feature_cols: list[str],
    round_trip_cost: float = 0.0,
    min_edge_vs_breakeven: float = 0.0,
) -> dict[str, Any]:
    best_result = None
    evaluated_count = 0

    for lookahead in profile_config["lookahead"]:
        for tp in profile_config["tp"]:
            for sl in profile_config["sl"]:
                for threshold in profile_config["threshold"]:
                    evaluated_count += 1

                    result = evaluate_fixed_setup(
                        ticker=ticker,
                        profile_name=profile_name,
                        profile_config=profile_config,
                        feature_set_name=feature_set_name,
                        df_features=df_features,
                        feature_cols=feature_cols,
                        lookahead=lookahead,
                        tp=tp,
                        sl=sl,
                        threshold=threshold,
                        round_trip_cost=round_trip_cost,
                        min_edge_vs_breakeven=min_edge_vs_breakeven,
                    )

                    result["Experiment_Mode"] = "reoptimized"
                    result["Evaluated_Setups"] = evaluated_count

                    if pd.isna(result["Score"]):
                        continue

                    if best_result is None or result["Score"] > best_result["Score"]:
                        best_result = result

    if best_result is None:
        return {
            "Ticker": ticker,
            "Profile": profile_name,
            "Feature_Set": feature_set_name,
            "Experiment_Mode": "reoptimized",
            "LOOKAHEAD": np.nan,
            "TP": np.nan,
            "SL": np.nan,
            "THRESH": np.nan,
            "Score": np.nan,
            "Validation_Precision": 0.0,
            "Base_Rate": np.nan,
            "Excess_Precision": np.nan,
            "Breakeven_Precision": np.nan,
            "Edge_vs_Breakeven": np.nan,
            "Trades": 0,
            "Valid_Folds": 0,
            "Avg_Return": np.nan,
            "Avg_Holding_Days": np.nan,
            "Expectancy_Per_Day": np.nan,
            "Timeout_Rate": np.nan,
            "Research_Status": "NO_SIGNAL",
            "Pass_Tradeable_Rule": False,
            "Evaluated_Setups": evaluated_count,
            "Rows": 0,
            "Reject_Note": "no_valid_evaluation",
        }

    best_result["Evaluated_Setups"] = evaluated_count

    return best_result


# =============================================================================
# FIXED SETUP RESOLUTION
# =============================================================================

def normalize_setup_row(row: pd.Series, source: str) -> dict[str, Any]:
    def get_first_available(keys: list[str]) -> Any:
        for key in keys:
            if key in row.index:
                return row[key]
        raise KeyError(f"None of these columns exist: {keys}")

    return {
        "lookahead": int(get_first_available(["Lookahead", "LOOKAHEAD"])),
        "tp": float(get_first_available(["TP"])),
        "sl": float(get_first_available(["SL"])),
        "threshold": float(get_first_available(["Threshold", "THRESH"])),
        "setup_source": source,
    }


def load_best_params_setup(ticker: str, profile_name: str) -> dict[str, Any] | None:
    if not BEST_PARAMS_FILE.exists():
        return None

    df = pd.read_csv(BEST_PARAMS_FILE)

    if df.empty:
        return None

    ticker_col = "Ticker"
    profile_col = "Profile"

    if ticker_col not in df.columns or profile_col not in df.columns:
        return None

    mask = (
        df[ticker_col].astype(str).str.upper().eq(ticker.upper())
        & df[profile_col].astype(str).str.upper().eq(profile_name.upper())
    )

    rows = df.loc[mask].copy()

    if rows.empty:
        return None

    if "Score" in rows.columns:
        rows = rows.sort_values(by="Score", ascending=False)

    return normalize_setup_row(rows.iloc[0], source="best_params")


def load_diagnostics_fallback_setup(ticker: str, profile_name: str) -> dict[str, Any] | None:
    diagnostic_path = SETUP_DIAGNOSTIC_DIR / f"{ticker}_{profile_name}_setup_diagnostics.csv"

    if not diagnostic_path.exists():
        return None

    df = pd.read_csv(diagnostic_path)

    if df.empty:
        return None

    df = df.dropna(subset=["Score"])

    if df.empty:
        return None

    df = df.sort_values(by="Score", ascending=False)

    return normalize_setup_row(df.iloc[0], source="setup_diagnostics_fallback")


def resolve_fixed_setup(
    ticker: str,
    profile_name: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    explicit_values = [
        args.lookahead,
        args.tp,
        args.sl,
        args.threshold,
    ]

    if all(value is not None for value in explicit_values):
        return {
            "lookahead": int(args.lookahead),
            "tp": float(args.tp),
            "sl": float(args.sl),
            "threshold": float(args.threshold),
            "setup_source": "cli_explicit",
        }

    setup = load_best_params_setup(ticker, profile_name)

    if setup is not None:
        return setup

    if not args.no_diagnostics_fallback:
        setup = load_diagnostics_fallback_setup(ticker, profile_name)

        if setup is not None:
            return setup

    return None


# =============================================================================
# COMPARISON REPORTS
# =============================================================================

def add_delta_vs_baseline(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    metrics = [
        "Score",
        "Validation_Precision",
        "Excess_Precision",
        "Edge_vs_Breakeven",
        "Avg_Return",
        "Expectancy_Per_Day",
        "Avg_Holding_Days",
        "Timeout_Rate",
        "Trades",
    ]

    results_df = results_df.copy()

    for metric in metrics:
        results_df[f"{metric}_Delta_vs_Baseline"] = np.nan

    group_cols = ["Ticker", "Profile", "Experiment_Mode"]

    for _, group in results_df.groupby(group_cols):
        baseline_rows = group[group["Feature_Set"] == "baseline"]

        if baseline_rows.empty:
            continue

        baseline = baseline_rows.iloc[0]

        for idx in group.index:
            for metric in metrics:
                if metric in results_df.columns and pd.notna(baseline.get(metric, np.nan)):
                    results_df.loc[idx, f"{metric}_Delta_vs_Baseline"] = (
                        results_df.loc[idx, metric] - baseline[metric]
                    )

    return results_df


def create_feature_set_decision(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    records = []

    candidate_df = results_df[results_df["Feature_Set"] != "baseline"].copy()

    for _, row in candidate_df.iterrows():
        score_delta = row.get("Score_Delta_vs_Baseline", np.nan)
        edge_delta = row.get("Edge_vs_Breakeven_Delta_vs_Baseline", np.nan)
        return_delta = row.get("Avg_Return_Delta_vs_Baseline", np.nan)
        expectancy_delta = row.get("Expectancy_Per_Day_Delta_vs_Baseline", np.nan)
        timeout_delta = row.get("Timeout_Rate_Delta_vs_Baseline", np.nan)
        trades_delta = row.get("Trades_Delta_vs_Baseline", np.nan)

        decision = "RESEARCH_ONLY"
        reason = "Candidate has mixed or weak evidence."

        if (
            pd.notna(edge_delta)
            and pd.notna(return_delta)
            and pd.notna(expectancy_delta)
            and edge_delta > 0
            and return_delta > 0
            and expectancy_delta > 0
            and (pd.isna(timeout_delta) or timeout_delta <= 0.10)
            and (pd.isna(trades_delta) or trades_delta >= -0.30 * max(row.get("Trades", 0), 1))
        ):
            decision = "ACCEPT_CANDIDATE"
            reason = "Candidate improves edge, return, and expectancy without a major timeout penalty."

        if (
            pd.notna(score_delta)
            and pd.notna(edge_delta)
            and pd.notna(return_delta)
            and score_delta < 0
            and edge_delta <= 0
            and return_delta <= 0
        ):
            decision = "DROP_CANDIDATE"
            reason = "Candidate weakens score, edge, and return versus baseline."

        if row.get("Research_Status") == "TRADEABLE_EDGE" and edge_delta > 0:
            decision = "TRADEABLE_UPGRADE"
            reason = "Candidate improves baseline and reaches tradeable-edge status."

        records.append(
            {
                "Ticker": row["Ticker"],
                "Profile": row["Profile"],
                "Experiment_Mode": row["Experiment_Mode"],
                "Feature_Set": row["Feature_Set"],
                "Decision": decision,
                "Reason": reason,
                "Research_Status": row.get("Research_Status"),
                "Score_Delta_vs_Baseline": score_delta,
                "Edge_vs_Breakeven_Delta_vs_Baseline": edge_delta,
                "Avg_Return_Delta_vs_Baseline": return_delta,
                "Expectancy_Per_Day_Delta_vs_Baseline": expectancy_delta,
                "Timeout_Rate_Delta_vs_Baseline": timeout_delta,
                "Trades_Delta_vs_Baseline": trades_delta,
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# RUNNER
# =============================================================================

def run_feature_experiments(args: argparse.Namespace) -> None:
    df_vnindex = radar.load_data(VNINDEX_FILE, is_vnindex=True)

    if df_vnindex is None:
        raise ValueError("VNINDEX data could not be loaded.")

    if args.all:
        tickers = [ticker.upper() for ticker in STOCK_TICKERS]
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        raise ValueError("Please provide either --ticker TICKER or --all.")

    if not tickers:
        raise ValueError("No tickers were provided or found in config.py.")

    if args.profile:
        profile_names = [args.profile.upper()]
    else:
        profile_names = list(STRATEGY_PROFILES.keys())

    feature_sets = get_feature_sets()

    if args.feature_set != "all":
        if args.feature_set not in feature_sets:
            raise ValueError(
                f"Unknown feature set: {args.feature_set}. "
                f"Available sets: {list(feature_sets.keys())}"
            )
        feature_sets = {args.feature_set: feature_sets[args.feature_set]}

    all_results = []
    all_health_reports = []

    for ticker in tickers:
        logger.info("=" * 100)
        logger.info("Feature experiment started | ticker=%s", ticker)
        logger.info("=" * 100)

        df_features = build_feature_dataframe(ticker, df_vnindex)

        for feature_set_name, feature_cols in feature_sets.items():
            health_report = create_feature_health_report(
                ticker=ticker,
                feature_set_name=feature_set_name,
                df_features=df_features,
                feature_cols=feature_cols,
            )

            all_health_reports.append(health_report)

        for profile_name in profile_names:
            if profile_name not in STRATEGY_PROFILES:
                logger.warning(
                    "Unknown profile skipped | ticker=%s | profile=%s",
                    ticker,
                    profile_name,
                )
                continue

            profile_config = STRATEGY_PROFILES[profile_name]

            setup = resolve_fixed_setup(
                ticker=ticker,
                profile_name=profile_name,
                args=args,
            )

            for feature_set_name, feature_cols in feature_sets.items():
                logger.info(
                    "Evaluating feature set | ticker=%s | profile=%s | feature_set=%s",
                    ticker,
                    profile_name,
                    feature_set_name,
                )

                if args.mode in ["fixed", "both"]:
                    if setup is None:
                        logger.warning(
                            "Fixed setup skipped because no setup was found | ticker=%s | profile=%s",
                            ticker,
                            profile_name,
                        )
                    else:
                        fixed_result = evaluate_fixed_setup(
                            ticker=ticker,
                            profile_name=profile_name,
                            profile_config=profile_config,
                            feature_set_name=feature_set_name,
                            df_features=df_features,
                            feature_cols=feature_cols,
                            lookahead=setup["lookahead"],
                            tp=setup["tp"],
                            sl=setup["sl"],
                            threshold=setup["threshold"],
                            round_trip_cost=args.round_trip_cost,
                            min_edge_vs_breakeven=args.min_edge_vs_breakeven,
                        )

                        fixed_result["Setup_Source"] = setup["setup_source"]
                        all_results.append(fixed_result)

                if args.mode in ["reoptimized", "both"]:
                    reoptimized_result = evaluate_reoptimized_setup(
                        ticker=ticker,
                        profile_name=profile_name,
                        profile_config=profile_config,
                        feature_set_name=feature_set_name,
                        df_features=df_features,
                        feature_cols=feature_cols,
                        round_trip_cost=args.round_trip_cost,
                        min_edge_vs_breakeven=args.min_edge_vs_breakeven,
                    )

                    reoptimized_result["Setup_Source"] = "profile_grid_search"
                    all_results.append(reoptimized_result)

    results_df = pd.DataFrame(all_results)
    results_df = add_delta_vs_baseline(results_df)

    decision_df = create_feature_set_decision(results_df)

    health_df = (
        pd.concat(all_health_reports, ignore_index=True)
        if all_health_reports
        else pd.DataFrame()
    )

    comparison_path = FEATURE_EXPERIMENT_DIR / "feature_set_comparison.csv"
    health_path = FEATURE_EXPERIMENT_DIR / "feature_health_report.csv"
    decision_path = FEATURE_EXPERIMENT_DIR / "feature_set_decision_report.csv"

    results_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    health_df.to_csv(
        health_path,
        index=False,
        encoding="utf-8-sig",
    )

    decision_df.to_csv(
        decision_path,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("Feature set comparison saved | path=%s", comparison_path)
    logger.info("Feature health report saved | path=%s", health_path)
    logger.info("Feature decision report saved | path=%s", decision_path)

    if not results_df.empty:
        display_cols = [
            "Ticker",
            "Profile",
            "Experiment_Mode",
            "Feature_Set",
            "LOOKAHEAD",
            "TP",
            "SL",
            "THRESH",
            "Score",
            "Validation_Precision",
            "Base_Rate",
            "Breakeven_Precision",
            "Edge_vs_Breakeven",
            "Avg_Return",
            "Avg_Holding_Days",
            "Expectancy_Per_Day",
            "Timeout_Rate",
            "Trades",
            "Research_Status",
            "Pass_Tradeable_Rule",
            "Score_Delta_vs_Baseline",
            "Edge_vs_Breakeven_Delta_vs_Baseline",
            "Avg_Return_Delta_vs_Baseline",
        ]

        display_cols = [
            col for col in display_cols
            if col in results_df.columns
        ]

        print("\nFeature Set Comparison")
        print(results_df[display_cols].to_string(index=False))

    if not decision_df.empty:
        print("\nFeature Set Decision Report")
        print(decision_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feature set experiments for the AI Entry Signal Radar."
    )

    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker to test. Example: CTD",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run experiments for all tickers defined in config.py.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile to test. Example: SWING. If omitted, all profiles are tested.",
    )

    parser.add_argument(
        "--feature-set",
        type=str,
        default="all",
        help="Feature set to test. Use 'all' to test all candidate sets.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["fixed", "reoptimized", "both"],
        default="both",
        help="Experiment mode.",
    )

    parser.add_argument(
        "--lookahead",
        type=int,
        default=None,
        help="Explicit fixed setup lookahead. Must be used with --tp, --sl, and --threshold.",
    )

    parser.add_argument(
        "--tp",
        type=float,
        default=None,
        help="Explicit fixed setup take-profit threshold.",
    )

    parser.add_argument(
        "--sl",
        type=float,
        default=None,
        help="Explicit fixed setup stop-loss threshold.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Explicit fixed setup probability threshold.",
    )

    parser.add_argument(
        "--round-trip-cost",
        type=float,
        default=0.0,
        help="Round-trip transaction cost used in breakeven precision calculation.",
    )

    parser.add_argument(
        "--min-edge-vs-breakeven",
        type=float,
        default=0.0,
        help="Minimum edge over breakeven precision required for tradeable-edge status.",
    )

    parser.add_argument(
        "--no-diagnostics-fallback",
        action="store_true",
        help="Disable fallback to outputs/setup_diagnostics when best_params.csv has no setup.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_feature_experiments(parse_args())
