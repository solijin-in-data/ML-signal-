from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config import (
    CHART_DIR,
    CV_SPLITS,
    EXPECTANCY_PER_DAY_SCALE,
    HOLDING_PENALTY_WEIGHT,
    LEARNING_RATE,
    MAX_DEPTH,
    MAX_HALF_KELLY_PCT,
    MIN_AVG_RETURN_REQUIRED,
    MIN_EXCESS_PRECISION_REQUIRED,
    MIN_LABELED_ROWS_REQUIRED,
    MIN_MERGED_ROWS_REQUIRED,
    MIN_PRECISION_REQUIRED,
    MIN_TRADES_REQUIRED,
    MIN_VALID_FOLDS_REQUIRED,
    N_ESTIMATORS,
    OUTPUT_DIR,
    PRECISION_STD_PENALTY_WEIGHT,
    RANDOM_STATE,
    STOCK_FILES,
    STRATEGY_PROFILES,
    STRONG_WATCH_THRESHOLD,
    TIMEOUT_PENALTY_WEIGHT,
    TRADE_BONUS_CAP,
    TRADE_BONUS_PER_TRADE,
    TRAIN_TEST_SPLIT_RATIO,
    VNINDEX_FILE,
    WEAK_SIGNAL_THRESHOLD,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)
SETUP_DIAGNOSTIC_DIR = OUTPUT_DIR / "setup_diagnostics"
SETUP_DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_number(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text in ["", "-", "--", "nan", "NaN", "None"]:
        return np.nan
    text = text.replace(" ", "").replace(",", "")
    multiplier = 1.0
    if text.upper().endswith("K"):
        multiplier = 1_000
        text = text[:-1]
    elif text.upper().endswith("M"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.upper().endswith("B"):
        multiplier = 1_000_000_000
        text = text[:-1]
    if "%" in text:
        text = text.replace("%", "")
        multiplier = multiplier / 100
    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def load_data(file_path: Path, is_vnindex: bool = False) -> pd.DataFrame | None:
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("File not found | path=%s", file_path)
        return None
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception as exc:
        logger.error("Failed to read file | path=%s | error=%s", file_path, exc)
        return None
    df.columns = df.columns.astype(str).str.strip()
    if is_vnindex:
        if "VN_Date" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"VN_Date": "Date"})
        required_cols = ["Date", "VN_Close"]
    else:
        required_cols = ["Date", "Close", "Volume"]
        if "Net_Volume_Foreign" in df.columns:
            required_cols.append("Net_Volume_Foreign")
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error("Missing required columns | file=%s | columns=%s", file_path.name, missing_cols)
        return None
    df = df[required_cols].copy()
    for col in required_cols:
        if col != "Date":
            df[col] = df[col].apply(parse_number)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date").sort_index()
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Data is not sorted in ascending time order: {file_path}")
    logger.info(
        "Loaded data | file=%s | rows=%d | range=%s to %s",
        file_path.name,
        len(df),
        df.index.min().date(),
        df.index.max().date(),
    )
    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Keep stock-specific absolute momentum for local per-ticker models.
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
        "Log_Return", "RSI_14", "Momentum_14", "Volatility", "Volume",
        "MACD_Hist_Custom", "EMA_13_21_Cross", "Dist_EMA_13",
        "Dist_BB_Upper", "Dist_BB_Lower", "VN_Return", "VN_Volatility",
        "Relative_Strength", "RS_Trend", "Market_Distance_EMA",
        "Foreign_Net_5D", "Foreign_Net_20D_Mean", "Foreign_Mutation",
    ]


def calculate_targets(df: pd.DataFrame, lookahead: int, tp: float, sl: float) -> pd.DataFrame:
    df = df.copy().sort_index()
    targets, holding_days, exit_returns, exit_reasons, timeouts = [], [], [], [], []
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


def calculate_scale_pos_weight(y_train: np.ndarray) -> float:
    pos_cases = np.sum(y_train == 1)
    neg_cases = np.sum(y_train == 0)
    return 1.0 if pos_cases == 0 else neg_cases / pos_cases


def classify_signal(probability: float, threshold: float) -> str:
    if probability >= STRONG_WATCH_THRESHOLD:
        return "STRONG WATCH"
    if probability >= threshold:
        return "VALID ENTRY"
    if probability >= WEAK_SIGNAL_THRESHOLD:
        return "WEAK SIGNAL"
    return "NO TRADE"


def calculate_position_sizing(signal: str, precision: float, tp: float, sl: float) -> str:
    if signal not in ["VALID ENTRY", "STRONG WATCH"] or sl == 0:
        return "0.0%"
    reward_risk = tp / abs(sl)
    if reward_risk <= 0:
        return "0.0%"
    kelly_fraction = precision - ((1 - precision) / reward_risk)
    half_kelly_pct = (kelly_fraction / 2) * 100
    if half_kelly_pct <= 0:
        return "0.0%"
    return f"{min(half_kelly_pct, MAX_HALF_KELLY_PCT):.1f}%"


def create_empty_threshold_stats() -> dict:
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


def update_threshold_stats(stats: dict, df_test: pd.DataFrame, y_test: np.ndarray, probabilities: np.ndarray, threshold: float) -> None:
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


def finalize_threshold_stats(stats: dict, profile_name: str, profile_config: dict) -> dict | None:
    total_trades = stats["total_trades"]
    if total_trades == 0:
        return None

    avg_precision = stats["total_correct"] / total_trades
    base_rate = stats["test_positive"] / max(stats["test_count"], 1)
    excess_precision = avg_precision - base_rate
    precision_lift = avg_precision / (base_rate + 1e-9)
    avg_return = stats["return_sum"] / total_trades
    avg_holding_days = stats["holding_sum"] / total_trades
    expectancy_per_day = avg_return / (avg_holding_days + 1e-9)
    timeout_rate = stats["timeout_sum"] / total_trades
    precision_std = float(np.std(stats["precision_values"], ddof=0)) if len(stats["precision_values"]) > 1 else 0.0

    max_avg_holding_days = profile_config["max_avg_holding_days"]
    trade_bonus = min(total_trades, TRADE_BONUS_CAP) * TRADE_BONUS_PER_TRADE
    holding_penalty = (avg_holding_days / max_avg_holding_days) * HOLDING_PENALTY_WEIGHT

    score = (
        excess_precision
        + expectancy_per_day * EXPECTANCY_PER_DAY_SCALE
        + trade_bonus
        - precision_std * PRECISION_STD_PENALTY_WEIGHT
        - timeout_rate * TIMEOUT_PENALTY_WEIGHT
        - holding_penalty
    )

    reject_reasons = []
    if stats["valid_folds"] < MIN_VALID_FOLDS_REQUIRED:
        reject_reasons.append("insufficient_valid_folds")
    if total_trades < MIN_TRADES_REQUIRED:
        reject_reasons.append("insufficient_trades")
    if avg_precision < MIN_PRECISION_REQUIRED:
        reject_reasons.append("low_precision")
    if excess_precision < MIN_EXCESS_PRECISION_REQUIRED:
        reject_reasons.append("low_excess_precision")
    if avg_return <= MIN_AVG_RETURN_REQUIRED:
        reject_reasons.append("non_positive_avg_return")
    if avg_holding_days > profile_config["max_avg_holding_days"]:
        reject_reasons.append("avg_holding_too_long")
    if timeout_rate > profile_config["max_timeout_rate"]:
        reject_reasons.append("timeout_rate_too_high")

    return {
        "Profile": profile_name,
        "Valid": len(reject_reasons) == 0,
        "Reject_Reasons": ",".join(reject_reasons),
        "Score": score,
        "Validation_Precision": avg_precision,
        "Precision_Std": precision_std,
        "Base_Rate": base_rate,
        "Excess_Precision": excess_precision,
        "Precision_Lift": precision_lift,
        "Trades": total_trades,
        "Valid_Folds": stats["valid_folds"],
        "Avg_Return": avg_return,
        "Avg_Holding_Days": avg_holding_days,
        "Expectancy_Per_Day": expectancy_per_day,
        "Timeout_Rate": timeout_rate,
    }


def evaluate_setup_with_walk_forward(ticker_name: str, profile_name: str, profile_config: dict, df_labeled: pd.DataFrame, feature_cols: list[str]) -> dict:
    X = df_labeled[feature_cols].values
    y = df_labeled["Target"].values
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    threshold_stats = {threshold: create_empty_threshold_stats() for threshold in profile_config["threshold"]}

    for train_idx, test_idx in tscv.split(X):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        model = create_xgb_model(calculate_scale_pos_weight(y_train))
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            logger.warning("Model training failed | ticker=%s | profile=%s | error=%s", ticker_name, profile_name, exc)
            continue
        probabilities = model.predict_proba(X_test)[:, 1]
        df_test = df_labeled.iloc[test_idx].copy()
        for threshold in profile_config["threshold"]:
            update_threshold_stats(threshold_stats[threshold], df_test, y_test, probabilities, threshold)

    finalized_results = {}
    for threshold, stats in threshold_stats.items():
        result = finalize_threshold_stats(stats, profile_name, profile_config)
        if result is not None:
            finalized_results[threshold] = result
    return finalized_results


def save_setup_diagnostics(ticker_name: str, profile_name: str, candidate_records: list[dict]) -> None:
    if not candidate_records:
        logger.warning(
            "No setup diagnostics to save | ticker=%s | profile=%s",
            ticker_name,
            profile_name,
        )
        return

    diagnostics_df = pd.DataFrame(candidate_records)
    diagnostics_df = diagnostics_df.sort_values(
        by=["Valid", "Score"],
        ascending=[False, False],
    )

    output_path = SETUP_DIAGNOSTIC_DIR / f"{ticker_name}_{profile_name}_setup_diagnostics.csv"
    diagnostics_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info(
        "Setup diagnostics saved | ticker=%s | profile=%s | path=%s",
        ticker_name,
        profile_name,
        output_path,
    )


def log_best_rejected_candidate(ticker_name: str, profile_name: str, candidate_records: list[dict]) -> None:
    if not candidate_records:
        logger.warning(
            "No valid setup found and no candidate was evaluated | ticker=%s | profile=%s",
            ticker_name,
            profile_name,
        )
        return

    candidate_df = pd.DataFrame(candidate_records)
    candidate_df = candidate_df.sort_values(by="Score", ascending=False)
    best_rejected = candidate_df.iloc[0]

    logger.warning(
        "No valid setup found | ticker=%s | profile=%s | best_rejected=%dD TP=%.0f%% SL=%.0f%% threshold=%.0f%% score=%.4f | reasons=%s",
        ticker_name,
        profile_name,
        int(best_rejected["LOOKAHEAD"]),
        best_rejected["TP"] * 100,
        best_rejected["SL"] * 100,
        best_rejected["THRESH"] * 100,
        best_rejected["Score"],
        best_rejected["Reject_Reasons"],
    )


def optimize_profile(ticker_name: str, profile_name: str, profile_config: dict, df_features: pd.DataFrame, feature_cols: list[str]) -> dict | None:
    best_result = None
    candidate_records = []
    scenario_count = len(profile_config["lookahead"]) * len(profile_config["tp"]) * len(profile_config["sl"])
    current_scenario = 0
    logger.info("Profile optimization started | ticker=%s | profile=%s | scenarios=%d", ticker_name, profile_name, scenario_count)

    for lookahead in profile_config["lookahead"]:
        for tp in profile_config["tp"]:
            for sl in profile_config["sl"]:
                current_scenario += 1
                df_labeled = calculate_targets(df_features, lookahead, tp, sl)

                if len(df_labeled) < MIN_LABELED_ROWS_REQUIRED:
                    candidate_records.append({
                        "Ticker": ticker_name,
                        "Profile": profile_name,
                        "LOOKAHEAD": lookahead,
                        "TP": tp,
                        "SL": sl,
                        "THRESH": np.nan,
                        "Valid": False,
                        "Reject_Reasons": "insufficient_labeled_rows",
                        "Score": np.nan,
                        "Validation_Precision": np.nan,
                        "Base_Rate": np.nan,
                        "Excess_Precision": np.nan,
                        "Trades": 0,
                        "Valid_Folds": 0,
                        "Avg_Return": np.nan,
                        "Avg_Holding_Days": np.nan,
                        "Timeout_Rate": np.nan,
                    })
                    continue

                threshold_results = evaluate_setup_with_walk_forward(ticker_name, profile_name, profile_config, df_labeled, feature_cols)

                for threshold, metrics in threshold_results.items():
                    candidate = {
                        "Ticker": ticker_name,
                        "Profile": profile_name,
                        "LOOKAHEAD": lookahead,
                        "TP": tp,
                        "SL": sl,
                        "THRESH": threshold,
                        **metrics,
                    }
                    candidate_records.append(candidate.copy())

                    if not candidate["Valid"]:
                        continue

                    if best_result is None or candidate["Score"] > best_result["Score"]:
                        best_result = candidate

                if current_scenario % 5 == 0 or current_scenario == scenario_count:
                    if best_result is None:
                        logger.info("Profile progress | ticker=%s | profile=%s | %d/%d | best=none", ticker_name, profile_name, current_scenario, scenario_count)
                    else:
                        logger.info(
                            "Profile progress | ticker=%s | profile=%s | %d/%d | best=%dD TP=%.0f%% SL=%.0f%% threshold=%.0f%% score=%.4f",
                            ticker_name, profile_name, current_scenario, scenario_count,
                            best_result["LOOKAHEAD"], best_result["TP"] * 100, best_result["SL"] * 100,
                            best_result["THRESH"] * 100, best_result["Score"],
                        )

    save_setup_diagnostics(ticker_name, profile_name, candidate_records)

    if best_result is None:
        log_best_rejected_candidate(ticker_name, profile_name, candidate_records)
        return None

    logger.info(
        "Profile optimization completed | ticker=%s | profile=%s | lookahead=%d | tp=%.2f | sl=%.2f | threshold=%.2f | score=%.4f",
        ticker_name, profile_name, best_result["LOOKAHEAD"], best_result["TP"], best_result["SL"], best_result["THRESH"], best_result["Score"],
    )
    return best_result


def train_final_model_and_report(ticker_name: str, profile_name: str, df_features: pd.DataFrame, feature_cols: list[str], best_params: dict) -> tuple[float, str] | None:
    df_labeled = calculate_targets(df_features, best_params["LOOKAHEAD"], best_params["TP"], best_params["SL"])
    if len(df_labeled) < MIN_LABELED_ROWS_REQUIRED:
        logger.warning("Insufficient labeled data | ticker=%s | profile=%s | rows=%d", ticker_name, profile_name, len(df_labeled))
        return None

    X = df_labeled[feature_cols].values
    y = df_labeled["Target"].values
    train_size = int(len(df_labeled) * TRAIN_TEST_SPLIT_RATIO)
    X_train_raw, X_test_raw = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    if len(np.unique(y_train)) < 2:
        logger.warning("Final training set has only one class | ticker=%s | profile=%s", ticker_name, profile_name)
        return None

    eval_scaler = StandardScaler()
    X_train = eval_scaler.fit_transform(X_train_raw)
    X_test = eval_scaler.transform(X_test_raw)
    eval_model = create_xgb_model(calculate_scale_pos_weight(y_train))
    eval_model.fit(X_train, y_train)
    test_probabilities = eval_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= best_params["THRESH"]).astype(int)

    chart_path = create_evaluation_chart(
        ticker_name=ticker_name,
        profile_name=profile_name,
        df_labeled=df_labeled,
        test_index=df_labeled.index[train_size:],
        close_test=df_labeled["Close"].values[train_size:],
        y_test=y_test,
        predictions=test_predictions,
        model=eval_model,
        feature_cols=feature_cols,
        best_params=best_params,
    )

    production_scaler = StandardScaler()
    X_all = production_scaler.fit_transform(X)
    production_model = create_xgb_model(calculate_scale_pos_weight(y))
    production_model.fit(X_all, y)
    latest_feature_row = df_features[feature_cols].replace([np.inf, -np.inf], np.nan).dropna().iloc[-1]
    latest_scaled = production_scaler.transform([latest_feature_row.values])
    latest_probability = production_model.predict_proba(latest_scaled)[0, 1]

    logger.info(
        "Latest signal generated | ticker=%s | profile=%s | date=%s | probability=%.2f%% | threshold=%.2f%%",
        ticker_name, profile_name, latest_feature_row.name.date(), latest_probability * 100, best_params["THRESH"] * 100,
    )
    return latest_probability, chart_path


def create_evaluation_chart(ticker_name: str, profile_name: str, df_labeled: pd.DataFrame, test_index: pd.Index, close_test: np.ndarray, y_test: np.ndarray, predictions: np.ndarray, model: xgb.XGBClassifier, feature_cols: list[str], best_params: dict) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(test_index, close_test, label=f"{ticker_name} Close Price", linewidth=1.5, alpha=0.8)
    buy_signal_idx = np.where(predictions == 1)[0]
    win_idx = [idx for idx in buy_signal_idx if y_test[idx] == 1]
    loss_idx = [idx for idx in buy_signal_idx if y_test[idx] == 0]
    if win_idx:
        ax1.scatter(test_index[win_idx], close_test[win_idx], marker="^", s=120, label=f"Correct Buy Signals ({len(win_idx)})", zorder=5)
    if loss_idx:
        ax1.scatter(test_index[loss_idx], close_test[loss_idx], marker="x", s=100, label=f"Incorrect Buy Signals ({len(loss_idx)})", zorder=5)
    ax1.set_title(
        f"AI Signal Evaluation - {ticker_name} - {profile_name}\n"
        f"Lookahead: {best_params['LOOKAHEAD']} days | TP: {best_params['TP'] * 100:.0f}% | "
        f"SL: {best_params['SL'] * 100:.0f}% | Threshold: {best_params['THRESH'] * 100:.0f}% | Score: {best_params['Score']:.4f}",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", alpha=0.5)

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    bars = ax2.barh(range(len(sorted_idx)), importances[sorted_idx], alpha=0.8)
    ax2.set_yticks(range(len(sorted_idx)))
    ax2.set_yticklabels([feature_cols[i] for i in sorted_idx])
    ax2.set_title("Feature Importance", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)
    for bar in bars:
        ax2.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {bar.get_width() * 100:.1f}%", va="center")
    plt.tight_layout()
    chart_path = CHART_DIR / f"{ticker_name}_{profile_name}_evaluation_chart.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info("Evaluation chart saved | ticker=%s | profile=%s | path=%s", ticker_name, profile_name, chart_path)
    return str(chart_path)


def auto_optimize_ticker_profiles(ticker_name: str, df_stock: pd.DataFrame, df_vnindex: pd.DataFrame) -> list[dict]:
    df = pd.merge(df_stock, df_vnindex, left_index=True, right_index=True, how="inner").sort_index()
    if len(df) < MIN_MERGED_ROWS_REQUIRED:
        logger.warning("Skipping ticker due to insufficient merged data | ticker=%s | rows=%d", ticker_name, len(df))
        return []

    df_features = calculate_features(df)
    feature_cols = get_feature_columns()
    profile_results = []

    for profile_name, profile_config in STRATEGY_PROFILES.items():
        best_params = optimize_profile(ticker_name, profile_name, profile_config, df_features, feature_cols)
        if best_params is None:
            continue
        final_result = train_final_model_and_report(ticker_name, profile_name, df_features, feature_cols, best_params)
        if final_result is None:
            continue
        latest_probability, chart_path = final_result
        profile_results.append({
            "Ticker": ticker_name,
            "Profile": profile_name,
            "Signal_Probability": latest_probability,
            "Chart": chart_path,
            "Best_Params": best_params,
        })
    return profile_results


def main():
    logger.info("=" * 100)
    logger.info("AI ENTRY SIGNAL RADAR V2 - START")
    logger.info("=" * 100)

    df_vnindex = load_data(VNINDEX_FILE, is_vnindex=True)
    if df_vnindex is None:
        logger.error("VNINDEX data could not be loaded. Pipeline stopped.")
        return

    scan_results = []
    best_params_records = []

    for file_path in STOCK_FILES:
        ticker = Path(file_path).stem.replace("_standardized", "").upper()
        logger.info("-" * 100)
        logger.info("Processing ticker | ticker=%s | file=%s", ticker, Path(file_path).name)
        df_stock = load_data(Path(file_path), is_vnindex=False)
        if df_stock is None:
            logger.warning("Skipping ticker because stock data could not be loaded | ticker=%s", ticker)
            continue

        profile_results = auto_optimize_ticker_profiles(ticker, df_stock, df_vnindex)
        for result in profile_results:
            best_params = result["Best_Params"]
            probability = result["Signal_Probability"]
            profile = result["Profile"]
            signal = classify_signal(probability, best_params["THRESH"])
            allocation = calculate_position_sizing(signal, best_params["Validation_Precision"], best_params["TP"], best_params["SL"])
            row = {
                "Ticker": ticker,
                "Profile": profile,
                "Signal": signal,
                "Signal_Probability": probability,
                "Lookahead": best_params["LOOKAHEAD"],
                "TP": best_params["TP"],
                "SL": best_params["SL"],
                "Threshold": best_params["THRESH"],
                "Score": best_params["Score"],
                "Validation_Precision": best_params["Validation_Precision"],
                "Base_Rate": best_params["Base_Rate"],
                "Excess_Precision": best_params["Excess_Precision"],
                "Precision_Lift": best_params["Precision_Lift"],
                "Precision_Std": best_params["Precision_Std"],
                "Trades": best_params["Trades"],
                "Valid_Folds": best_params["Valid_Folds"],
                "Avg_Return": best_params["Avg_Return"],
                "Avg_Holding_Days": best_params["Avg_Holding_Days"],
                "Expectancy_Per_Day": best_params["Expectancy_Per_Day"],
                "Timeout_Rate": best_params["Timeout_Rate"],
                "Half_Kelly_Cap": allocation,
                "Chart": result["Chart"],
            }
            scan_results.append(row)
            best_params_records.append(row.copy())

    logger.info("=" * 100)
    logger.info("FINAL SIGNAL SUMMARY")
    logger.info("=" * 100)

    if not scan_results:
        logger.warning("No valid scan results.")
        return

    results_df = pd.DataFrame(scan_results)
    results_df = results_df.sort_values(by=["Ticker", "Profile", "Score"], ascending=[True, True, False])

    signal_summary_path = OUTPUT_DIR / "signal_summary.csv"
    best_params_path = OUTPUT_DIR / "best_params.csv"
    results_df.to_csv(signal_summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(best_params_records).to_csv(best_params_path, index=False, encoding="utf-8-sig")

    display_df = results_df.copy()
    pct_cols = [
        "Signal_Probability", "TP", "SL", "Threshold", "Validation_Precision",
        "Base_Rate", "Excess_Precision", "Avg_Return", "Expectancy_Per_Day", "Timeout_Rate",
    ]
    for col in pct_cols:
        display_df[col] = display_df[col].map(lambda x: f"{x * 100:.2f}%")
    display_df["Avg_Holding_Days"] = display_df["Avg_Holding_Days"].map(lambda x: f"{x:.1f}")
    display_df["Score"] = display_df["Score"].map(lambda x: f"{x:.4f}")
    display_df["Precision_Lift"] = display_df["Precision_Lift"].map(lambda x: f"{x:.2f}x")

    print("\n" + display_df.to_string(index=False))
    logger.info("Signal summary saved | path=%s", signal_summary_path)
    logger.info("Best parameters saved | path=%s", best_params_path)
    logger.info("AI ENTRY SIGNAL RADAR V2 - END")


if __name__ == "__main__":
    main()
