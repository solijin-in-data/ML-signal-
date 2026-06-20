from __future__ import annotations

from pathlib import Path
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)

VNINDEX_FILE = PROCESSED_DATA_DIR / "VNINDEX_standardized.csv"

STOCK_TICKERS = [
    "CTD"
]

STOCK_FILES = [
    PROCESSED_DATA_DIR / f"{ticker}_standardized.csv"
    for ticker in STOCK_TICKERS
]


# =============================================================================
# MODEL AND OPTIMIZATION CONFIGURATION
# =============================================================================

GRID_LOOKAHEAD = [10, 20, 30, 40, 50, 60]
GRID_TP = [0.05, 0.08, 0.10, 0.15]
GRID_SL = [-0.04, -0.05, -0.07, -0.08, -0.10]
GRID_THRESH = [0.60, 0.65, 0.70]

MIN_TRADES_REQUIRED = 5
CV_SPLITS = 3

N_ESTIMATORS = 150
MAX_DEPTH = 4
LEARNING_RATE = 0.05

RANDOM_STATE = 42


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_number(value):
    """
    Fallback parser for numeric values.
    Standardized files should already be clean, but this function protects
    the pipeline from unexpected raw numeric formats.
    """
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
    """
    Load standardized stock or VNINDEX data.

    Required stock columns:
    - Date, Close, Volume
    - Optional: Net_Volume_Foreign

    Required VNINDEX columns:
    - Date, VN_Close
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return None

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception as exc:
        logger.error("Failed to read file: %s | Error: %s", file_path, exc)
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
        logger.error(
            "Missing required columns in %s: %s",
            file_path.name,
            missing_cols,
        )
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


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate model features.

    Important:
    The dataframe must be sorted from oldest to newest before this function runs.
    """
    df = df.copy()
    df = df.sort_index()

    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["Momentum_14"] = df["Close"] - df["Close"].shift(14)

    df["EMA_13"] = df["Close"].ewm(span=13, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_13_21_Cross"] = (df["EMA_13"] - df["EMA_21"]) / df["EMA_21"]
    df["Dist_EMA_13"] = (df["Close"] - df["EMA_13"]) / df["EMA_13"]

    ema_10 = df["Close"].ewm(span=10, adjust=False).mean()
    ema_50 = df["Close"].ewm(span=50, adjust=False).mean()
    df["MACD_10_50"] = ema_10 - ema_50
    df["MACD_Signal_100"] = df["MACD_10_50"].ewm(span=100, adjust=False).mean()
    df["MACD_Hist_Custom"] = df["MACD_10_50"] - df["MACD_Signal_100"]

    df["BB_Mid"] = df["Close"].rolling(window=20).mean()
    df["BB_Std"] = df["Close"].rolling(window=20).std(ddof=0)
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    df["Dist_BB_Upper"] = (df["Close"] - df["BB_Upper"]) / df["BB_Upper"]
    df["Dist_BB_Lower"] = (df["Close"] - df["BB_Lower"]) / df["BB_Lower"]

    df["Volatility"] = df["Log_Return"].rolling(window=20).std()

    df["VN_Return"] = np.log(df["VN_Close"] / df["VN_Close"].shift(1))
    df["VN_Volatility"] = df["VN_Return"].rolling(window=20).std()

    df["Relative_Strength"] = df["Log_Return"] - df["VN_Return"]
    df["RS_Trend"] = df["Relative_Strength"].rolling(window=10).mean()

    df["VN_EMA20"] = df["VN_Close"].ewm(span=20, adjust=False).mean()
    df["Market_Distance_EMA"] = (df["VN_Close"] - df["VN_EMA20"]) / df["VN_EMA20"]

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


# =============================================================================
# TARGET CREATION
# =============================================================================

def calculate_targets(
    df: pd.DataFrame,
    lookahead: int,
    tp: float,
    sl: float,
) -> pd.DataFrame:
    """
    Create binary target.

    Target = 1 if price reaches TP before SL within the lookahead window.
    Target = 0 otherwise.
    """
    df = df.copy().sort_index()

    targets = []
    close_prices = df["Close"].values
    n_samples = len(close_prices)

    for i in range(n_samples):
        if i + lookahead >= n_samples:
            targets.append(np.nan)
            continue

        entry_price = close_prices[i]
        label = 0

        for j in range(1, lookahead + 1):
            future_price = close_prices[i + j]
            future_return = (future_price - entry_price) / entry_price

            if future_return <= sl:
                label = 0
                break

            if future_return >= tp:
                label = 1
                break

        targets.append(label)

    df["Target"] = targets

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


# =============================================================================
# MODEL UTILITIES
# =============================================================================

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

    if pos_cases == 0:
        return 1.0

    return neg_cases / pos_cases


def classify_signal(probability: float, threshold: float) -> str:
    if probability >= 0.75:
        return "STRONG WATCH"

    if probability >= threshold:
        return "VALID ENTRY"

    if probability >= 0.55:
        return "WEAK SIGNAL"

    return "NO TRADE"


# =============================================================================
# OPTIMIZATION AND TRAINING
# =============================================================================

def auto_optimize_ticker(
    ticker_name: str,
    df_stock: pd.DataFrame,
    df_vnindex: pd.DataFrame,
):
    df = pd.merge(
        df_stock,
        df_vnindex,
        left_index=True,
        right_index=True,
        how="inner",
    ).sort_index()

    if len(df) < 100:
        logger.warning(
            "Skipping ticker due to insufficient merged data | ticker=%s | rows=%d",
            ticker_name,
            len(df),
        )
        return None

    df_features = calculate_features(df)
    feature_cols = get_feature_columns()

    best_score = -1.0

    best_params = {
        "LOOKAHEAD": 30,
        "TP": 0.05,
        "SL": -0.05,
        "THRESH": 0.60,
        "PRECISION": 0.0,
        "TRADES": 0,
    }

    total_iterations = (
        len(GRID_LOOKAHEAD)
        * len(GRID_TP)
        * len(GRID_SL)
        * len(GRID_THRESH)
    )

    current_iteration = 0

    logger.info(
        "Starting walk-forward validation | ticker=%s | cv_splits=%d | scenarios=%d",
        ticker_name,
        CV_SPLITS,
        total_iterations,
    )

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    for lookahead in GRID_LOOKAHEAD:
        for tp in GRID_TP:
            for sl in GRID_SL:
                df_labeled = calculate_targets(df_features, lookahead, tp, sl)

                if len(df_labeled) < 100:
                    current_iteration += len(GRID_THRESH)
                    continue

                X = df_labeled[feature_cols].values
                y = df_labeled["Target"].values

                threshold_scores = {
                    threshold: {
                        "precision_sum": 0.0,
                        "trades_sum": 0,
                        "valid_folds": 0,
                    }
                    for threshold in GRID_THRESH
                }

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

                    scale_pos_weight = calculate_scale_pos_weight(y_train)

                    model = create_xgb_model(scale_pos_weight=scale_pos_weight)

                    try:
                        model.fit(X_train, y_train)
                    except Exception as exc:
                        logger.warning(
                            "Model training failed in validation fold | ticker=%s | error=%s",
                            ticker_name,
                            exc,
                        )
                        continue

                    probabilities = model.predict_proba(X_test)[:, 1]

                    for threshold in GRID_THRESH:
                        predictions = (probabilities >= threshold).astype(int)
                        num_trades = int(np.sum(predictions == 1))

                        if num_trades == 0:
                            continue

                        precision = precision_score(
                            y_test,
                            predictions,
                            zero_division=0,
                        )

                        threshold_scores[threshold]["precision_sum"] += precision
                        threshold_scores[threshold]["trades_sum"] += num_trades
                        threshold_scores[threshold]["valid_folds"] += 1

                for threshold in GRID_THRESH:
                    current_iteration += 1
                    stats = threshold_scores[threshold]

                    if stats["valid_folds"] > 0:
                        avg_precision = (
                            stats["precision_sum"] / stats["valid_folds"]
                        )

                        total_trades = stats["trades_sum"]

                        if total_trades >= MIN_TRADES_REQUIRED:
                            score = avg_precision + (total_trades * 0.001)

                            if score > best_score:
                                best_score = score
                                best_params = {
                                    "LOOKAHEAD": lookahead,
                                    "TP": tp,
                                    "SL": sl,
                                    "THRESH": threshold,
                                    "PRECISION": avg_precision,
                                    "TRADES": total_trades,
                                }

                    if current_iteration % 50 == 0 or current_iteration == total_iterations:
                        logger.info(
                            "Validation progress | ticker=%s | %d/%d | "
                            "best=%dD TP=%.0f%% SL=%.0f%% precision=%.1f%%",
                            ticker_name,
                            current_iteration,
                            total_iterations,
                            best_params["LOOKAHEAD"],
                            best_params["TP"] * 100,
                            best_params["SL"] * 100,
                            best_params["PRECISION"] * 100,
                        )

    if best_score == -1:
        logger.warning(
            "No valid strategy found | ticker=%s",
            ticker_name,
        )
        return 0.0, 0.0, None, best_params

    return train_final_model_and_report(
        ticker_name=ticker_name,
        df_features=df_features,
        feature_cols=feature_cols,
        best_params=best_params,
    )


def train_final_model_and_report(
    ticker_name: str,
    df_features: pd.DataFrame,
    feature_cols: list[str],
    best_params: dict,
):
    final_lookahead = best_params["LOOKAHEAD"]
    final_tp = best_params["TP"]
    final_sl = best_params["SL"]
    final_threshold = best_params["THRESH"]

    df_labeled = calculate_targets(
        df_features,
        final_lookahead,
        final_tp,
        final_sl,
    )

    if len(df_labeled) < 100:
        logger.warning(
            "Insufficient labeled data for final training | ticker=%s | rows=%d",
            ticker_name,
            len(df_labeled),
        )
        return 0.0, 0.0, None, best_params

    X = df_labeled[feature_cols].values
    y = df_labeled["Target"].values

    train_size = int(len(df_labeled) * 0.8)

    X_train_raw = X[:train_size]
    X_test_raw = X[train_size:]

    y_train = y[:train_size]
    y_test = y[train_size:]

    close_test = df_labeled["Close"].values[train_size:]
    test_index = df_labeled.index[train_size:]

    if len(np.unique(y_train)) < 2:
        logger.warning(
            "Final training set has only one class | ticker=%s",
            ticker_name,
        )
        return 0.0, 0.0, None, best_params

    eval_scaler = StandardScaler()
    X_train = eval_scaler.fit_transform(X_train_raw)
    X_test = eval_scaler.transform(X_test_raw)

    eval_scale_pos_weight = calculate_scale_pos_weight(y_train)
    eval_model = create_xgb_model(scale_pos_weight=eval_scale_pos_weight)
    eval_model.fit(X_train, y_train)

    test_probabilities = eval_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= final_threshold).astype(int)

    chart_filename = create_evaluation_chart(
        ticker_name=ticker_name,
        df_labeled=df_labeled,
        test_index=test_index,
        close_test=close_test,
        y_test=y_test,
        predictions=test_predictions,
        model=eval_model,
        feature_cols=feature_cols,
        best_params=best_params,
    )

    # Production model:
    # Train on all labeled history, then predict the latest available feature row.
    production_scaler = StandardScaler()
    X_all = production_scaler.fit_transform(X)

    production_scale_pos_weight = calculate_scale_pos_weight(y)
    production_model = create_xgb_model(scale_pos_weight=production_scale_pos_weight)
    production_model.fit(X_all, y)

    latest_feature_row = (
        df_features[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .iloc[-1]
    )

    latest_scaled = production_scaler.transform([latest_feature_row.values])
    latest_probability = production_model.predict_proba(latest_scaled)[0, 1]

    logger.info(
        "Latest signal generated | ticker=%s | date=%s | probability=%.2f%% | threshold=%.2f%%",
        ticker_name,
        latest_feature_row.name.date(),
        latest_probability * 100,
        final_threshold * 100,
    )

    return (
        latest_probability,
        best_params.get("PRECISION", 0.0),
        chart_filename,
        best_params,
    )


# =============================================================================
# REPORTING
# =============================================================================

def create_evaluation_chart(
    ticker_name: str,
    df_labeled: pd.DataFrame,
    test_index: pd.Index,
    close_test: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
    model: xgb.XGBClassifier,
    feature_cols: list[str],
    best_params: dict,
) -> str:
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 12),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax1.plot(
        test_index,
        close_test,
        label=f"{ticker_name} Close Price",
        linewidth=1.5,
        alpha=0.8,
    )

    buy_signal_idx = np.where(predictions == 1)[0]

    win_idx = [
        idx for idx in buy_signal_idx
        if y_test[idx] == 1
    ]

    loss_idx = [
        idx for idx in buy_signal_idx
        if y_test[idx] == 0
    ]

    if win_idx:
        ax1.scatter(
            test_index[win_idx],
            close_test[win_idx],
            marker="^",
            s=120,
            label=f"Correct Buy Signals ({len(win_idx)})",
            zorder=5,
        )

    if loss_idx:
        ax1.scatter(
            test_index[loss_idx],
            close_test[loss_idx],
            marker="x",
            s=100,
            label=f"Incorrect Buy Signals ({len(loss_idx)})",
            zorder=5,
        )

    ax1.set_title(
        f"AI Signal Evaluation - {ticker_name}\n"
        f"Lookahead: {best_params['LOOKAHEAD']} days | "
        f"TP: {best_params['TP'] * 100:.0f}% | "
        f"SL: {best_params['SL'] * 100:.0f}% | "
        f"Validation Precision: {best_params.get('PRECISION', 0) * 100:.1f}%",
        fontsize=13,
        fontweight="bold",
    )

    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", alpha=0.5)

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    bars = ax2.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        alpha=0.8,
    )

    ax2.set_yticks(range(len(sorted_idx)))
    ax2.set_yticklabels([feature_cols[i] for i in sorted_idx])
    ax2.set_title(
        "Feature Importance",
        fontsize=12,
        fontweight="bold",
    )

    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        ax2.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {bar.get_width() * 100:.1f}%",
            va="center",
        )

    plt.tight_layout()

    chart_path = CHART_DIR / f"{ticker_name}_evaluation_chart.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    logger.info(
        "Evaluation chart saved | ticker=%s | path=%s",
        ticker_name,
        chart_path,
    )

    return str(chart_path)


def calculate_position_sizing(
    action: str,
    precision: float,
    tp: float,
    sl: float,
) -> str:
    """
    Half-Kelly sizing based on historical validation precision.

    Note:
    This is only an indicative sizing metric. It should not be used as an
    automatic execution rule.
    """
    if sl == 0:
        return "0.0%"

    reward_risk = tp / abs(sl)

    if reward_risk <= 0:
        return "0.0%"

    kelly_fraction = precision - ((1 - precision) / reward_risk)
    half_kelly_pct = (kelly_fraction / 2) * 100

    if action == "NO TRADE" or half_kelly_pct <= 0:
        return "0.0%"

    capped_half_kelly_pct = min(half_kelly_pct, 10.0)

    return f"{capped_half_kelly_pct:.1f}%"


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    logger.info("=" * 100)
    logger.info("AI ENTRY SIGNAL RADAR - START")
    logger.info("=" * 100)

    logger.info("Loading market benchmark data")

    df_vnindex = load_data(VNINDEX_FILE, is_vnindex=True)

    if df_vnindex is None:
        logger.error("VNINDEX data could not be loaded. Pipeline stopped.")
        return

    scan_results = []

    for file_path in STOCK_FILES:
        ticker = file_path.stem.replace("_standardized", "").upper()

        logger.info("-" * 100)
        logger.info("Processing ticker | ticker=%s | file=%s", ticker, file_path.name)

        df_stock = load_data(file_path, is_vnindex=False)

        if df_stock is None:
            logger.warning("Skipping ticker because stock data could not be loaded | ticker=%s", ticker)
            continue

        result = auto_optimize_ticker(
            ticker_name=ticker,
            df_stock=df_stock,
            df_vnindex=df_vnindex,
        )

        if result is None:
            logger.warning("No result returned | ticker=%s", ticker)
            continue

        probability, precision, chart_path, best_params = result

        if chart_path is None:
            logger.warning("No valid chart generated | ticker=%s", ticker)
            continue

        signal = classify_signal(
            probability=probability,
            threshold=best_params["THRESH"],
        )

        allocation = calculate_position_sizing(
            action=signal,
            precision=precision,
            tp=best_params["TP"],
            sl=best_params["SL"],
        )

        scan_results.append(
            {
                "Ticker": ticker,
                "Validation_Precision": precision,
                "Signal_Probability": probability,
                "Best_Setup": (
                    f"{best_params['LOOKAHEAD']}D / "
                    f"TP {best_params['TP'] * 100:.0f}% / "
                    f"SL {best_params['SL'] * 100:.0f}%"
                ),
                "Test_Trades": best_params["TRADES"],
                "Signal": signal,
                "Half_Kelly_Cap": allocation,
                "Chart": chart_path,
            }
        )

    logger.info("=" * 100)
    logger.info("FINAL SIGNAL SUMMARY")
    logger.info("=" * 100)

    if not scan_results:
        logger.warning("No valid scan results.")
        return

    results_df = pd.DataFrame(scan_results)

    results_df = results_df.sort_values(
        by="Signal_Probability",
        ascending=False,
    )

    display_df = results_df.copy()
    display_df["Validation_Precision"] = display_df["Validation_Precision"].map(
        lambda x: f"{x * 100:.1f}%"
    )
    display_df["Signal_Probability"] = display_df["Signal_Probability"].map(
        lambda x: f"{x * 100:.2f}%"
    )

    print("\n" + display_df.to_string(index=False))

    output_path = OUTPUT_DIR / "signal_summary.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info("Signal summary saved | path=%s", output_path)
    logger.info("AI ENTRY SIGNAL RADAR - END")


if __name__ == "__main__":
    main()