from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
UNPROCESSED_DATA_DIR = DATA_DIR / "unprocessed_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
VNINDEX_FILE = PROCESSED_DATA_DIR / "VNINDEX_standardized.csv"

STOCK_TICKERS = [
    "CTD",
    # "VPB",
    # "STB",
    # "MBB",
    # "TCB",
    # "SHB",
    # "CTG",
    # "CTD",
]

STOCK_FILES = [PROCESSED_DATA_DIR / f"{ticker}_standardized.csv" for ticker in STOCK_TICKERS]

# Each profile represents a different trading horizon. The optimizer selects
# the best setup inside each profile instead of forcing short and long horizons
# to compete directly.
STRATEGY_PROFILES = {
    "FAST_SWING": {
        "lookahead": [10, 20],
        "tp": [0.05, 0.08],
        "sl": [-0.04, -0.05],
        "threshold": [0.50, 0.55, 0.60, 0.65, 0.70],
        "max_avg_holding_days": 15,
        "max_timeout_rate": 0.40,
    },
    "SWING": {
        "lookahead": [20, 30, 40],
        "tp": [0.08, 0.10],
        "sl": [-0.05, -0.07],
        "threshold": [0.50, 0.55, 0.60, 0.65, 0.70],
        "max_avg_holding_days": 25,
        "max_timeout_rate": 0.45,
    },
    "POSITION": {
        "lookahead": [40, 50, 60],
        "tp": [0.10, 0.15],
        "sl": [-0.07, -0.08, -0.10],
        "threshold": [0.50, 0.55, 0.60, 0.65, 0.70],
        "max_avg_holding_days": 40,
        "max_timeout_rate": 0.50,
    },
}

CV_SPLITS = 3
TRAIN_TEST_SPLIT_RATIO = 0.80

MIN_MERGED_ROWS_REQUIRED = 100
MIN_LABELED_ROWS_REQUIRED = 100
MIN_TRADES_REQUIRED = 5

# Research gates. Start permissive so the optimizer can produce candidates,
# then tighten after reviewing setup_diagnostics/*.csv.
MIN_VALID_FOLDS_REQUIRED = 1

MIN_PRECISION_REQUIRED = 0.50
MIN_EXCESS_PRECISION_REQUIRED = 0.00
MIN_AVG_RETURN_REQUIRED = -0.02

N_ESTIMATORS = 150
MAX_DEPTH = 4
LEARNING_RATE = 0.05
RANDOM_STATE = 42

# Holding-aware scoring.
EXPECTANCY_PER_DAY_SCALE = 30.0
PRECISION_STD_PENALTY_WEIGHT = 0.30
TIMEOUT_PENALTY_WEIGHT = 0.10
HOLDING_PENALTY_WEIGHT = 0.05
TRADE_BONUS_CAP = 50
TRADE_BONUS_PER_TRADE = 0.001

STRONG_WATCH_THRESHOLD = 0.75
WEAK_SIGNAL_THRESHOLD = 0.55
MAX_HALF_KELLY_PCT = 10.0
