from __future__ import annotations

from pathlib import Path
import shutil


TARGET_FILE = Path("feature_experiment_runner.py")
BACKUP_FILE = Path("feature_experiment_runner.py.bak_before_valuation_patch")


VALUATION_DIR_BLOCK = """
VALUATION_DATA_DIR = PROCESSED_DATA_DIR / "valuation"
"""


VALUATION_FUNCTIONS = r"""
# =============================================================================
# VALUATION FEATURE LOADING
# =============================================================================

def get_valuation_feature_columns() -> list[str]:
    return [
        "PB",
        "Book_to_Market",
        "Log_PB",
        "PB_Z_756",
        "PB_Percentile_756",
    ]


def get_extended_valuation_columns() -> list[str]:
    return [
        "MarketCap_TyDong",
        "Book_Equity_Parent_TyDong",
    ]


def load_valuation_features(ticker: str) -> pd.DataFrame | None:
    ticker = ticker.upper()
    valuation_path = VALUATION_DATA_DIR / f"{ticker}_valuation_standardized.csv"

    if not valuation_path.exists():
        logger.warning(
            "Valuation file not found | ticker=%s | path=%s",
            ticker,
            valuation_path,
        )
        return None

    df_valuation = pd.read_csv(valuation_path, encoding="utf-8-sig")
    df_valuation.columns = df_valuation.columns.astype(str).str.strip()

    if "Date" not in df_valuation.columns:
        raise ValueError(f"Valuation file must contain Date column: {valuation_path}")

    df_valuation["Date"] = pd.to_datetime(df_valuation["Date"], errors="coerce")
    df_valuation = df_valuation.dropna(subset=["Date"])
    df_valuation = df_valuation.sort_values("Date")
    df_valuation = df_valuation.drop_duplicates(subset=["Date"], keep="last")

    candidate_cols = get_valuation_feature_columns() + get_extended_valuation_columns()
    available_cols = [col for col in candidate_cols if col in df_valuation.columns]

    for col in available_cols:
        df_valuation[col] = pd.to_numeric(df_valuation[col], errors="coerce")

    df_valuation = df_valuation[["Date"] + available_cols].copy()
    df_valuation = df_valuation.set_index("Date").sort_index()

    logger.info(
        "Loaded valuation features | ticker=%s | rows=%d | columns=%s",
        ticker,
        len(df_valuation),
        available_cols,
    )

    return df_valuation


def merge_valuation_features(df_features: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # The valuation file is already point-in-time. This exact-date join avoids
    # forward-filling MarketCap/PB across missing trading days.
    df_features = df_features.copy().sort_index()
    valuation_df = load_valuation_features(ticker)

    required_cols = get_valuation_feature_columns()

    if valuation_df is None:
        for col in required_cols:
            if col not in df_features.columns:
                df_features[col] = np.nan
        return df_features

    df_features = df_features.join(valuation_df, how="left")

    for col in required_cols:
        if col not in df_features.columns:
            df_features[col] = np.nan

    return df_features
"""


FEATURE_SET_INSERT = """
    feature_sets.update(
        {
            "candidate_valuation_v1": baseline + [
                "PB",
                "Book_to_Market",
                "Log_PB",
                "PB_Z_756",
                "PB_Percentile_756",
            ],

            "candidate_valuation_plus_trend_v1": baseline + [
                "PB",
                "Book_to_Market",
                "Log_PB",
                "PB_Z_756",
                "PB_Percentile_756",
                "ER_10",
                "Volume_Ratio_20",
                "Drawdown_60",
                "Distance_52W_High",
            ],
        }
    )

"""


def apply_patch() -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(
            "feature_experiment_runner.py was not found. "
            "Run this patch from the project root."
        )

    text = TARGET_FILE.read_text(encoding="utf-8")

    if "candidate_valuation_v1" in text and "merge_valuation_features" in text:
        print("Valuation feature patch already appears to be applied.")
        return

    if not BACKUP_FILE.exists():
        shutil.copy2(TARGET_FILE, BACKUP_FILE)
        print(f"Backup created: {BACKUP_FILE}")

    if "VALUATION_DATA_DIR" not in text:
        marker = 'FEATURE_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)\n'
        if marker not in text:
            raise ValueError("Could not find FEATURE_EXPERIMENT_DIR marker.")
        text = text.replace(marker, marker + VALUATION_DIR_BLOCK + "\n", 1)

    if "def merge_valuation_features" not in text:
        marker = "# =============================================================================\n# DATA LOADING\n# ============================================================================="
        if marker not in text:
            raise ValueError("Could not find DATA LOADING section marker.")
        text = text.replace(marker, VALUATION_FUNCTIONS + "\n\n" + marker, 1)

    if "candidate_valuation_v1" not in text:
        marker = "    return {\n        name: unique_preserve_order(cols)\n        for name, cols in feature_sets.items()\n    }\n"
        if marker not in text:
            raise ValueError("Could not find feature_sets return block.")
        text = text.replace(marker, FEATURE_SET_INSERT + marker, 1)

    old_block = """    df_features = radar.calculate_features(df)
    df_features = add_candidate_features(df_features)

    return df_features
"""

    new_block = """    df_features = radar.calculate_features(df)
    df_features = add_candidate_features(df_features)
    df_features = merge_valuation_features(df_features, ticker)

    return df_features
"""

    if old_block in text:
        text = text.replace(old_block, new_block, 1)
    elif "merge_valuation_features(df_features, ticker)" not in text:
        raise ValueError(
            "Could not find build_feature_dataframe block to update. "
            "Please patch it manually."
        )

    TARGET_FILE.write_text(text, encoding="utf-8")
    print("Valuation feature patch applied successfully.")
    print("Updated file: feature_experiment_runner.py")


if __name__ == "__main__":
    apply_patch()
