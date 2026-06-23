from __future__ import annotations

import shutil
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parent

CORE_IMPORT_BLOCK = '# =============================================================================\n# PHASE 2 CORE MODULE IMPORT\n# =============================================================================\n# The research runner now uses the extracted reusable core modules.\n# This keeps the command-line behavior the same while moving reusable logic\n# toward src/ml_signal/.\ntry:\n    from ml_signal import core_compat as radar\nexcept ModuleNotFoundError:\n    import sys\n\n    _HERE = Path(__file__).resolve()\n    _SRC_CANDIDATES = [\n        _HERE.parent / "src",\n        _HERE.parent.parent / "src",\n        _HERE.parent.parent.parent / "src",\n    ]\n\n    for _SRC_DIR in _SRC_CANDIDATES:\n        if (_SRC_DIR / "ml_signal").exists():\n            sys.path.insert(0, str(_SRC_DIR))\n            break\n\n    from ml_signal import core_compat as radar\n'

MODULES = {'src/ml_signal/core_compat.py': 'from __future__ import annotations\n'
                                 '\n'
                                 '"""\n'
                                 'Compatibility layer for the legacy research scripts.\n'
                                 '\n'
                                 '`feature_experiment_runner.py` can import this module as `radar` '
                                 'and keep using\n'
                                 'the old function names while the implementation gradually moves '
                                 'into src/.\n'
                                 '"""\n'
                                 '\n'
                                 'from ml_signal.data.loaders import load_data, parse_number\n'
                                 'from ml_signal.features.base_price_volume import '
                                 'calculate_features, get_feature_columns\n'
                                 'from ml_signal.labels.tp_sl import calculate_targets\n'
                                 'from ml_signal.models.xgb import calculate_scale_pos_weight, '
                                 'create_xgb_model\n'
                                 '\n'
                                 '\n'
                                 '__all__ = [\n'
                                 '    "parse_number",\n'
                                 '    "load_data",\n'
                                 '    "calculate_features",\n'
                                 '    "get_feature_columns",\n'
                                 '    "calculate_targets",\n'
                                 '    "calculate_scale_pos_weight",\n'
                                 '    "create_xgb_model",\n'
                                 ']\n',
 'src/ml_signal/data/loaders.py': 'from __future__ import annotations\n'
                                  '\n'
                                  'import logging\n'
                                  'from pathlib import Path\n'
                                  '\n'
                                  'import numpy as np\n'
                                  'import pandas as pd\n'
                                  '\n'
                                  '\n'
                                  'logger = logging.getLogger(__name__)\n'
                                  '\n'
                                  '\n'
                                  'def parse_number(value):\n'
                                  '    """\n'
                                  '    Parse numeric values from standardized or semi-standardized '
                                  'market data.\n'
                                  '    """\n'
                                  '    if pd.isna(value):\n'
                                  '        return np.nan\n'
                                  '\n'
                                  '    if isinstance(value, (int, float, np.integer, '
                                  'np.floating)):\n'
                                  '        return float(value)\n'
                                  '\n'
                                  '    text = str(value).strip()\n'
                                  '\n'
                                  '    if text in ["", "-", "--", "nan", "NaN", "None"]:\n'
                                  '        return np.nan\n'
                                  '\n'
                                  '    text = text.replace(" ", "").replace(",", "")\n'
                                  '\n'
                                  '    multiplier = 1.0\n'
                                  '\n'
                                  '    if text.upper().endswith("K"):\n'
                                  '        multiplier = 1_000\n'
                                  '        text = text[:-1]\n'
                                  '    elif text.upper().endswith("M"):\n'
                                  '        multiplier = 1_000_000\n'
                                  '        text = text[:-1]\n'
                                  '    elif text.upper().endswith("B"):\n'
                                  '        multiplier = 1_000_000_000\n'
                                  '        text = text[:-1]\n'
                                  '\n'
                                  '    if "%" in text:\n'
                                  '        text = text.replace("%", "")\n'
                                  '        multiplier = multiplier / 100\n'
                                  '\n'
                                  '    try:\n'
                                  '        return float(text) * multiplier\n'
                                  '    except ValueError:\n'
                                  '        return np.nan\n'
                                  '\n'
                                  '\n'
                                  'def load_data(file_path: Path, is_vnindex: bool = False) -> '
                                  'pd.DataFrame | None:\n'
                                  '    """\n'
                                  '    Load standardized stock or VNINDEX data.\n'
                                  '    """\n'
                                  '    file_path = Path(file_path)\n'
                                  '\n'
                                  '    if not file_path.exists():\n'
                                  '        logger.error("File not found | path=%s", file_path)\n'
                                  '        return None\n'
                                  '\n'
                                  '    try:\n'
                                  '        df = pd.read_csv(file_path, encoding="utf-8-sig")\n'
                                  '    except Exception as exc:\n'
                                  '        logger.error("Failed to read file | path=%s | '
                                  'error=%s", file_path, exc)\n'
                                  '        return None\n'
                                  '\n'
                                  '    df.columns = df.columns.astype(str).str.strip()\n'
                                  '\n'
                                  '    if is_vnindex:\n'
                                  '        if "VN_Date" in df.columns and "Date" not in '
                                  'df.columns:\n'
                                  '            df = df.rename(columns={"VN_Date": "Date"})\n'
                                  '        required_cols = ["Date", "VN_Close"]\n'
                                  '    else:\n'
                                  '        required_cols = ["Date", "Close", "Volume"]\n'
                                  '        if "Net_Volume_Foreign" in df.columns:\n'
                                  '            required_cols.append("Net_Volume_Foreign")\n'
                                  '\n'
                                  '    missing_cols = [col for col in required_cols if col not in '
                                  'df.columns]\n'
                                  '\n'
                                  '    if missing_cols:\n'
                                  '        logger.error(\n'
                                  '            "Missing required columns | file=%s | columns=%s",\n'
                                  '            file_path.name,\n'
                                  '            missing_cols,\n'
                                  '        )\n'
                                  '        return None\n'
                                  '\n'
                                  '    df = df[required_cols].copy()\n'
                                  '\n'
                                  '    for col in required_cols:\n'
                                  '        if col != "Date":\n'
                                  '            df[col] = df[col].apply(parse_number)\n'
                                  '\n'
                                  '    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")\n'
                                  '    df = df.dropna(subset=["Date"])\n'
                                  '    df = df.set_index("Date").sort_index()\n'
                                  '\n'
                                  '    if not df.index.is_monotonic_increasing:\n'
                                  '        raise ValueError(f"Data is not sorted in ascending time '
                                  'order: {file_path}")\n'
                                  '\n'
                                  '    logger.info(\n'
                                  '        "Loaded data | file=%s | rows=%d | range=%s to %s",\n'
                                  '        file_path.name,\n'
                                  '        len(df),\n'
                                  '        df.index.min().date(),\n'
                                  '        df.index.max().date(),\n'
                                  '    )\n'
                                  '\n'
                                  '    return df\n',
 'src/ml_signal/features/base_price_volume.py': 'from __future__ import annotations\n'
                                                '\n'
                                                'import numpy as np\n'
                                                'import pandas as pd\n'
                                                '\n'
                                                '\n'
                                                'def calculate_features(df: pd.DataFrame) -> '
                                                'pd.DataFrame:\n'
                                                '    """\n'
                                                '    Calculate baseline price, volume, foreign '
                                                'flow, and market-relative features.\n'
                                                '    """\n'
                                                '    df = df.copy().sort_index()\n'
                                                '\n'
                                                '    df["Log_Return"] = np.log(df["Close"] / '
                                                'df["Close"].shift(1))\n'
                                                '\n'
                                                '    delta = df["Close"].diff()\n'
                                                '    gain = delta.where(delta > 0, 0).ewm(alpha=1 '
                                                '/ 14, adjust=False).mean()\n'
                                                '    loss = (-delta.where(delta < 0, '
                                                '0)).ewm(alpha=1 / 14, adjust=False).mean()\n'
                                                '    rs = gain / (loss + 1e-9)\n'
                                                '    df["RSI_14"] = 100 - (100 / (1 + rs))\n'
                                                '\n'
                                                '    df["Momentum_14"] = df["Close"] - '
                                                'df["Close"].shift(14)\n'
                                                '\n'
                                                '    df["EMA_13"] = df["Close"].ewm(span=13, '
                                                'adjust=False).mean()\n'
                                                '    df["EMA_21"] = df["Close"].ewm(span=21, '
                                                'adjust=False).mean()\n'
                                                '    df["EMA_13_21_Cross"] = (df["EMA_13"] - '
                                                'df["EMA_21"]) / (df["EMA_21"] + 1e-9)\n'
                                                '    df["Dist_EMA_13"] = (df["Close"] - '
                                                'df["EMA_13"]) / (df["EMA_13"] + 1e-9)\n'
                                                '\n'
                                                '    ema_10 = df["Close"].ewm(span=10, '
                                                'adjust=False).mean()\n'
                                                '    ema_50 = df["Close"].ewm(span=50, '
                                                'adjust=False).mean()\n'
                                                '    df["MACD_10_50"] = ema_10 - ema_50\n'
                                                '    df["MACD_Signal_100"] = '
                                                'df["MACD_10_50"].ewm(span=100, '
                                                'adjust=False).mean()\n'
                                                '    df["MACD_Hist_Custom"] = df["MACD_10_50"] - '
                                                'df["MACD_Signal_100"]\n'
                                                '\n'
                                                '    df["BB_Mid"] = '
                                                'df["Close"].rolling(window=20).mean()\n'
                                                '    df["BB_Std"] = '
                                                'df["Close"].rolling(window=20).std(ddof=0)\n'
                                                '    df["BB_Upper"] = df["BB_Mid"] + 2 * '
                                                'df["BB_Std"]\n'
                                                '    df["BB_Lower"] = df["BB_Mid"] - 2 * '
                                                'df["BB_Std"]\n'
                                                '    df["Dist_BB_Upper"] = (df["Close"] - '
                                                'df["BB_Upper"]) / (df["BB_Upper"] + 1e-9)\n'
                                                '    df["Dist_BB_Lower"] = (df["Close"] - '
                                                'df["BB_Lower"]) / (df["BB_Lower"] + 1e-9)\n'
                                                '\n'
                                                '    df["Volatility"] = '
                                                'df["Log_Return"].rolling(window=20).std()\n'
                                                '\n'
                                                '    df["VN_Return"] = np.log(df["VN_Close"] / '
                                                'df["VN_Close"].shift(1))\n'
                                                '    df["VN_Volatility"] = '
                                                'df["VN_Return"].rolling(window=20).std()\n'
                                                '    df["Relative_Strength"] = df["Log_Return"] - '
                                                'df["VN_Return"]\n'
                                                '    df["RS_Trend"] = '
                                                'df["Relative_Strength"].rolling(window=10).mean()\n'
                                                '    df["VN_EMA20"] = df["VN_Close"].ewm(span=20, '
                                                'adjust=False).mean()\n'
                                                '    df["Market_Distance_EMA"] = (df["VN_Close"] - '
                                                'df["VN_EMA20"]) / (df["VN_EMA20"] + 1e-9)\n'
                                                '\n'
                                                '    if "Net_Volume_Foreign" in df.columns:\n'
                                                '        df["Foreign_Net_5D"] = '
                                                'df["Net_Volume_Foreign"].rolling(window=5).sum()\n'
                                                '        df["Foreign_Net_20D_Mean"] = '
                                                'df["Net_Volume_Foreign"].rolling(window=20).mean()\n'
                                                '        df["Foreign_Net_20D_Std"] = '
                                                'df["Net_Volume_Foreign"].rolling(window=20).std()\n'
                                                '        df["Foreign_Mutation"] = (\n'
                                                '            (df["Net_Volume_Foreign"] - '
                                                'df["Foreign_Net_20D_Mean"])\n'
                                                '            / (df["Foreign_Net_20D_Std"] + 1e-9)\n'
                                                '        )\n'
                                                '    else:\n'
                                                '        df["Foreign_Net_5D"] = 0.0\n'
                                                '        df["Foreign_Net_20D_Mean"] = 0.0\n'
                                                '        df["Foreign_Mutation"] = 0.0\n'
                                                '\n'
                                                '    return df\n'
                                                '\n'
                                                '\n'
                                                'def get_feature_columns() -> list[str]:\n'
                                                '    return [\n'
                                                '        "Log_Return",\n'
                                                '        "RSI_14",\n'
                                                '        "Momentum_14",\n'
                                                '        "Volatility",\n'
                                                '        "Volume",\n'
                                                '        "MACD_Hist_Custom",\n'
                                                '        "EMA_13_21_Cross",\n'
                                                '        "Dist_EMA_13",\n'
                                                '        "Dist_BB_Upper",\n'
                                                '        "Dist_BB_Lower",\n'
                                                '        "VN_Return",\n'
                                                '        "VN_Volatility",\n'
                                                '        "Relative_Strength",\n'
                                                '        "RS_Trend",\n'
                                                '        "Market_Distance_EMA",\n'
                                                '        "Foreign_Net_5D",\n'
                                                '        "Foreign_Net_20D_Mean",\n'
                                                '        "Foreign_Mutation",\n'
                                                '    ]\n',
 'src/ml_signal/labels/tp_sl.py': 'from __future__ import annotations\n'
                                  '\n'
                                  'import numpy as np\n'
                                  'import pandas as pd\n'
                                  '\n'
                                  '\n'
                                  'def calculate_targets(df: pd.DataFrame, lookahead: int, tp: '
                                  'float, sl: float) -> pd.DataFrame:\n'
                                  '    """\n'
                                  '    Create TP/SL first-hit labels.\n'
                                  '    """\n'
                                  '    df = df.copy().sort_index()\n'
                                  '\n'
                                  '    targets = []\n'
                                  '    holding_days = []\n'
                                  '    exit_returns = []\n'
                                  '    exit_reasons = []\n'
                                  '    timeouts = []\n'
                                  '\n'
                                  '    close_prices = df["Close"].values\n'
                                  '    n_samples = len(close_prices)\n'
                                  '\n'
                                  '    for i in range(n_samples):\n'
                                  '        if i + lookahead >= n_samples:\n'
                                  '            targets.append(np.nan)\n'
                                  '            holding_days.append(np.nan)\n'
                                  '            exit_returns.append(np.nan)\n'
                                  '            exit_reasons.append(np.nan)\n'
                                  '            timeouts.append(np.nan)\n'
                                  '            continue\n'
                                  '\n'
                                  '        entry_price = close_prices[i]\n'
                                  '        label = 0\n'
                                  '        hold_days = lookahead\n'
                                  '        exit_return = (close_prices[i + lookahead] - '
                                  'entry_price) / entry_price\n'
                                  '        exit_reason = "timeout"\n'
                                  '        timeout = 1\n'
                                  '\n'
                                  '        for j in range(1, lookahead + 1):\n'
                                  '            future_return = (close_prices[i + j] - entry_price) '
                                  '/ entry_price\n'
                                  '\n'
                                  '            if future_return <= sl:\n'
                                  '                label = 0\n'
                                  '                hold_days = j\n'
                                  '                exit_return = future_return\n'
                                  '                exit_reason = "sl"\n'
                                  '                timeout = 0\n'
                                  '                break\n'
                                  '\n'
                                  '            if future_return >= tp:\n'
                                  '                label = 1\n'
                                  '                hold_days = j\n'
                                  '                exit_return = future_return\n'
                                  '                exit_reason = "tp"\n'
                                  '                timeout = 0\n'
                                  '                break\n'
                                  '\n'
                                  '        targets.append(label)\n'
                                  '        holding_days.append(hold_days)\n'
                                  '        exit_returns.append(exit_return)\n'
                                  '        exit_reasons.append(exit_reason)\n'
                                  '        timeouts.append(timeout)\n'
                                  '\n'
                                  '    df["Target"] = targets\n'
                                  '    df["Holding_Days"] = holding_days\n'
                                  '    df["Exit_Return"] = exit_returns\n'
                                  '    df["Exit_Reason"] = exit_reasons\n'
                                  '    df["Timeout"] = timeouts\n'
                                  '\n'
                                  '    return df.replace([np.inf, -np.inf], np.nan).dropna()\n',
 'src/ml_signal/models/xgb.py': 'from __future__ import annotations\n'
                                '\n'
                                'import numpy as np\n'
                                'import xgboost as xgb\n'
                                '\n'
                                '\n'
                                'try:\n'
                                '    from config import (\n'
                                '        LEARNING_RATE,\n'
                                '        MAX_DEPTH,\n'
                                '        N_ESTIMATORS,\n'
                                '        RANDOM_STATE,\n'
                                '    )\n'
                                'except Exception:\n'
                                '    N_ESTIMATORS = 150\n'
                                '    MAX_DEPTH = 4\n'
                                '    LEARNING_RATE = 0.05\n'
                                '    RANDOM_STATE = 42\n'
                                '\n'
                                '\n'
                                'def calculate_scale_pos_weight(y_train: np.ndarray) -> float:\n'
                                '    pos_cases = np.sum(y_train == 1)\n'
                                '    neg_cases = np.sum(y_train == 0)\n'
                                '\n'
                                '    return 1.0 if pos_cases == 0 else neg_cases / pos_cases\n'
                                '\n'
                                '\n'
                                'def create_xgb_model(scale_pos_weight: float) -> '
                                'xgb.XGBClassifier:\n'
                                '    return xgb.XGBClassifier(\n'
                                '        n_estimators=N_ESTIMATORS,\n'
                                '        max_depth=MAX_DEPTH,\n'
                                '        learning_rate=LEARNING_RATE,\n'
                                '        scale_pos_weight=scale_pos_weight,\n'
                                '        random_state=RANDOM_STATE,\n'
                                '        eval_metric="logloss",\n'
                                '        n_jobs=-1,\n'
                                '    )\n'}


def backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_before_phase2_core")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[BACKUP] {backup}")
    else:
        print(f"[SKIP] backup exists: {backup}")


def write_module(rel_path: str, content: str) -> None:
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.strip() + "\n"

    if path.exists() and path.read_text(encoding="utf-8") == normalized:
        print(f"[SKIP] unchanged: {rel_path}")
        return

    path.write_text(normalized, encoding="utf-8")
    print(f"[WRITE] {rel_path}")


def patch_runner(path: Path) -> None:
    if not path.exists():
        print(f"[SKIP] runner not found: {path}")
        return

    text = path.read_text(encoding="utf-8")

    if "PHASE 2 CORE MODULE IMPORT" in text:
        print(f"[SKIP] already patched: {path}")
        return

    old_import = "import main_signal_radar_v2 as radar"

    if old_import not in text:
        print(f"[SKIP] import target not found: {path}")
        return

    backup_file(path)
    text = text.replace(old_import, CORE_IMPORT_BLOCK.rstrip(), 1)
    path.write_text(text, encoding="utf-8")
    print(f"[PATCH] {path}")


def compile_modules() -> None:
    targets = [ROOT / rel_path for rel_path in MODULES.keys()]
    for target in targets:
        py_compile.compile(str(target), doraise=True)
        print(f"[COMPILE OK] {target.relative_to(ROOT)}")


def main() -> None:
    print("=" * 100)
    print("Phase 2 patch: extract reusable core modules")
    print("=" * 100)

    init_files = [
        "src/ml_signal/models/__init__.py",
        "src/ml_signal/data/__init__.py",
        "src/ml_signal/features/__init__.py",
        "src/ml_signal/labels/__init__.py",
    ]

    for rel_path in init_files:
        path = ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("", encoding="utf-8")
            print(f"[WRITE] {rel_path}")

    for rel_path, content in MODULES.items():
        write_module(rel_path, content)

    patch_runner(ROOT / "feature_experiment_runner.py")
    patch_runner(ROOT / "scripts" / "research" / "run_feature_experiment.py")

    compile_modules()

    print("")
    print("=" * 100)
    print("Done.")
    print("Next validation command:")
    print(
        "python feature_experiment_runner.py --ticker CTD --profile SWING "
        "--mode fixed --feature-set ablation_trend_recovery_minus_market_regime_v1 "
        "--lookahead 30 --tp 0.08 --sl -0.05 --threshold 0.55 "
        "--min-edge-vs-breakeven 0.02 --no-diagnostics-fallback"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
