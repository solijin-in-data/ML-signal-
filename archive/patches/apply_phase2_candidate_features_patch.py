from __future__ import annotations

import py_compile
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent


CANDIDATE_OVERRIDE_BLOCK = """
# =============================================================================
# PHASE 2.3 EXTERNAL CANDIDATE FEATURE ENGINEERING
# =============================================================================
# Prefer the extracted candidate feature module. Keep the local implementation
# above as a fallback during the transition period.
try:
    from ml_signal.features.candidate import add_candidate_features as _external_add_candidate_features
    add_candidate_features = _external_add_candidate_features
except Exception as exc:
    logger.warning(
        "Could not import external candidate features. Falling back to local add_candidate_features. error=%s",
        exc,
    )
"""


MODULES: dict[str, str] = {
    "src/ml_signal/features/momentum.py": """
from __future__ import annotations

import numpy as np
import pandas as pd


MOMENTUM_COLUMNS = [
    "Abs_Momentum_14",
    "Log_Momentum_14",
    "EWM_Return_10",
    "Momentum_Dropoff_14",
    "Momentum_Dropoff_Z_14",
]


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

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

    return df
""",

    "src/ml_signal/features/trend.py": """
from __future__ import annotations

import pandas as pd


TREND_QUALITY_COLUMNS = [
    "ER_10",
    "EMA_13_Slope",
    "EMA_21_55_Gap",
    "BB_Position",
]


def add_trend_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

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

    return df
""",

    "src/ml_signal/features/volume.py": """
from __future__ import annotations

import numpy as np
import pandas as pd


VOLUME_COLUMNS = [
    "Volume_Ratio_20",
    "Log_Volume_Change",
    "Foreign_Net_5D_Ratio",
]


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

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

    return df
""",

    "src/ml_signal/features/recovery.py": """
from __future__ import annotations

import pandas as pd


RECOVERY_COLUMNS = [
    "Drawdown_60",
    "Distance_52W_High",
    "Distance_52W_Low",
]


def add_recovery_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

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

    return df
""",

    "src/ml_signal/features/candidate.py": """
from __future__ import annotations

import numpy as np
import pandas as pd

from ml_signal.features.momentum import add_momentum_features
from ml_signal.features.trend import add_trend_quality_features
from ml_signal.features.volume import add_volume_features
from ml_signal.features.recovery import add_recovery_features


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Add experimental feature families on top of baseline features.

    All features use current and past information only.
    \"\"\"
    df = df.copy().sort_index()

    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df = add_momentum_features(df)
    df = add_trend_quality_features(df)
    df = add_volume_features(df)
    df = add_recovery_features(df)

    return df.replace([np.inf, -np.inf], np.nan)


__all__ = [
    "add_candidate_features",
]
""",
}


def backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_before_phase2_candidate_features")

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

    if "PHASE 2.3 EXTERNAL CANDIDATE FEATURE ENGINEERING" in text:
        print(f"[SKIP] already patched: {path}")
        return

    return_marker = "    return df.replace([np.inf, -np.inf], np.nan)\n"

    marker_index = text.find(return_marker)

    if marker_index == -1:
        print(f"[SKIP] candidate feature return marker not found: {path}")
        return

    insertion_index = marker_index + len(return_marker)

    backup_file(path)

    text = (
        text[:insertion_index]
        + "\n\n"
        + CANDIDATE_OVERRIDE_BLOCK
        + text[insertion_index:]
    )

    path.write_text(text, encoding="utf-8")
    print(f"[PATCH] {path}")


def compile_modules() -> None:
    for rel_path in MODULES:
        path = ROOT / rel_path
        py_compile.compile(str(path), doraise=True)
        print(f"[COMPILE OK] {rel_path}")


def main() -> None:
    print("=" * 100)
    print("Phase 2.3 patch: extract candidate feature engineering")
    print("=" * 100)

    for rel_path, content in MODULES.items():
        write_module(rel_path, content)

    patch_runner(ROOT / "feature_experiment_runner.py")
    patch_runner(ROOT / "scripts" / "research" / "run_feature_experiment.py")

    compile_modules()

    print("")
    print("=" * 100)
    print("Done.")
    print("Validate with:")
    print(
        "python feature_experiment_runner.py --ticker CTD --profile SWING "
        "--mode fixed --feature-set ablation_trend_recovery_minus_market_regime_v1 "
        "--lookahead 30 --tp 0.08 --sl -0.05 --threshold 0.55 "
        "--min-edge-vs-breakeven 0.02 --no-diagnostics-fallback"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
