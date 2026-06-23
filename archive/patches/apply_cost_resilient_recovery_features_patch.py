from __future__ import annotations

import py_compile
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MODULES = {'src/ml_signal/features/liquidity.py': '\nfrom __future__ import annotations\n\nimport numpy as np\nimport pandas as pd\n\n\nLIQUIDITY_COST_COLUMNS = [\n    "Dollar_Volume",\n    "Log_Dollar_Volume",\n    "Dollar_Volume_Z_60",\n    "Volume_Dry_Up_20",\n    "Amihud_Illiquidity_20",\n    "Liquidity_Shock_20",\n]\n\n\ndef add_liquidity_cost_features(df: pd.DataFrame) -> pd.DataFrame:\n    df = df.copy().sort_index()\n\n    if "Log_Return" not in df.columns:\n        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))\n\n    df["Dollar_Volume"] = df["Close"] * df["Volume"]\n    df["Log_Dollar_Volume"] = np.log(df["Dollar_Volume"] + 1e-9)\n\n    dollar_volume_mean_60 = df["Dollar_Volume"].rolling(60).mean()\n    dollar_volume_std_60 = df["Dollar_Volume"].rolling(60).std()\n\n    df["Dollar_Volume_Z_60"] = (\n        (df["Dollar_Volume"] - dollar_volume_mean_60)\n        / (dollar_volume_std_60 + 1e-9)\n    )\n\n    df["Volume_Dry_Up_20"] = (\n        df["Volume"]\n        / (df["Volume"].rolling(20).mean() + 1e-9)\n    )\n\n    raw_amihud = df["Log_Return"].abs() / (df["Dollar_Volume"] + 1e-9)\n    df["Amihud_Illiquidity_20"] = raw_amihud.rolling(20).mean()\n\n    df["Liquidity_Shock_20"] = (\n        df["Dollar_Volume"]\n        / (df["Dollar_Volume"].rolling(20).mean() + 1e-9)\n    )\n\n    return df\n', 'src/ml_signal/features/noise.py': '\nfrom __future__ import annotations\n\nimport numpy as np\nimport pandas as pd\n\n\nNOISE_FILTER_COLUMNS = [\n    "Volatility_Z_60",\n    "Downside_Volatility_20",\n    "Trend_Efficiency_20",\n    "ER_20",\n    "ER_30",\n]\n\n\ndef add_noise_filter_features(df: pd.DataFrame) -> pd.DataFrame:\n    df = df.copy().sort_index()\n\n    if "Log_Return" not in df.columns:\n        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))\n\n    if "Volatility" not in df.columns:\n        df["Volatility"] = df["Log_Return"].rolling(20).std()\n\n    vol_mean_60 = df["Volatility"].rolling(60).mean()\n    vol_std_60 = df["Volatility"].rolling(60).std()\n\n    df["Volatility_Z_60"] = (\n        (df["Volatility"] - vol_mean_60)\n        / (vol_std_60 + 1e-9)\n    )\n\n    downside_returns = df["Log_Return"].where(df["Log_Return"] < 0, 0.0)\n    df["Downside_Volatility_20"] = downside_returns.rolling(20).std()\n\n    abs_path_20 = df["Close"].diff().abs().rolling(20).sum()\n    abs_path_30 = df["Close"].diff().abs().rolling(30).sum()\n\n    df["Trend_Efficiency_20"] = (\n        (df["Close"] - df["Close"].shift(20)).abs()\n        / (abs_path_20 + 1e-9)\n    )\n\n    df["ER_20"] = df["Trend_Efficiency_20"]\n\n    df["ER_30"] = (\n        (df["Close"] - df["Close"].shift(30)).abs()\n        / (abs_path_30 + 1e-9)\n    )\n\n    return df\n', 'src/ml_signal/features/recovery.py': '\nfrom __future__ import annotations\n\nimport numpy as np\nimport pandas as pd\n\n\nRECOVERY_COLUMNS = [\n    "Drawdown_60",\n    "Distance_52W_High",\n    "Distance_52W_Low",\n]\n\n\nRECOVERY_QUALITY_COLUMNS = [\n    "Recovery_20_From_60D_Low",\n    "Recovery_Slope_10",\n    "Days_Since_60D_Low",\n    "Reclaim_EMA55",\n    "Dist_EMA55",\n    "Higher_Low_20",\n    "Drawdown_Recovery_Ratio",\n]\n\n\ndef add_recovery_features(df: pd.DataFrame) -> pd.DataFrame:\n    df = df.copy().sort_index()\n\n    rolling_high_60 = df["Close"].rolling(60).max()\n    rolling_high_252 = df["Close"].rolling(252).max()\n    rolling_low_252 = df["Close"].rolling(252).min()\n\n    df["Drawdown_60"] = (\n        df["Close"] / (rolling_high_60 + 1e-9)\n        - 1\n    )\n\n    df["Distance_52W_High"] = (\n        df["Close"] / (rolling_high_252 + 1e-9)\n        - 1\n    )\n\n    df["Distance_52W_Low"] = (\n        df["Close"] / (rolling_low_252 + 1e-9)\n        - 1\n    )\n\n    return df\n\n\ndef _days_since_rolling_low(values: np.ndarray) -> float:\n    if len(values) == 0 or np.all(np.isnan(values)):\n        return np.nan\n\n    low_position = int(np.nanargmin(values))\n    return float(len(values) - 1 - low_position)\n\n\ndef add_recovery_quality_features(df: pd.DataFrame) -> pd.DataFrame:\n    df = df.copy().sort_index()\n\n    rolling_low_60 = df["Close"].rolling(60).min()\n    rolling_low_20 = df["Low"].rolling(20).min() if "Low" in df.columns else df["Close"].rolling(20).min()\n    prior_rolling_low_20 = rolling_low_20.shift(20)\n\n    df["Recovery_20_From_60D_Low"] = (\n        df["Close"] / (rolling_low_60 + 1e-9)\n        - 1\n    )\n\n    df["Recovery_Slope_10"] = (\n        df["Close"].pct_change(10)\n        / 10\n    )\n\n    df["Days_Since_60D_Low"] = df["Close"].rolling(60).apply(\n        _days_since_rolling_low,\n        raw=True,\n    )\n\n    ema55 = df["Close"].ewm(span=55, adjust=False).mean()\n    df["Reclaim_EMA55"] = (df["Close"] > ema55).astype(float)\n    df["Dist_EMA55"] = (df["Close"] - ema55) / (ema55 + 1e-9)\n\n    df["Higher_Low_20"] = (\n        rolling_low_20 > prior_rolling_low_20\n    ).astype(float)\n\n    drawdown_abs = df.get("Drawdown_60", pd.Series(index=df.index, dtype=float)).abs()\n    df["Drawdown_Recovery_Ratio"] = (\n        df["Recovery_20_From_60D_Low"]\n        / (drawdown_abs + 1e-9)\n    )\n\n    return df\n', 'src/ml_signal/features/candidate.py': '\nfrom __future__ import annotations\n\nimport numpy as np\nimport pandas as pd\n\nfrom ml_signal.features.momentum import add_momentum_features\nfrom ml_signal.features.trend import add_trend_quality_features\nfrom ml_signal.features.volume import add_volume_features\nfrom ml_signal.features.recovery import (\n    add_recovery_features,\n    add_recovery_quality_features,\n)\nfrom ml_signal.features.liquidity import add_liquidity_cost_features\nfrom ml_signal.features.noise import add_noise_filter_features\n\n\ndef add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:\n    # Add experimental feature families on top of baseline features.\n    # All features use current and past information only.\n    df = df.copy().sort_index()\n\n    if "Log_Return" not in df.columns:\n        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))\n\n    df = add_momentum_features(df)\n    df = add_trend_quality_features(df)\n    df = add_volume_features(df)\n    df = add_recovery_features(df)\n\n    # Cost-resilient recovery extension.\n    df = add_liquidity_cost_features(df)\n    df = add_recovery_quality_features(df)\n    df = add_noise_filter_features(df)\n\n    return df.replace([np.inf, -np.inf], np.nan)\n\n\n__all__ = [\n    "add_candidate_features",\n]\n'}

REGISTRY_IMPORTS = 'from ml_signal.features.liquidity import LIQUIDITY_COST_COLUMNS\nfrom ml_signal.features.recovery import RECOVERY_QUALITY_COLUMNS\nfrom ml_signal.features.noise import NOISE_FILTER_COLUMNS\n'

REGISTRY_INSERT = '\n    # -------------------------------------------------------------------------\n    # Cost-resilient recovery feature sets\n    # -------------------------------------------------------------------------\n    # These are designed to test whether the CTD recovery setup remains robust\n    # after round-trip cost/slippage assumptions are added.\n    base_cost_resilient_cols = feature_sets.get(\n        "ablation_trend_recovery_minus_market_regime_v1",\n        feature_sets.get("candidate_trend_recovery_v1", baseline),\n    )\n\n    cost_resilient_cols = unique_preserve_order(\n        base_cost_resilient_cols\n        + LIQUIDITY_COST_COLUMNS\n        + RECOVERY_QUALITY_COLUMNS\n        + NOISE_FILTER_COLUMNS\n    )\n\n    feature_sets.update(\n        {\n            "candidate_cost_resilient_recovery_v1": cost_resilient_cols,\n\n            "ablation_cost_resilient_recovery_full_v1": cost_resilient_cols,\n\n            "ablation_cost_resilient_recovery_minus_liquidity_v1": unique_preserve_order(\n                subtract_columns(cost_resilient_cols, LIQUIDITY_COST_COLUMNS)\n            ),\n\n            "ablation_cost_resilient_recovery_minus_recovery_quality_v1": unique_preserve_order(\n                subtract_columns(cost_resilient_cols, RECOVERY_QUALITY_COLUMNS)\n            ),\n\n            "ablation_cost_resilient_recovery_minus_noise_filter_v1": unique_preserve_order(\n                subtract_columns(cost_resilient_cols, NOISE_FILTER_COLUMNS)\n            ),\n        }\n    )\n\n'


def backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_before_cost_resilient_features")

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

    if path.exists():
        backup_file(path)

    path.write_text(normalized, encoding="utf-8")
    print(f"[WRITE] {rel_path}")


def patch_registry() -> None:
    registry_path = ROOT / "src" / "ml_signal" / "features" / "registry.py"

    if not registry_path.exists():
        raise FileNotFoundError("Missing src/ml_signal/features/registry.py")

    text = registry_path.read_text(encoding="utf-8")

    if "candidate_cost_resilient_recovery_v1" in text:
        print("[SKIP] cost-resilient feature sets already exist in registry.py")
        return

    backup_file(registry_path)

    valuation_import = "from ml_signal.features.valuation import get_valuation_feature_columns\n"

    if valuation_import in text and "LIQUIDITY_COST_COLUMNS" not in text:
        text = text.replace(
            valuation_import,
            valuation_import + REGISTRY_IMPORTS,
            1,
        )
    elif "LIQUIDITY_COST_COLUMNS" not in text:
        raise ValueError("Could not find valuation import marker in registry.py")

    return_marker = """    return {
        name: unique_preserve_order(cols)
        for name, cols in feature_sets.items()
    }
"""

    if return_marker not in text:
        raise ValueError("Could not find registry return marker.")

    text = text.replace(return_marker, REGISTRY_INSERT + return_marker, 1)

    registry_path.write_text(text, encoding="utf-8")
    print("[PATCH] src/ml_signal/features/registry.py")


def compile_modules() -> None:
    targets = [
        ROOT / path
        for path in [
            *MODULES.keys(),
            "src/ml_signal/features/registry.py",
        ]
    ]

    for target in targets:
        py_compile.compile(str(target), doraise=True)
        print(f"[COMPILE OK] {target.relative_to(ROOT)}")


def main() -> None:
    print("=" * 100)
    print("Cost-resilient recovery feature patch")
    print("=" * 100)

    for rel_path, content in MODULES.items():
        write_module(rel_path, content)

    patch_registry()
    compile_modules()

    print("")
    print("=" * 100)
    print("Done.")
    print("Validation command:")
    print(
        "python feature_experiment_runner.py --ticker CTD --profile SWING "
        "--mode fixed --feature-set candidate_cost_resilient_recovery_v1 "
        "--lookahead 30 --tp 0.08 --sl -0.05 --threshold 0.55 "
        "--min-edge-vs-breakeven 0.02 --round-trip-cost 0.003 "
        "--no-diagnostics-fallback"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
