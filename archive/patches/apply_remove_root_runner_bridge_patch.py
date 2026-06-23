from __future__ import annotations

import py_compile
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

FILES = {
    "src/ml_signal/pipelines/feature_builder.py": '\nfrom __future__ import annotations\n\nimport logging\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n\nimport config as cfg\nfrom ml_signal.core_compat import calculate_features, load_data\nfrom ml_signal.features.candidate import add_candidate_features\nfrom ml_signal.features.registry import get_feature_sets as registry_get_feature_sets\nfrom ml_signal.features.valuation import (\n    get_extended_valuation_columns,\n    get_valuation_feature_columns,\n)\nfrom ml_signal.labels.tp_sl import calculate_targets\n\n\nlogger = logging.getLogger(__name__)\n\nPROJECT_ROOT = getattr(cfg, "PROJECT_ROOT", Path(__file__).resolve().parents[3])\nPROCESSED_DATA_DIR = getattr(\n    cfg,\n    "PROCESSED_DATA_DIR",\n    PROJECT_ROOT / "data" / "processed_data",\n)\nVNINDEX_FILE = getattr(\n    cfg,\n    "VNINDEX_FILE",\n    PROCESSED_DATA_DIR / "VNINDEX_standardized.csv",\n)\nSTOCK_FILES = getattr(cfg, "STOCK_FILES", [])\nVALUATION_DATA_DIR = PROCESSED_DATA_DIR / "valuation"\n\n\ndef get_stock_file_map() -> dict[str, Path]:\n    file_map: dict[str, Path] = {}\n\n    for file_path in STOCK_FILES:\n        path = Path(file_path)\n        ticker = path.stem.replace("_standardized", "").upper()\n        file_map[ticker] = path\n\n    return file_map\n\n\ndef resolve_stock_file(ticker: str) -> Path:\n    ticker = ticker.upper()\n    file_map = get_stock_file_map()\n\n    if ticker in file_map:\n        return file_map[ticker]\n\n    fallback_path = PROCESSED_DATA_DIR / f"{ticker}_standardized.csv"\n\n    if fallback_path.exists():\n        return fallback_path\n\n    raise FileNotFoundError(\n        f"Cannot find standardized data file for ticker: {ticker}"\n    )\n\n\ndef load_vnindex_data() -> pd.DataFrame:\n    df_vnindex = load_data(VNINDEX_FILE, is_vnindex=True)\n\n    if df_vnindex is None:\n        raise ValueError(f"VNINDEX data could not be loaded: {VNINDEX_FILE}")\n\n    return df_vnindex\n\n\ndef load_valuation_features(ticker: str) -> pd.DataFrame | None:\n    ticker = ticker.upper()\n    valuation_path = VALUATION_DATA_DIR / f"{ticker}_valuation_standardized.csv"\n\n    if not valuation_path.exists():\n        logger.warning(\n            "Valuation file not found | ticker=%s | path=%s",\n            ticker,\n            valuation_path,\n        )\n        return None\n\n    df_valuation = pd.read_csv(valuation_path, encoding="utf-8-sig")\n    df_valuation.columns = df_valuation.columns.astype(str).str.strip()\n\n    if "Date" not in df_valuation.columns:\n        raise ValueError(f"Valuation file must contain Date column: {valuation_path}")\n\n    df_valuation["Date"] = pd.to_datetime(df_valuation["Date"], errors="coerce")\n    df_valuation = df_valuation.dropna(subset=["Date"])\n    df_valuation = df_valuation.sort_values("Date")\n    df_valuation = df_valuation.drop_duplicates(subset=["Date"], keep="last")\n\n    candidate_cols = get_valuation_feature_columns() + get_extended_valuation_columns()\n    available_cols = [col for col in candidate_cols if col in df_valuation.columns]\n\n    for col in available_cols:\n        df_valuation[col] = pd.to_numeric(df_valuation[col], errors="coerce")\n\n    df_valuation = df_valuation[["Date"] + available_cols].copy()\n    df_valuation = df_valuation.set_index("Date").sort_index()\n\n    logger.info(\n        "Loaded valuation features | ticker=%s | rows=%d | columns=%s",\n        ticker,\n        len(df_valuation),\n        available_cols,\n    )\n\n    return df_valuation\n\n\ndef merge_valuation_features(df_features: pd.DataFrame, ticker: str) -> pd.DataFrame:\n    # Valuation data is expected to be point-in-time standardized.\n    # Exact-date join avoids forward-filling PB/market cap across missing dates.\n    df_features = df_features.copy().sort_index()\n    valuation_df = load_valuation_features(ticker)\n\n    required_cols = get_valuation_feature_columns()\n\n    if valuation_df is None:\n        for col in required_cols:\n            if col not in df_features.columns:\n                df_features[col] = np.nan\n        return df_features\n\n    df_features = df_features.join(valuation_df, how="left")\n\n    for col in required_cols:\n        if col not in df_features.columns:\n            df_features[col] = np.nan\n\n    return df_features\n\n\ndef build_feature_dataframe(\n    ticker: str,\n    df_vnindex: pd.DataFrame | None = None,\n) -> pd.DataFrame:\n    if df_vnindex is None:\n        df_vnindex = load_vnindex_data()\n\n    stock_file = resolve_stock_file(ticker)\n    df_stock = load_data(stock_file, is_vnindex=False)\n\n    if df_stock is None:\n        raise ValueError(f"Stock data could not be loaded: {ticker}")\n\n    df = pd.merge(\n        df_stock,\n        df_vnindex,\n        left_index=True,\n        right_index=True,\n        how="inner",\n    ).sort_index()\n\n    df_features = calculate_features(df)\n    df_features = add_candidate_features(df_features)\n    df_features = merge_valuation_features(df_features, ticker)\n\n    return df_features.replace([np.inf, -np.inf], np.nan)\n\n\ndef get_feature_sets() -> dict[str, list[str]]:\n    return registry_get_feature_sets()\n\n\ndef validate_feature_columns(\n    df_features: pd.DataFrame,\n    requested_columns: list[str],\n) -> list[str]:\n    missing_cols = [\n        feature for feature in requested_columns\n        if feature not in df_features.columns\n    ]\n\n    if missing_cols:\n        raise ValueError(\n            f"Feature columns are missing from dataframe: {missing_cols}"\n        )\n\n    return requested_columns\n\n\ndef get_feature_columns(\n    df_features: pd.DataFrame,\n    feature_set: str,\n) -> list[str]:\n    feature_sets = get_feature_sets()\n\n    if feature_set not in feature_sets:\n        available = ", ".join(sorted(feature_sets.keys()))\n        raise ValueError(\n            f"Unknown feature_set={feature_set}. Available feature sets: {available}"\n        )\n\n    return validate_feature_columns(df_features, feature_sets[feature_set])\n\n\n__all__ = [\n    "PROJECT_ROOT",\n    "PROCESSED_DATA_DIR",\n    "VNINDEX_FILE",\n    "STOCK_FILES",\n    "VALUATION_DATA_DIR",\n    "build_feature_dataframe",\n    "calculate_targets",\n    "get_feature_columns",\n    "get_feature_sets",\n    "get_stock_file_map",\n    "load_valuation_features",\n    "load_vnindex_data",\n    "merge_valuation_features",\n    "resolve_stock_file",\n    "validate_feature_columns",\n]\n',
    "src/ml_signal/models/factory.py": '\nfrom __future__ import annotations\n\nfrom typing import Any\n\nimport numpy as np\n\nfrom ml_signal.models.xgb import (\n    calculate_scale_pos_weight as _calculate_scale_pos_weight,\n    create_xgb_model as _create_xgb_model,\n)\n\n\ndef calculate_scale_pos_weight(y_train: np.ndarray) -> float:\n    return float(_calculate_scale_pos_weight(y_train))\n\n\ndef create_xgb_model(scale_pos_weight: float) -> Any:\n    return _create_xgb_model(scale_pos_weight)\n\n\n__all__ = [\n    "calculate_scale_pos_weight",\n    "create_xgb_model",\n]\n',
    "src/ml_signal/pipelines/__init__.py": "\n",
    "src/ml_signal/evaluation/__init__.py": "\n",
}


def backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_before_remove_root_runner_bridge")

    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[BACKUP] {backup.relative_to(ROOT)}")
    else:
        print(f"[SKIP] backup exists: {backup.relative_to(ROOT)}")


def write_file(rel_path: str, content: str) -> None:
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


def assert_bridge_removed() -> None:
    checked_files = [
        ROOT / "src/ml_signal/pipelines/feature_builder.py",
        ROOT / "src/ml_signal/models/factory.py",
    ]

    forbidden = "feature_experiment_runner"

    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        if forbidden in text:
            raise ValueError(f"Bridge still present in {path.relative_to(ROOT)}")


def main() -> None:
    print("=" * 100)
    print("Remove root runner bridge patch")
    print("=" * 100)

    for rel_path, content in FILES.items():
        write_file(rel_path, content)

    assert_bridge_removed()

    for rel_path in FILES:
        py_compile.compile(str(ROOT / rel_path), doraise=True)
        print(f"[COMPILE OK] {rel_path}")

    print("=" * 100)
    print("Done.")
    print("Validate:")
    print("python scripts/production/run_signal.py --ticker=CTD --profile=SWING --config=configs/experiments/ctd_cost_resilient_swing_v1.yaml")
    print("=" * 100)


if __name__ == "__main__":
    main()
