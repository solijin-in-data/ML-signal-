from __future__ import annotations

import py_compile
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent


REGISTRY_OVERRIDE_BLOCK = """
# =============================================================================
# PHASE 2.2 EXTERNAL FEATURE REGISTRY
# =============================================================================
# Prefer the extracted registry module. Keep the local implementation above as a
# fallback during the transition period.
try:
    from ml_signal.features.registry import get_feature_sets as _external_get_feature_sets
    get_feature_sets = _external_get_feature_sets
except Exception as exc:
    logger.warning(
        "Could not import external feature registry. Falling back to local get_feature_sets. error=%s",
        exc,
    )
"""


MODULES: dict[str, str] = {
    "src/ml_signal/features/valuation.py": """
from __future__ import annotations


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
""",

    "src/ml_signal/features/registry.py": """
from __future__ import annotations

from ml_signal import core_compat as radar
from ml_signal.features.valuation import get_valuation_feature_columns


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    output = []

    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)

    return output


def subtract_columns(source_cols: list[str], remove_cols: list[str]) -> list[str]:
    remove_set = set(remove_cols)
    return [col for col in source_cols if col not in remove_set]


def get_feature_sets() -> dict[str, list[str]]:
    # Central feature-set registry.
    # This registry is ticker-agnostic. Sector-specific feature sets can be
    # added here later without changing the experiment runner.
    baseline = radar.get_feature_columns()

    feature_sets: dict[str, list[str]] = {
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

    valuation_cols = get_valuation_feature_columns()

    feature_sets.update(
        {
            "candidate_valuation_v1": baseline + valuation_cols,

            "candidate_valuation_plus_trend_v1": baseline + valuation_cols + [
                "ER_10",
                "Volume_Ratio_20",
                "Drawdown_60",
                "Distance_52W_High",
            ],
        }
    )

    recovery_cols = feature_sets.get("candidate_recovery_v1", baseline)
    trend_quality_cols = feature_sets.get("candidate_trend_quality_v1", baseline)

    feature_sets.update(
        {
            "candidate_recovery_valuation_v1": unique_preserve_order(
                recovery_cols + valuation_cols
            ),

            "candidate_trend_recovery_v1": unique_preserve_order(
                trend_quality_cols
                + [col for col in recovery_cols if col not in baseline]
            ),
        }
    )

    # -------------------------------------------------------------------------
    # Ablation feature sets
    # -------------------------------------------------------------------------
    trend_cols = feature_sets.get("candidate_trend_quality_v1", baseline)
    recovery_cols = feature_sets.get("candidate_recovery_v1", baseline)

    trend_extra_cols = [col for col in trend_cols if col not in baseline]
    recovery_extra_cols = [col for col in recovery_cols if col not in baseline]

    trend_recovery_cols = feature_sets.get(
        "candidate_trend_recovery_v1",
        unique_preserve_order(
            trend_cols + [col for col in recovery_cols if col not in baseline]
        ),
    )

    recovery_valuation_cols = feature_sets.get(
        "candidate_recovery_valuation_v1",
        unique_preserve_order(recovery_cols + valuation_cols),
    )

    rsi_cols = [
        col for col in trend_recovery_cols
        if "RSI" in col.upper()
    ]

    drawdown_recovery_name_cols = [
        col for col in trend_recovery_cols
        if any(
            token in col.upper()
            for token in [
                "DRAWDOWN",
                "RECOVERY",
                "52W",
                "LOW",
                "HIGH",
                "DISTANCE",
            ]
        )
    ]

    market_regime_name_cols = [
        col for col in trend_recovery_cols
        if any(
            token in col.upper()
            for token in [
                "VN_",
                "VNINDEX",
                "MARKET",
                "INDEX",
                "RELATIVE",
                "RS_VN",
            ]
        )
    ]

    feature_sets.update(
        {
            "ablation_trend_recovery_full_v1": trend_recovery_cols,

            "ablation_trend_recovery_minus_trend_v1": unique_preserve_order(
                subtract_columns(trend_recovery_cols, trend_extra_cols)
            ),

            "ablation_trend_recovery_minus_recovery_v1": unique_preserve_order(
                subtract_columns(trend_recovery_cols, recovery_extra_cols)
            ),

            "ablation_trend_recovery_minus_rsi_v1": unique_preserve_order(
                subtract_columns(trend_recovery_cols, rsi_cols)
            ),

            "ablation_trend_recovery_minus_drawdown_recovery_v1": unique_preserve_order(
                subtract_columns(trend_recovery_cols, drawdown_recovery_name_cols)
            ),

            "ablation_trend_recovery_minus_market_regime_v1": unique_preserve_order(
                subtract_columns(trend_recovery_cols, market_regime_name_cols)
            ),

            "ablation_recovery_valuation_full_v1": recovery_valuation_cols,

            "ablation_recovery_valuation_minus_recovery_v1": unique_preserve_order(
                subtract_columns(recovery_valuation_cols, recovery_extra_cols)
            ),

            "ablation_recovery_valuation_minus_valuation_v1": unique_preserve_order(
                subtract_columns(recovery_valuation_cols, valuation_cols)
            ),
        }
    )

    return {
        name: unique_preserve_order(cols)
        for name, cols in feature_sets.items()
    }


__all__ = [
    "unique_preserve_order",
    "subtract_columns",
    "get_feature_sets",
]
""",
}


def backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_before_phase2_feature_registry")

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

    if "PHASE 2.2 EXTERNAL FEATURE REGISTRY" in text:
        print(f"[SKIP] already patched: {path}")
        return

    return_marker = """    return {
        name: unique_preserve_order(cols)
        for name, cols in feature_sets.items()
    }
"""

    if return_marker not in text:
        print(f"[SKIP] feature set return marker not found: {path}")
        return

    backup_file(path)

    text = text.replace(
        return_marker,
        return_marker + REGISTRY_OVERRIDE_BLOCK,
        1,
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
    print("Phase 2.2 patch: extract feature registry")
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
