from __future__ import annotations

from pathlib import Path
import shutil


TARGET_FILE = Path("feature_experiment_runner.py")
BACKUP_FILE = Path("feature_experiment_runner.py.bak_before_ablation_patch")


ABLATION_FEATURE_SET_INSERT = '''
    # -------------------------------------------------------------------------
    # Ablation feature sets
    # -------------------------------------------------------------------------
    # These sets identify which feature family creates the edge.

    def subtract_columns(source_cols: list[str], remove_cols: list[str]) -> list[str]:
        remove_set = set(remove_cols)
        return [col for col in source_cols if col not in remove_set]

    valuation_cols = (
        get_valuation_feature_columns()
        if "get_valuation_feature_columns" in globals()
        else [
            "PB",
            "Book_to_Market",
            "Log_PB",
            "PB_Z_756",
            "PB_Percentile_756",
        ]
    )

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

'''


def apply_patch() -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(
            "feature_experiment_runner.py was not found. "
            "Run this patch from the project root."
        )

    text = TARGET_FILE.read_text(encoding="utf-8")

    if "ablation_trend_recovery_full_v1" in text:
        print("Ablation feature patch already appears to be applied.")
        return

    if "candidate_trend_recovery_v1" not in text:
        raise ValueError(
            "candidate_trend_recovery_v1 was not found. "
            "Apply the combo feature patch before this ablation patch."
        )

    if not BACKUP_FILE.exists():
        shutil.copy2(TARGET_FILE, BACKUP_FILE)
        print(f"Backup created: {BACKUP_FILE}")

    return_marker = '''    return {
        name: unique_preserve_order(cols)
        for name, cols in feature_sets.items()
    }
'''

    if return_marker not in text:
        raise ValueError("Could not find the feature_sets return block.")

    text = text.replace(return_marker, ABLATION_FEATURE_SET_INSERT + return_marker, 1)
    TARGET_FILE.write_text(text, encoding="utf-8")

    print("Ablation feature patch applied successfully.")
    print("Added ablation feature sets:")
    print("- ablation_trend_recovery_full_v1")
    print("- ablation_trend_recovery_minus_trend_v1")
    print("- ablation_trend_recovery_minus_recovery_v1")
    print("- ablation_trend_recovery_minus_rsi_v1")
    print("- ablation_trend_recovery_minus_drawdown_recovery_v1")
    print("- ablation_trend_recovery_minus_market_regime_v1")
    print("- ablation_recovery_valuation_full_v1")
    print("- ablation_recovery_valuation_minus_recovery_v1")
    print("- ablation_recovery_valuation_minus_valuation_v1")


if __name__ == "__main__":
    apply_patch()
