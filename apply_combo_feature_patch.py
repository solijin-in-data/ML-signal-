from __future__ import annotations

from pathlib import Path
import shutil


TARGET_FILE = Path("feature_experiment_runner.py")
BACKUP_FILE = Path("feature_experiment_runner.py.bak_before_combo_patch")


COMBO_FEATURE_SET_INSERT = '''
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

'''


def apply_patch() -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(
            "feature_experiment_runner.py was not found. "
            "Run this patch from the project root."
        )

    text = TARGET_FILE.read_text(encoding="utf-8")

    if "candidate_recovery_valuation_v1" in text and "candidate_trend_recovery_v1" in text:
        print("Combo feature patch already appears to be applied.")
        return

    if not BACKUP_FILE.exists():
        shutil.copy2(TARGET_FILE, BACKUP_FILE)
        print(f"Backup created: {BACKUP_FILE}")

    if "candidate_recovery_v1" not in text:
        raise ValueError("candidate_recovery_v1 was not found in feature_experiment_runner.py.")

    if "candidate_trend_quality_v1" not in text:
        raise ValueError("candidate_trend_quality_v1 was not found in feature_experiment_runner.py.")

    return_marker = '''    return {
        name: unique_preserve_order(cols)
        for name, cols in feature_sets.items()
    }
'''

    if return_marker not in text:
        raise ValueError("Could not find the feature_sets return block.")

    text = text.replace(return_marker, COMBO_FEATURE_SET_INSERT + return_marker, 1)

    TARGET_FILE.write_text(text, encoding="utf-8")

    print("Combo feature patch applied successfully.")
    print("Added feature sets:")
    print("- candidate_recovery_valuation_v1")
    print("- candidate_trend_recovery_v1")


if __name__ == "__main__":
    apply_patch()
