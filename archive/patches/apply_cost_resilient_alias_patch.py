from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "src" / "ml_signal" / "features" / "registry.py"
CONFIG_PATH = ROOT / "configs" / "experiments" / "ctd_cost_resilient_swing_v1.yaml"

ALIAS_LINE = '            "candidate_cost_resilient_recovery_no_noise_v1": unique_preserve_order(\n                subtract_columns(cost_resilient_cols, NOISE_FILTER_COLUMNS)\n            ),\n\n'
CONFIG_TEXT = 'experiment:\n  name: ctd_cost_resilient_swing_v1\n  ticker: CTD\n  profile: SWING\n  feature_set: candidate_cost_resilient_recovery_no_noise_v1\n\nfrozen_setup:\n  lookahead: 40\n  tp: 0.10\n  sl: -0.05\n  threshold: 0.60\n\nrisk_assumptions:\n  round_trip_cost_base: 0.003\n  round_trip_cost_stress: 0.005\n  min_edge_vs_breakeven: 0.02\n\nvalidation_summary:\n  cost_0_003:\n    pass_periods: 3\n    periods: 4\n    min_edge: 0.0867\n    avg_return: 0.0348\n    min_return: 0.0265\n    total_trades: 887\n  cost_0_005:\n    pass_periods: 3\n    periods: 4\n    min_edge: 0.0733\n    avg_return: 0.0348\n    min_return: 0.0265\n    total_trades: 887\n\nstatus:\n  stage: production_candidate_cost_resilient\n  notes: >\n    Best frozen setup after cost-resilient feature experiment. Uses liquidity\n    and recovery-quality features while excluding noise-filter features, which\n    weakened the full candidate in CTD tests.\n'


def backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak_before_cost_resilient_alias")
    if path.exists() and not backup_path.exists():
        shutil.copy2(path, backup_path)
        print(f"[BACKUP] {backup_path}")


def patch_registry() -> None:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing registry file: {REGISTRY_PATH}")

    text = REGISTRY_PATH.read_text(encoding="utf-8")

    if "candidate_cost_resilient_recovery_no_noise_v1" in text:
        print("[SKIP] alias already exists in registry.py")
        return

    marker = """            "ablation_cost_resilient_recovery_full_v1": cost_resilient_cols,

"""

    if marker not in text:
        raise ValueError("Could not find cost-resilient registry insertion marker.")

    backup(REGISTRY_PATH)
    text = text.replace(marker, marker + ALIAS_LINE, 1)
    REGISTRY_PATH.write_text(text, encoding="utf-8")
    print("[PATCH] added candidate_cost_resilient_recovery_no_noise_v1 alias")


def write_config() -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CONFIG_PATH.exists():
        print(f"[SKIP] config already exists: {CONFIG_PATH}")
        return

    CONFIG_PATH.write_text(CONFIG_TEXT, encoding="utf-8")
    print(f"[WRITE] {CONFIG_PATH}")


def main() -> None:
    patch_registry()
    write_config()
    print("")
    print("Validate with:")
    print(
        "python feature_experiment_runner.py --ticker=CTD --profile=SWING "
        "--mode=fixed --feature-set=candidate_cost_resilient_recovery_no_noise_v1 "
        "--lookahead=40 --tp=0.10 --sl=-0.05 --threshold=0.60 "
        "--round-trip-cost=0.005 --min-edge-vs-breakeven=0.02 "
        "--no-diagnostics-fallback"
    )


if __name__ == "__main__":
    main()
