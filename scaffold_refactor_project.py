from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DIRS = [
    "configs", "configs/experiments",
    "data/raw/market", "data/raw/fundamental", "data/raw/macro", "data/raw/sector",
    "data/interim",
    "data/processed/market", "data/processed/fundamental", "data/processed/valuation",
    "data/processed/macro", "data/processed/sector",
    "data/external/vendor_data",
    "src/ml_signal", "src/ml_signal/config",
    "src/ml_signal/data", "src/ml_signal/data/standardizers",
    "src/ml_signal/features", "src/ml_signal/features/sectors",
    "src/ml_signal/labels", "src/ml_signal/models",
    "src/ml_signal/evaluation", "src/ml_signal/pipelines", "src/ml_signal/reporting",
    "scripts/data", "scripts/research", "scripts/reports",
    "archive/legacy", "archive/patches", "notebooks",
]

INIT_DIRS = [
    "src/ml_signal", "src/ml_signal/config",
    "src/ml_signal/data", "src/ml_signal/data/standardizers",
    "src/ml_signal/features", "src/ml_signal/features/sectors",
    "src/ml_signal/labels", "src/ml_signal/models",
    "src/ml_signal/evaluation", "src/ml_signal/pipelines", "src/ml_signal/reporting",
]

SCRIPT_COPIES = {
    "data_standardizer.py": "scripts/data/standardize_market_data.py",
    "balance_sheet_standardizer.py": "scripts/data/standardize_balance_sheet.py",
    "valuation_builder.py": "scripts/data/build_valuation.py",
    "feature_experiment_runner.py": "scripts/research/run_feature_experiment.py",
    "walkforward_stability_runner.py": "scripts/research/run_walkforward_stability.py",
    "frozen_walkforward_runner.py": "scripts/research/run_frozen_walkforward.py",
    "frozen_parameter_grid_runner.py": "scripts/research/run_frozen_grid.py",
    "summarize_feature_experiments.py": "scripts/reports/summarize_feature_experiments.py",
    "summarize_ablation_results.py": "scripts/reports/summarize_ablation_results.py",
}

PATCH_FILES = [
    "apply_ablation_feature_patch.py",
    "apply_combo_feature_patch.py",
    "apply_valuation_feature_patch.py",
    "apply_walkforward_tabulate_fix.py",
]

CONFIG_TEMPLATES = {
    "configs/tickers.yaml": """tickers:
  CTD:
    sector: construction
    exchange: HOSE
    benchmark: VNINDEX
    market_data: data/processed_data/CTD_standardized.csv
    valuation_data: data/processed_data/valuation/CTD_valuation_standardized.csv
    active: true

  VPB:
    sector: banking
    exchange: HOSE
    benchmark: VNINDEX
    market_data: data/processed_data/VPB_standardized.csv
    valuation_data: data/processed_data/valuation/VPB_valuation_standardized.csv
    active: false
""",
    "configs/sectors.yaml": """sectors:
  construction:
    feature_modules: [base_price_volume, trend, recovery, liquidity, valuation, sectors.construction]
  banking:
    feature_modules: [base_price_volume, trend, recovery, liquidity, valuation, sectors.banking]
  steel:
    feature_modules: [base_price_volume, trend, recovery, liquidity, valuation, sectors.steel]
""",
    "configs/profiles.yaml": """profiles:
  FAST_SWING:
    lookahead: [10, 20]
    tp: [0.05, 0.08]
    sl: [-0.04, -0.05]
    threshold: [0.50, 0.55, 0.60, 0.65, 0.70]
    max_avg_holding_days: 15
    max_timeout_rate: 0.40
  SWING:
    lookahead: [20, 30, 40]
    tp: [0.08, 0.10]
    sl: [-0.05, -0.07]
    threshold: [0.50, 0.55, 0.60, 0.65, 0.70]
    max_avg_holding_days: 25
    max_timeout_rate: 0.45
  POSITION:
    lookahead: [40, 50, 60]
    tp: [0.10, 0.15]
    sl: [-0.07, -0.08, -0.10]
    threshold: [0.50, 0.55, 0.60, 0.65, 0.70]
    max_avg_holding_days: 40
    max_timeout_rate: 0.50
""",
    "configs/experiments/ctd_swing_production_candidate_v1.yaml": """experiment:
  name: ctd_swing_production_candidate_v1
  ticker: CTD
  profile: SWING
  feature_set: ablation_trend_recovery_minus_market_regime_v1

frozen_setup:
  lookahead: 30
  tp: 0.08
  sl: -0.05
  threshold: 0.55

risk_assumptions:
  min_edge_vs_breakeven: 0.02
  round_trip_cost: 0.003

status:
  stage: production_candidate_low_cost
  notes: >
    Passed no-cost frozen grid. Cost 0.3% was close to strict threshold.
    Treat as research-production candidate until cost-aware feature work is complete.
""",
}

MIGRATION_PLAN = """# MIGRATION PLAN

Phase 1 creates a safe scaffold only. It does not rewrite the core logic.

Run:

```bash
python scaffold_refactor_project.py
python scaffold_refactor_project.py --apply
```

After that, confirm the old commands still work.

Then archive patch scripts:

```bash
python scaffold_refactor_project.py --apply --archive-patches
```

Suggested validation command:

```bash
python feature_experiment_runner.py --ticker CTD --profile SWING --mode fixed --feature-set ablation_trend_recovery_minus_market_regime_v1 --lookahead 30 --tp 0.08 --sl -0.05 --threshold 0.55 --min-edge-vs-breakeven 0.02 --no-diagnostics-fallback
```
"""

def ensure_dir(path: Path, apply: bool):
    print(("[CREATE] " if apply else "[DRY] create dir: ") + str(path))
    if apply:
        path.mkdir(parents=True, exist_ok=True)

def write_file(path: Path, content: str, apply: bool):
    if path.exists():
        print(f"[SKIP] exists: {path}")
        return
    print(("[WRITE] " if apply else "[DRY] write file: ") + str(path))
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

def copy_file(src: Path, dst: Path, apply: bool):
    if not src.exists():
        print(f"[SKIP] missing source: {src}")
        return
    if dst.exists():
        print(f"[SKIP] destination exists: {dst}")
        return
    print(f"{'[COPY]' if apply else '[DRY] copy'} {src} -> {dst}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def move_file(src: Path, dst: Path, apply: bool):
    if not src.exists():
        print(f"[SKIP] missing source: {src}")
        return
    if dst.exists():
        print(f"[SKIP] destination exists: {dst}")
        return
    print(f"{'[MOVE]' if apply else '[DRY] move'} {src} -> {dst}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--archive-patches", action="store_true")
    parser.add_argument("--archive-legacy", action="store_true")
    args = parser.parse_args()

    print("=" * 100)
    print(f"ML-signal refactor scaffold | mode={'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Root: {ROOT}")
    print("=" * 100)

    for rel in DIRS:
        ensure_dir(ROOT / rel, args.apply)

    for rel in INIT_DIRS:
        write_file(ROOT / rel / "__init__.py", "", args.apply)

    for rel, content in CONFIG_TEMPLATES.items():
        write_file(ROOT / rel, content, args.apply)

    write_file(ROOT / "MIGRATION_PLAN.md", MIGRATION_PLAN, args.apply)

    print("\nCopy active scripts to clearer script entrypoints:")
    for src_rel, dst_rel in SCRIPT_COPIES.items():
        copy_file(ROOT / src_rel, ROOT / dst_rel, args.apply)

    if args.archive_patches:
        print("\nArchive patch scripts:")
        for f in PATCH_FILES:
            move_file(ROOT / f, ROOT / "archive" / "patches" / f, args.apply)

    if args.archive_legacy:
        move_file(ROOT / "main_signal.py", ROOT / "archive" / "legacy" / "main_signal.py", args.apply)

    print("\nDone.")

if __name__ == "__main__":
    main()
