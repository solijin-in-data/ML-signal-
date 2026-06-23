from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

FILES = {
    "README.md": '\n# ML Signal Radar\n\nA research-oriented machine learning signal framework for Vietnamese equity data.\n\nThe current production-candidate workflow focuses on **CTD SWING**, using a cost-resilient recovery feature set and a frozen-grid validation process. The project separates reusable logic under `src/ml_signal/`, command-line entrypoints under `scripts/`, configuration under `configs/`, generated reports under `reports/`, and archived legacy scripts under `archive/`.\n\n> Research only. This project is not investment advice, a trading recommendation, or a promise of future performance.\n\n## Current production candidate\n\n| Item | Value |\n|---|---|\n| Ticker | CTD |\n| Profile | SWING |\n| Candidate | `production_candidate_cost_resilient_v1` |\n| Feature set | `candidate_cost_resilient_recovery_no_noise_v1` |\n| Lookahead | 40 trading days |\n| Take profit | 10% |\n| Stop loss | -5% |\n| Signal threshold | 60% |\n| Stress round-trip cost | 0.5% |\n| Production config | `configs/experiments/ctd_cost_resilient_swing_v1.yaml` |\n\nThe production-candidate setup was selected from frozen-grid evidence after testing cost-resilient recovery features. The strongest CTD SWING setup retained liquidity and recovery-quality features while excluding the noise-filter family.\n\n## Project layout\n\n```text\nML-signal-/\n├── configs/\n│   └── experiments/\n│       └── ctd_cost_resilient_swing_v1.yaml\n├── data/\n│   └── processed_data/\n├── reports/\n│   ├── production_candidates/\n│   └── signals/\n├── scripts/\n│   ├── production/\n│   │   └── run_signal.py\n│   └── research/\n├── src/\n│   └── ml_signal/\n│       ├── evaluation/\n│       ├── features/\n│       ├── labels/\n│       ├── models/\n│       ├── pipelines/\n│       └── production/\n└── archive/\n    ├── legacy/\n    │   └── root_scripts/\n    └── patches/\n```\n\n## Run the latest production-candidate signal\n\nFrom the project root:\n\n```bash\npython scripts/production/run_signal.py ^\n  --ticker=CTD ^\n  --profile=SWING ^\n  --config=configs/experiments/ctd_cost_resilient_swing_v1.yaml\n```\n\nExpected console output:\n\n```text\nLatest production-candidate signal generated\nTicker/Profile: CTD SWING\nCandidate:      production_candidate_cost_resilient_v1\nAction:         HOLD / WATCHLIST / BUY\n```\n\nThe runner writes the latest signal report to:\n\n```text\nreports/signals/CTD_SWING_latest_signal.md\nreports/signals/CTD_SWING_latest_signal.json\n```\n\n## Reproduce the production candidate report\n\nThe current production candidate report is stored in:\n\n```text\nreports/production_candidates/CTD_SWING_production_candidate_cost_resilient_v1.md\nreports/production_candidates/CTD_SWING_production_candidate_cost_resilient_v1.json\n```\n\nThe frozen-grid setup behind the candidate uses:\n\n```text\nfeature_set = candidate_cost_resilient_recovery_no_noise_v1\nlookahead   = 40\ntp          = 0.10\nsl          = -0.05\nthreshold   = 0.60\ncost stress = 0.005\n```\n\n## Development workflow\n\nUse short-lived branches and Pull Requests:\n\n```bash\ngit checkout master\ngit pull origin master\ngit checkout -b <branch-name>\n```\n\nAfter changes:\n\n```bash\ngit status\ngit diff --cached --name-only\ngit commit -m "<message>"\ngit push -u origin <branch-name>\n```\n\nGenerated signal reports often change timestamps. Unless the report itself is intentionally updated, restore them before committing:\n\n```bash\ngit restore reports/signals/CTD_SWING_latest_signal.json\ngit restore reports/signals/CTD_SWING_latest_signal.md\n```\n\n## Active entrypoints\n\n| Purpose | Command / File |\n|---|---|\n| Latest production signal | `scripts/production/run_signal.py` |\n| Production config | `configs/experiments/ctd_cost_resilient_swing_v1.yaml` |\n| Production signal engine | `src/ml_signal/production/signal_engine.py` |\n| Production config loader | `src/ml_signal/production/config_loader.py` |\n| Signal report writer | `src/ml_signal/production/signal_report.py` |\n\n## Legacy scripts\n\nOlder root-level research runners were moved to:\n\n```text\narchive/legacy/root_scripts/\n```\n\nThese files are kept for reproducibility and historical reference. New reusable logic should be added under `src/ml_signal/` instead of importing from archived scripts.\n\n## Notes\n\nThis repository is a research system. Backtest and frozen-grid evidence are useful for hypothesis evaluation, but live market performance can differ due to regime shifts, liquidity, transaction costs, data quality, and execution constraints.\n',
    "docs/PROJECT_STRUCTURE.md": '\n# Project Structure\n\nThis document explains the current folder layout after the refactor and root cleanup work.\n\n## Main folders\n\n```text\nsrc/ml_signal/\n```\n\nReusable Python package. New production and research logic should live here.\n\n```text\nscripts/production/\n```\n\nProduction-oriented command-line entrypoints. The main runner is:\n\n```text\nscripts/production/run_signal.py\n```\n\n```text\nscripts/research/\n```\n\nResearch utilities and report-generation scripts.\n\n```text\nconfigs/\n```\n\nExperiment and production-candidate YAML configs.\n\n```text\nreports/\n```\n\nGenerated markdown and JSON reports.\n\n```text\narchive/\n```\n\nHistorical patch scripts and legacy runners. Files in this folder are retained for auditability, not as preferred active entrypoints.\n\n## Current production flow\n\n```text\nconfigs/experiments/ctd_cost_resilient_swing_v1.yaml\n        ↓\nscripts/production/run_signal.py\n        ↓\nsrc/ml_signal/production/config_loader.py\n        ↓\nsrc/ml_signal/production/signal_engine.py\n        ↓\nsrc/ml_signal/pipelines/\nsrc/ml_signal/features/\nsrc/ml_signal/models/\nsrc/ml_signal/evaluation/\n        ↓\nreports/signals/\n```\n\n## Where to add new logic\n\n| New work | Preferred location |\n|---|---|\n| New feature family | `src/ml_signal/features/` |\n| New feature set name | `src/ml_signal/features/registry.py` |\n| New label logic | `src/ml_signal/labels/` |\n| New model factory | `src/ml_signal/models/` |\n| New production runner | `scripts/production/` |\n| New research report script | `scripts/research/` |\n| New production config | `configs/experiments/` |\n\n## Legacy policy\n\nDo not import from:\n\n```text\narchive/legacy/root_scripts/\n```\n\nWhen legacy logic is needed, port the required function into `src/ml_signal/` first.\n',
    "docs/REPRODUCIBILITY.md": '\n# Reproducibility Notes\n\n## Production signal\n\nRun from project root:\n\n```bash\npython scripts/production/run_signal.py ^\n  --ticker=CTD ^\n  --profile=SWING ^\n  --config=configs/experiments/ctd_cost_resilient_swing_v1.yaml\n```\n\nGenerated files:\n\n```text\nreports/signals/CTD_SWING_latest_signal.md\nreports/signals/CTD_SWING_latest_signal.json\n```\n\n## Current production-candidate setup\n\n```text\nticker: CTD\nprofile: SWING\ncandidate: production_candidate_cost_resilient_v1\nfeature_set: candidate_cost_resilient_recovery_no_noise_v1\nlookahead: 40\ntp: 0.10\nsl: -0.05\nthreshold: 0.60\nstress_round_trip_cost: 0.005\n```\n\n## Data assumptions\n\nThe project expects standardized market data under:\n\n```text\ndata/processed_data/\n```\n\nExample expected files:\n\n```text\nCTD_standardized.csv\nVNINDEX_standardized.csv\n```\n\nValuation data, when used, is expected under:\n\n```text\ndata/processed_data/valuation/\n```\n\n## Git hygiene\n\nSignal reports can change when the production runner is executed because timestamps are regenerated. Restore them unless the report update is intentional:\n\n```bash\ngit restore reports/signals/CTD_SWING_latest_signal.json\ngit restore reports/signals/CTD_SWING_latest_signal.md\n```\n',
}


def backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_before_docs_readme_v1")

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


def main() -> None:
    print("=" * 100)
    print("Docs README v1 patch")
    print("=" * 100)

    for rel_path, content in FILES.items():
        write_file(rel_path, content)

    print("=" * 100)
    print("Done.")
    print("Review README.md, docs/PROJECT_STRUCTURE.md, and docs/REPRODUCIBILITY.md.")
    print("=" * 100)


if __name__ == "__main__":
    main()
