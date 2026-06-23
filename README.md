# ML Signal Radar

A research-oriented machine learning signal framework for Vietnamese equity data.

The current production-candidate workflow focuses on **CTD SWING**, using a cost-resilient recovery feature set and a frozen-grid validation process. The project separates reusable logic under `src/ml_signal/`, command-line entrypoints under `scripts/`, configuration under `configs/`, generated reports under `reports/`, and archived legacy scripts under `archive/`.

> Research only. This project is not investment advice, a trading recommendation, or a promise of future performance.

## Current production candidate

| Item | Value |
|---|---|
| Ticker | CTD |
| Profile | SWING |
| Candidate | `production_candidate_cost_resilient_v1` |
| Feature set | `candidate_cost_resilient_recovery_no_noise_v1` |
| Lookahead | 40 trading days |
| Take profit | 10% |
| Stop loss | -5% |
| Signal threshold | 60% |
| Stress round-trip cost | 0.5% |
| Production config | `configs/experiments/ctd_cost_resilient_swing_v1.yaml` |

The production-candidate setup was selected from frozen-grid evidence after testing cost-resilient recovery features. The strongest CTD SWING setup retained liquidity and recovery-quality features while excluding the noise-filter family.

## Project layout

```text
ML-signal-/
├── configs/
│   └── experiments/
│       └── ctd_cost_resilient_swing_v1.yaml
├── data/
│   └── processed_data/
├── reports/
│   ├── production_candidates/
│   └── signals/
├── scripts/
│   ├── production/
│   │   └── run_signal.py
│   └── research/
├── src/
│   └── ml_signal/
│       ├── evaluation/
│       ├── features/
│       ├── labels/
│       ├── models/
│       ├── pipelines/
│       └── production/
└── archive/
    ├── legacy/
    │   └── root_scripts/
    └── patches/
```

## Run the latest production-candidate signal

From the project root:

```bash
python scripts/production/run_signal.py ^
  --ticker=CTD ^
  --profile=SWING ^
  --config=configs/experiments/ctd_cost_resilient_swing_v1.yaml
```

Expected console output:

```text
Latest production-candidate signal generated
Ticker/Profile: CTD SWING
Candidate:      production_candidate_cost_resilient_v1
Action:         HOLD / WATCHLIST / BUY
```

The runner writes the latest signal report to:

```text
reports/signals/CTD_SWING_latest_signal.md
reports/signals/CTD_SWING_latest_signal.json
```

## Reproduce the production candidate report

The current production candidate report is stored in:

```text
reports/production_candidates/CTD_SWING_production_candidate_cost_resilient_v1.md
reports/production_candidates/CTD_SWING_production_candidate_cost_resilient_v1.json
```

The frozen-grid setup behind the candidate uses:

```text
feature_set = candidate_cost_resilient_recovery_no_noise_v1
lookahead   = 40
tp          = 0.10
sl          = -0.05
threshold   = 0.60
cost stress = 0.005
```

## Development workflow

Use short-lived branches and Pull Requests:

```bash
git checkout master
git pull origin master
git checkout -b <branch-name>
```

After changes:

```bash
git status
git diff --cached --name-only
git commit -m "<message>"
git push -u origin <branch-name>
```

Generated signal reports often change timestamps. Unless the report itself is intentionally updated, restore them before committing:

```bash
git restore reports/signals/CTD_SWING_latest_signal.json
git restore reports/signals/CTD_SWING_latest_signal.md
```

## Active entrypoints

| Purpose | Command / File |
|---|---|
| Latest production signal | `scripts/production/run_signal.py` |
| Production config | `configs/experiments/ctd_cost_resilient_swing_v1.yaml` |
| Production signal engine | `src/ml_signal/production/signal_engine.py` |
| Production config loader | `src/ml_signal/production/config_loader.py` |
| Signal report writer | `src/ml_signal/production/signal_report.py` |

## Legacy scripts

Older root-level research runners were moved to:

```text
archive/legacy/root_scripts/
```

These files are kept for reproducibility and historical reference. New reusable logic should be added under `src/ml_signal/` instead of importing from archived scripts.

## Notes

This repository is a research system. Backtest and frozen-grid evidence are useful for hypothesis evaluation, but live market performance can differ due to regime shifts, liquidity, transaction costs, data quality, and execution constraints.
