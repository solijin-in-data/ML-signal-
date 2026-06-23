# Project Structure

This document explains the current folder layout after the refactor and root cleanup work.

## Main folders

```text
src/ml_signal/
```

Reusable Python package. New production and research logic should live here.

```text
scripts/production/
```

Production-oriented command-line entrypoints. The main runner is:

```text
scripts/production/run_signal.py
```

```text
scripts/research/
```

Research utilities and report-generation scripts.

```text
configs/
```

Experiment and production-candidate YAML configs.

```text
reports/
```

Generated markdown and JSON reports.

```text
archive/
```

Historical patch scripts and legacy runners. Files in this folder are retained for auditability, not as preferred active entrypoints.

## Current production flow

```text
configs/experiments/ctd_cost_resilient_swing_v1.yaml
        ↓
scripts/production/run_signal.py
        ↓
src/ml_signal/production/config_loader.py
        ↓
src/ml_signal/production/signal_engine.py
        ↓
src/ml_signal/pipelines/
src/ml_signal/features/
src/ml_signal/models/
src/ml_signal/evaluation/
        ↓
reports/signals/
```

## Where to add new logic

| New work | Preferred location |
|---|---|
| New feature family | `src/ml_signal/features/` |
| New feature set name | `src/ml_signal/features/registry.py` |
| New label logic | `src/ml_signal/labels/` |
| New model factory | `src/ml_signal/models/` |
| New production runner | `scripts/production/` |
| New research report script | `scripts/research/` |
| New production config | `configs/experiments/` |

## Legacy policy

Do not import from:

```text
archive/legacy/root_scripts/
```

When legacy logic is needed, port the required function into `src/ml_signal/` first.
