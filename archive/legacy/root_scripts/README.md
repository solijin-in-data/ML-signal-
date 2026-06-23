# Legacy Root Scripts

This folder stores older root-level research runners that were moved out of the project root during `chore/root-cleanup-v1`.

These scripts are kept for reproducibility and historical reference. New production and reusable logic should live under:

```text
src/ml_signal/
scripts/production/
scripts/research/
configs/
reports/
```

## Why these files were archived

The project has moved toward a cleaner source layout:

```text
src/ml_signal/production/
src/ml_signal/pipelines/
src/ml_signal/features/
src/ml_signal/models/
src/ml_signal/evaluation/
```

The current production signal entrypoint is:

```bash
python scripts/production/run_signal.py --ticker=CTD --profile=SWING --config=configs/experiments/ctd_cost_resilient_swing_v1.yaml
```

## Important note

Archived scripts may not be maintained as active entrypoints. If an archived script is needed again, port the required function into `src/ml_signal/` instead of importing from this archive.
