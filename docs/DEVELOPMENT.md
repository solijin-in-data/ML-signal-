# Development Guide

This guide records the basic development workflow for the repository.

## 1. Create a branch

```bash
git checkout master
git pull origin master
git checkout -b <branch-name>
```

## 2. Make changes

Preferred locations:

```text
src/ml_signal/          reusable package logic
scripts/production/     production entrypoints
scripts/research/       research scripts
configs/                YAML configs
docs/                   documentation
reports/                committed report snapshots
archive/                historical scripts and patch history
```

## 3. Validate locally

Compile source and script files:

```bash
python -m compileall src scripts -q
```

Run the current production-candidate signal:

```bash
python scripts/production/run_signal.py ^
  --ticker=CTD ^
  --profile=SWING ^
  --config=configs/experiments/ctd_cost_resilient_swing_v1.yaml
```

Running the signal command may update timestamped report files. Restore them unless the report update is intentional:

```bash
git restore reports/signals/CTD_SWING_latest_signal.json
git restore reports/signals/CTD_SWING_latest_signal.md
```

## 4. Commit

```bash
git status
git diff --cached --name-status
git commit -m "<message>"
git push -u origin <branch-name>
```

## 5. Pull Request

Open a PR into `master`. The GitHub Actions workflow checks Python syntax and verifies that the `ml_signal` package can be imported.

## Notes

- Keep new reusable logic under `src/ml_signal/`.
- Do not import from `archive/legacy/root_scripts/`.
- Keep production entrypoints thin; move reusable logic into package modules.
- Do not commit local backup files or generated cache folders.
