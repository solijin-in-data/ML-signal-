from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

LEGACY_DIR = ROOT / "archive" / "legacy" / "root_scripts"

# Conservative cleanup list.
# Do not move config.py, data_standardizer.py, balance_sheet_standardizer.py,
# or valuation_builder.py because they may still be imported by active modules.
LEGACY_ROOT_FILES = [
    "feature_experiment_runner.py",
    "frozen_parameter_grid_runner.py",
    "frozen_walkforward_runner.py",
    "walkforward_stability_runner.py",
    "main_signal.py",
    "main_signal_radar_v2.py",
    "summarize_ablation_results.py",
    "summarize_feature_experiments.py",
]

README_CONTENT = """# Legacy Root Scripts

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
"""


def move_file_to_legacy(filename: str) -> None:
    src = ROOT / filename

    if not src.exists():
        print(f"[SKIP] missing: {filename}")
        return

    if src.is_dir():
        print(f"[SKIP] is directory, not moving: {filename}")
        return

    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    dst = LEGACY_DIR / filename

    if dst.exists():
        raise FileExistsError(
            f"Destination already exists. Refusing to overwrite: {dst.relative_to(ROOT)}"
        )

    shutil.move(str(src), str(dst))
    print(f"[MOVE] {filename} -> {dst.relative_to(ROOT)}")


def write_readme() -> None:
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    readme_path = LEGACY_DIR / "README.md"

    current = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    if current == README_CONTENT:
        print(f"[SKIP] unchanged: {readme_path.relative_to(ROOT)}")
        return

    readme_path.write_text(README_CONTENT, encoding="utf-8")
    print(f"[WRITE] {readme_path.relative_to(ROOT)}")


def main() -> None:
    print("=" * 100)
    print("Root cleanup v1 patch")
    print("=" * 100)

    for filename in LEGACY_ROOT_FILES:
        move_file_to_legacy(filename)

    write_readme()

    print("=" * 100)
    print("Done.")
    print("Validate production runner after cleanup:")
    print("python scripts/production/run_signal.py --ticker=CTD --profile=SWING --config=configs/experiments/ctd_cost_resilient_swing_v1.yaml")
    print("=" * 100)


if __name__ == "__main__":
    main()
