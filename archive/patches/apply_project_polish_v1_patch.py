from __future__ import annotations

import py_compile
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

FILES = {
    "requirements.txt": '\nnumpy>=1.24\npandas>=2.0\nscikit-learn>=1.3\nxgboost>=2.0\nPyYAML>=6.0\nopenpyxl>=3.1\nmatplotlib>=3.7\ntabulate>=0.9\n',
    "requirements-dev.txt": '\n-r requirements.txt\n\npytest>=8.0\nruff>=0.5\n',
    ".gitignore": '\n# Python\n__pycache__/\n*.py[cod]\n*.pyo\n*.pyd\n*.so\n.Python\n.pytest_cache/\n.ruff_cache/\n.mypy_cache/\n.pyright/\n.coverage\nhtmlcov/\n\n# Virtual environments\n.venv/\nvenv/\nenv/\nENV/\n\n# IDE / OS\n.vscode/\n.idea/\n.DS_Store\nThumbs.db\n\n# Local artifacts\n*.bak\n*.bak_*\n*.tmp\n*.log\n\n# Patch-generated backups\n*.bak_before_*\n\n# Local runtime output\noutputs/\nlogs/\n\n# Local notebooks checkpoints\n.ipynb_checkpoints/\n\n# Local secrets\n.env\n.env.*\n*.pem\n*.key\n\n# Do not ignore source code, configs, reports, or archived scripts by default.\n',
    ".github/workflows/python-ci.yml": '\nname: Python CI\n\non:\n  pull_request:\n    branches:\n      - master\n  push:\n    branches:\n      - master\n\njobs:\n  syntax-check:\n    name: Syntax and package check\n    runs-on: ubuntu-latest\n\n    steps:\n      - name: Checkout repository\n        uses: actions/checkout@v4\n\n      - name: Set up Python\n        uses: actions/setup-python@v5\n        with:\n          python-version: "3.11"\n\n      - name: Install dependencies\n        run: |\n          python -m pip install --upgrade pip\n          if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; elif [ -f requirements.txt ]; then pip install -r requirements.txt; fi\n\n      - name: Compile Python files\n        run: |\n          python -m compileall src scripts -q\n\n      - name: Import package\n        run: |\n          python - <<\'PY\'\n          import sys\n          from pathlib import Path\n\n          root = Path.cwd()\n          sys.path.insert(0, str(root / "src"))\n\n          import ml_signal  # noqa: F401\n\n          print("ml_signal package import OK")\n          PY\n',
    "docs/DEVELOPMENT.md": '\n# Development Guide\n\nThis guide records the basic development workflow for the repository.\n\n## 1. Create a branch\n\n```bash\ngit checkout master\ngit pull origin master\ngit checkout -b <branch-name>\n```\n\n## 2. Make changes\n\nPreferred locations:\n\n```text\nsrc/ml_signal/          reusable package logic\nscripts/production/     production entrypoints\nscripts/research/       research scripts\nconfigs/                YAML configs\ndocs/                   documentation\nreports/                committed report snapshots\narchive/                historical scripts and patch history\n```\n\n## 3. Validate locally\n\nCompile source and script files:\n\n```bash\npython -m compileall src scripts -q\n```\n\nRun the current production-candidate signal:\n\n```bash\npython scripts/production/run_signal.py ^\n  --ticker=CTD ^\n  --profile=SWING ^\n  --config=configs/experiments/ctd_cost_resilient_swing_v1.yaml\n```\n\nRunning the signal command may update timestamped report files. Restore them unless the report update is intentional:\n\n```bash\ngit restore reports/signals/CTD_SWING_latest_signal.json\ngit restore reports/signals/CTD_SWING_latest_signal.md\n```\n\n## 4. Commit\n\n```bash\ngit status\ngit diff --cached --name-status\ngit commit -m "<message>"\ngit push -u origin <branch-name>\n```\n\n## 5. Pull Request\n\nOpen a PR into `master`. The GitHub Actions workflow checks Python syntax and verifies that the `ml_signal` package can be imported.\n\n## Notes\n\n- Keep new reusable logic under `src/ml_signal/`.\n- Do not import from `archive/legacy/root_scripts/`.\n- Keep production entrypoints thin; move reusable logic into package modules.\n- Do not commit local backup files or generated cache folders.\n',
}


def backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_before_project_polish_v1")

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
    print("Project polish v1 patch")
    print("=" * 100)

    for rel_path, content in FILES.items():
        write_file(rel_path, content)

    py_compile.compile(str(ROOT / "apply_project_polish_v1_patch.py"), doraise=True)

    print("=" * 100)
    print("Done.")
    print("Recommended local checks:")
    print("python -m compileall src scripts -q")
    print("python scripts/production/run_signal.py --ticker=CTD --profile=SWING --config=configs/experiments/ctd_cost_resilient_swing_v1.yaml")
    print("=" * 100)


if __name__ == "__main__":
    main()
