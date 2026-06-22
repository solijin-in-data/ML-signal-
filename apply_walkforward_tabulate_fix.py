from __future__ import annotations

from pathlib import Path
import shutil


TARGET_FILE = Path("walkforward_stability_runner.py")
BACKUP_FILE = Path("walkforward_stability_runner.py.bak_before_tabulate_fix")


def apply_patch() -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(
            "walkforward_stability_runner.py was not found. "
            "Run this patch from the project root."
        )

    text = TARGET_FILE.read_text(encoding="utf-8")

    if ".to_markdown(index=False)" not in text:
        print("No pandas to_markdown calls found. Patch may already be applied.")
        return

    if not BACKUP_FILE.exists():
        shutil.copy2(TARGET_FILE, BACKUP_FILE)
        print(f"Backup created: {BACKUP_FILE}")

    text = text.replace(
        "stability.to_markdown(index=False)",
        "stability.to_string(index=False)",
    )

    text = text.replace(
        "reoptimized.to_markdown(index=False)",
        "reoptimized.to_string(index=False)",
    )

    TARGET_FILE.write_text(text, encoding="utf-8")

    print("Walk-forward tabulate fix applied successfully.")
    print("Replaced pandas to_markdown calls with to_string.")
    print("You can now rerun:")
    print("python walkforward_stability_runner.py --ticker CTD --profile SWING")


if __name__ == "__main__":
    apply_patch()
