from __future__ import annotations

import py_compile
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

CONFIG_LOADER = '\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass, asdict\nfrom pathlib import Path\nfrom typing import Any\n\n\n@dataclass(frozen=True)\nclass ProductionSignalConfig:\n    ticker: str\n    profile: str\n    candidate_name: str\n    feature_set: str\n    lookahead: int\n    tp: float\n    sl: float\n    threshold: float\n    round_trip_cost: float = 0.0\n    min_edge_vs_breakeven: float = 0.0\n    watch_margin: float = 0.05\n\n    def to_dict(self) -> dict[str, Any]:\n        return asdict(self)\n\n\ndef _parse_scalar(value: str) -> Any:\n    value = value.strip().strip("\'").strip(\'"\')\n\n    if value.lower() in {"true", "false"}:\n        return value.lower() == "true"\n\n    if value.lower() in {"none", "null", "~"}:\n        return None\n\n    try:\n        if any(char in value for char in [".", "e", "E"]):\n            return float(value)\n        return int(value)\n    except ValueError:\n        return value\n\n\ndef _simple_yaml_load(text: str) -> dict[str, Any]:\n    """\n    Lightweight fallback YAML parser for simple project config files.\n\n    Supports:\n    - top-level key: value\n    - one-level nested dictionaries\n    - two-level nested dictionaries for validation metadata\n    """\n    root: dict[str, Any] = {}\n    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]\n\n    for raw_line in text.splitlines():\n        line = raw_line.rstrip()\n\n        if not line.strip() or line.lstrip().startswith("#"):\n            continue\n\n        indent = len(line) - len(line.lstrip(" "))\n        stripped = line.strip()\n\n        if ":" not in stripped:\n            continue\n\n        key, value = stripped.split(":", 1)\n        key = key.strip()\n        value = value.strip()\n\n        while stack and indent <= stack[-1][0]:\n            stack.pop()\n\n        parent = stack[-1][1]\n\n        if value == "":\n            parent[key] = {}\n            stack.append((indent, parent[key]))\n        else:\n            parent[key] = _parse_scalar(value)\n\n    return root\n\n\ndef load_yaml_like(path: Path) -> dict[str, Any]:\n    text = path.read_text(encoding="utf-8")\n\n    try:\n        import yaml  # type: ignore\n\n        loaded = yaml.safe_load(text)\n        return loaded or {}\n    except Exception:\n        return _simple_yaml_load(text)\n\n\ndef discover_experiment_config(\n    project_root: Path,\n    ticker: str,\n    profile: str,\n) -> Path | None:\n    config_dir = project_root / "configs" / "experiments"\n\n    if not config_dir.exists():\n        return None\n\n    ticker_token = ticker.lower()\n    profile_token = profile.lower()\n\n    candidates = [\n        path\n        for path in config_dir.glob("*.y*ml")\n        if ticker_token in path.name.lower()\n        and profile_token in path.name.lower()\n    ]\n\n    if not candidates:\n        return None\n\n    def score(path: Path) -> tuple[int, str]:\n        name = path.name.lower()\n        points = 0\n        if "production" in name:\n            points += 3\n        if "cost_resilient" in name or "cost-resilient" in name:\n            points += 2\n        if "candidate" in name:\n            points += 1\n        return (-points, name)\n\n    return sorted(candidates, key=score)[0]\n\n\ndef _first_present(\n    data: dict[str, Any],\n    keys: list[str],\n    default: Any = None,\n) -> Any:\n    for key in keys:\n        if key in data and data[key] is not None:\n            return data[key]\n\n    # Backward-compatible sections from earlier patches.\n    for section_name in ["candidate", "production_candidate", "setup", "model", "strategy"]:\n        section = data.get(section_name)\n\n        if isinstance(section, dict):\n            for key in keys:\n                if key in section and section[key] is not None:\n                    return section[key]\n\n    return default\n\n\ndef _deep_first_present(\n    data: dict[str, Any],\n    paths: list[tuple[str, ...]],\n    default: Any = None,\n) -> Any:\n    """\n    Read either top-level or nested YAML paths.\n\n    Example:\n        _deep_first_present(data, [("feature_set",), ("experiment", "feature_set")])\n    """\n    for path in paths:\n        current: Any = data\n\n        for part in path:\n            if not isinstance(current, dict) or part not in current:\n                current = None\n                break\n            current = current[part]\n\n        if current is not None:\n            return current\n\n    return default\n\n\ndef _required_value(data: dict[str, Any], field_name: str, paths: list[tuple[str, ...]]) -> Any:\n    value = _deep_first_present(data, paths)\n\n    if value is None:\n        readable_paths = [".".join(path) for path in paths]\n        raise ValueError(\n            f"Config is missing required field \'{field_name}\'. "\n            f"Tried: {readable_paths}"\n        )\n\n    return value\n\n\ndef load_production_signal_config(\n    project_root: Path,\n    ticker: str,\n    profile: str,\n    config_path: str | None = None,\n) -> ProductionSignalConfig:\n    ticker = ticker.upper()\n    profile = profile.upper()\n\n    path: Path | None\n\n    if config_path:\n        path = Path(config_path)\n\n        if not path.is_absolute():\n            path = project_root / path\n    else:\n        path = discover_experiment_config(project_root, ticker, profile)\n\n    if path is None or not path.exists():\n        raise FileNotFoundError(\n            "Cannot find production experiment config. "\n            "Pass --config configs/experiments/<file>.yaml or create one first."\n        )\n\n    data = load_yaml_like(path)\n\n    loaded_ticker = _deep_first_present(\n        data,\n        [\n            ("ticker",),\n            ("experiment", "ticker"),\n        ],\n        ticker,\n    )\n\n    loaded_profile = _deep_first_present(\n        data,\n        [\n            ("profile",),\n            ("experiment", "profile"),\n        ],\n        profile,\n    )\n\n    candidate_name = _deep_first_present(\n        data,\n        [\n            ("candidate_name",),\n            ("production_candidate_name",),\n            ("candidate", "name"),\n            ("production_candidate", "name"),\n            ("experiment", "candidate_name"),\n            ("experiment", "name"),\n            ("status", "stage"),\n        ],\n        "production_candidate",\n    )\n\n    feature_set = _required_value(\n        data,\n        "feature_set",\n        [\n            ("feature_set",),\n            ("feature_set_name",),\n            ("features",),\n            ("candidate", "feature_set"),\n            ("production_candidate", "feature_set"),\n            ("setup", "feature_set"),\n            ("model", "feature_set"),\n            ("strategy", "feature_set"),\n            ("experiment", "feature_set"),\n        ],\n    )\n\n    lookahead = _required_value(\n        data,\n        "lookahead",\n        [\n            ("lookahead",),\n            ("LOOKAHEAD",),\n            ("setup", "lookahead"),\n            ("model", "lookahead"),\n            ("strategy", "lookahead"),\n            ("frozen_setup", "lookahead"),\n        ],\n    )\n\n    tp = _required_value(\n        data,\n        "tp",\n        [\n            ("tp",),\n            ("TP",),\n            ("setup", "tp"),\n            ("model", "tp"),\n            ("strategy", "tp"),\n            ("frozen_setup", "tp"),\n        ],\n    )\n\n    sl = _required_value(\n        data,\n        "sl",\n        [\n            ("sl",),\n            ("SL",),\n            ("setup", "sl"),\n            ("model", "sl"),\n            ("strategy", "sl"),\n            ("frozen_setup", "sl"),\n        ],\n    )\n\n    threshold = _required_value(\n        data,\n        "threshold",\n        [\n            ("threshold",),\n            ("THRESH",),\n            ("thresh",),\n            ("setup", "threshold"),\n            ("model", "threshold"),\n            ("strategy", "threshold"),\n            ("frozen_setup", "threshold"),\n        ],\n    )\n\n    round_trip_cost = _deep_first_present(\n        data,\n        [\n            ("round_trip_cost",),\n            ("roundtrip_cost",),\n            ("cost",),\n            ("setup", "round_trip_cost"),\n            ("risk_assumptions", "round_trip_cost"),\n            ("risk_assumptions", "round_trip_cost_stress"),\n            ("risk_assumptions", "round_trip_cost_base"),\n        ],\n        0.0,\n    )\n\n    min_edge_vs_breakeven = _deep_first_present(\n        data,\n        [\n            ("min_edge_vs_breakeven",),\n            ("min_edge",),\n            ("setup", "min_edge_vs_breakeven"),\n            ("risk_assumptions", "min_edge_vs_breakeven"),\n        ],\n        0.0,\n    )\n\n    watch_margin = _deep_first_present(\n        data,\n        [\n            ("watch_margin",),\n            ("setup", "watch_margin"),\n            ("risk_assumptions", "watch_margin"),\n        ],\n        0.05,\n    )\n\n    return ProductionSignalConfig(\n        ticker=str(loaded_ticker).upper(),\n        profile=str(loaded_profile).upper(),\n        candidate_name=str(candidate_name),\n        feature_set=str(feature_set),\n        lookahead=int(lookahead),\n        tp=float(tp),\n        sl=float(sl),\n        threshold=float(threshold),\n        round_trip_cost=float(round_trip_cost),\n        min_edge_vs_breakeven=float(min_edge_vs_breakeven),\n        watch_margin=float(watch_margin),\n    )\n\n\ndef apply_overrides(\n    config: ProductionSignalConfig,\n    *,\n    candidate_name: str | None = None,\n    feature_set: str | None = None,\n    lookahead: int | None = None,\n    tp: float | None = None,\n    sl: float | None = None,\n    threshold: float | None = None,\n    round_trip_cost: float | None = None,\n    min_edge_vs_breakeven: float | None = None,\n    watch_margin: float | None = None,\n) -> ProductionSignalConfig:\n    return ProductionSignalConfig(\n        ticker=config.ticker,\n        profile=config.profile,\n        candidate_name=candidate_name or config.candidate_name,\n        feature_set=feature_set or config.feature_set,\n        lookahead=lookahead if lookahead is not None else config.lookahead,\n        tp=tp if tp is not None else config.tp,\n        sl=sl if sl is not None else config.sl,\n        threshold=threshold if threshold is not None else config.threshold,\n        round_trip_cost=(\n            round_trip_cost\n            if round_trip_cost is not None\n            else config.round_trip_cost\n        ),\n        min_edge_vs_breakeven=(\n            min_edge_vs_breakeven\n            if min_edge_vs_breakeven is not None\n            else config.min_edge_vs_breakeven\n        ),\n        watch_margin=watch_margin if watch_margin is not None else config.watch_margin,\n    )\n'
YAML_CONTENT = '\nexperiment:\n  name: ctd_cost_resilient_swing_v1\n  ticker: CTD\n  profile: SWING\n  candidate_name: production_candidate_cost_resilient_v1\n  feature_set: candidate_cost_resilient_recovery_no_noise_v1\n\nfrozen_setup:\n  lookahead: 40\n  tp: 0.10\n  sl: -0.05\n  threshold: 0.60\n\nrisk_assumptions:\n  round_trip_cost_base: 0.003\n  round_trip_cost_stress: 0.005\n  min_edge_vs_breakeven: 0.02\n  watch_margin: 0.05\n\nvalidation_summary:\n  cost_0_003:\n    pass_periods: 3\n    periods: 4\n    min_edge: 0.0867\n    avg_return: 0.0348\n    min_return: 0.0265\n    total_trades: 887\n  cost_0_005:\n    pass_periods: 3\n    periods: 4\n    min_edge: 0.0733\n    avg_return: 0.0348\n    min_return: 0.0265\n    total_trades: 887\n\nstatus:\n  stage: production_candidate_cost_resilient\n  notes: >\n    Best frozen setup after cost-resilient feature experiment. Uses liquidity\n    and recovery-quality features while excluding noise-filter features, which\n    weakened the full candidate in CTD tests.\n'


def backup_file(path: Path, suffix: str) -> None:
    backup = path.with_suffix(path.suffix + suffix)

    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[BACKUP] {backup.relative_to(ROOT)}")
    else:
        print(f"[SKIP] backup exists: {backup.relative_to(ROOT)}")


def write_file(rel_path: str, content: str, backup_suffix: str) -> None:
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized = content.strip() + "\n"

    if path.exists() and path.read_text(encoding="utf-8") == normalized:
        print(f"[SKIP] unchanged: {rel_path}")
        return

    if path.exists():
        backup_file(path, backup_suffix)

    path.write_text(normalized, encoding="utf-8")
    print(f"[WRITE] {rel_path}")


def main() -> None:
    print("=" * 100)
    print("Nested production config loader patch")
    print("=" * 100)

    write_file(
        "src/ml_signal/production/config_loader.py",
        CONFIG_LOADER,
        ".bak_before_nested_config_loader",
    )

    write_file(
        "configs/experiments/ctd_cost_resilient_swing_v1.yaml",
        YAML_CONTENT,
        ".bak_before_nested_config_loader",
    )

    py_compile.compile(
        str(ROOT / "src/ml_signal/production/config_loader.py"),
        doraise=True,
    )

    print("[COMPILE OK] src/ml_signal/production/config_loader.py")
    print("=" * 100)
    print("Done.")
    print("Run:")
    print("python scripts/production/run_signal.py --ticker=CTD --profile=SWING --config=configs/experiments/ctd_cost_resilient_swing_v1.yaml")
    print("=" * 100)


if __name__ == "__main__":
    main()
