from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProductionSignalConfig:
    ticker: str
    profile: str
    candidate_name: str
    feature_set: str
    lookahead: int
    tp: float
    sl: float
    threshold: float
    round_trip_cost: float = 0.0
    min_edge_vs_breakeven: float = 0.0
    watch_margin: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip("'").strip('"')

    if value.lower() in {"true", "false"}:
        return value.lower() == "true"

    if value.lower() in {"none", "null", "~"}:
        return None

    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """
    Small fallback parser for simple project config files.

    It supports:
    - top-level `key: value`
    - one-level nested dictionaries via indentation

    PyYAML is used when available. This fallback keeps the production runner usable
    in lightweight environments.
    """
    root: dict[str, Any] = {}
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if not line.strip() or line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        if indent == 0 and value == "":
            current_section = key
            root[current_section] = {}
            continue

        if indent == 0:
            current_section = None
            root[key] = _parse_scalar(value)
            continue

        if current_section is not None:
            root.setdefault(current_section, {})
            root[current_section][key] = _parse_scalar(value)

    return root


def load_yaml_like(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        return _simple_yaml_load(text)


def discover_experiment_config(
    project_root: Path,
    ticker: str,
    profile: str,
) -> Path | None:
    config_dir = project_root / "configs" / "experiments"

    if not config_dir.exists():
        return None

    ticker_token = ticker.lower()
    profile_token = profile.lower()

    candidates = [
        path
        for path in config_dir.glob("*.y*ml")
        if ticker_token in path.name.lower()
        and profile_token in path.name.lower()
    ]

    if not candidates:
        return None

    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        points = 0
        if "production" in name:
            points += 3
        if "cost_resilient" in name or "cost-resilient" in name:
            points += 2
        if "candidate" in name:
            points += 1
        return (-points, name)

    return sorted(candidates, key=score)[0]


def _first_present(
    data: dict[str, Any],
    keys: list[str],
    default: Any = None,
) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    for section_name in ["candidate", "production_candidate", "setup", "model", "strategy"]:
        section = data.get(section_name)

        if isinstance(section, dict):
            for key in keys:
                if key in section and section[key] is not None:
                    return section[key]

    return default


def load_production_signal_config(
    project_root: Path,
    ticker: str,
    profile: str,
    config_path: str | None = None,
) -> ProductionSignalConfig:
    ticker = ticker.upper()
    profile = profile.upper()

    path: Path | None

    if config_path:
        path = Path(config_path)

        if not path.is_absolute():
            path = project_root / path
    else:
        path = discover_experiment_config(project_root, ticker, profile)

    if path is None or not path.exists():
        raise FileNotFoundError(
            "Cannot find production experiment config. "
            "Pass --config configs/experiments/<file>.yaml or create one first."
        )

    data = load_yaml_like(path)

    candidate_name = _first_present(
        data,
        ["candidate_name", "name", "production_candidate_name"],
        "production_candidate",
    )

    feature_set = _first_present(
        data,
        ["feature_set", "features", "feature_set_name"],
        None,
    )

    if feature_set is None:
        raise ValueError(f"Config is missing feature_set: {path}")

    lookahead = _first_present(data, ["lookahead", "LOOKAHEAD"], None)
    tp = _first_present(data, ["tp", "TP"], None)
    sl = _first_present(data, ["sl", "SL"], None)
    threshold = _first_present(data, ["threshold", "THRESH", "thresh"], None)

    missing = [
        name
        for name, value in {
            "lookahead": lookahead,
            "tp": tp,
            "sl": sl,
            "threshold": threshold,
        }.items()
        if value is None
    ]

    if missing:
        raise ValueError(f"Config is missing required setup values: {missing}")

    return ProductionSignalConfig(
        ticker=str(_first_present(data, ["ticker"], ticker)).upper(),
        profile=str(_first_present(data, ["profile"], profile)).upper(),
        candidate_name=str(candidate_name),
        feature_set=str(feature_set),
        lookahead=int(lookahead),
        tp=float(tp),
        sl=float(sl),
        threshold=float(threshold),
        round_trip_cost=float(
            _first_present(data, ["round_trip_cost", "cost", "roundtrip_cost"], 0.0)
        ),
        min_edge_vs_breakeven=float(
            _first_present(data, ["min_edge_vs_breakeven", "min_edge"], 0.0)
        ),
    )


def apply_overrides(
    config: ProductionSignalConfig,
    *,
    candidate_name: str | None = None,
    feature_set: str | None = None,
    lookahead: int | None = None,
    tp: float | None = None,
    sl: float | None = None,
    threshold: float | None = None,
    round_trip_cost: float | None = None,
    min_edge_vs_breakeven: float | None = None,
    watch_margin: float | None = None,
) -> ProductionSignalConfig:
    return ProductionSignalConfig(
        ticker=config.ticker,
        profile=config.profile,
        candidate_name=candidate_name or config.candidate_name,
        feature_set=feature_set or config.feature_set,
        lookahead=lookahead if lookahead is not None else config.lookahead,
        tp=tp if tp is not None else config.tp,
        sl=sl if sl is not None else config.sl,
        threshold=threshold if threshold is not None else config.threshold,
        round_trip_cost=(
            round_trip_cost
            if round_trip_cost is not None
            else config.round_trip_cost
        ),
        min_edge_vs_breakeven=(
            min_edge_vs_breakeven
            if min_edge_vs_breakeven is not None
            else config.min_edge_vs_breakeven
        ),
        watch_margin=watch_margin if watch_margin is not None else config.watch_margin,
    )
