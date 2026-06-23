from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_GRID_COLUMNS = {
    "Feature_Set",
    "LOOKAHEAD",
    "TP",
    "SL",
    "THRESH",
    "Pass_Periods",
    "Avg_Edge",
    "Min_Edge",
    "Avg_Return",
    "Total_Trades",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a markdown production-candidate report from frozen grid outputs."
        )
    )

    parser.add_argument("--ticker", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--alias-name", default=None)
    parser.add_argument("--alias-of", default=None)

    parser.add_argument("--lookahead", type=int, default=None)
    parser.add_argument("--tp", type=float, default=None)
    parser.add_argument("--sl", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--round-trip-cost", type=float, default=None)

    parser.add_argument("--input", default=None, help="Optional frozen grid CSV path.")
    parser.add_argument(
        "--output-dir",
        default="reports/production_candidates",
        help="Directory where the markdown and JSON reports are written.",
    )

    parser.add_argument("--min-pass-periods", type=int, default=3)
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--min-avg-return", type=float, default=0.02)
    parser.add_argument("--min-trades", type=int, default=100)

    parser.add_argument(
        "--baseline-feature-set",
        default=None,
        help="Optional baseline feature set for comparison.",
    )

    return parser.parse_args()


def safe_float(value) -> float:
    if value is None:
        return math.nan

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return math.nan

    is_percent = text.endswith("%")
    text = text.replace("%", "").replace(",", "")

    try:
        number = float(text)
    except ValueError:
        return math.nan

    return number / 100.0 if is_percent else number


def format_percent(value, digits: int = 2) -> str:
    number = safe_float(value)

    if math.isnan(number):
        return "n/a"

    return f"{number * 100:.{digits}f}%"


def format_float(value, digits: int = 4) -> str:
    number = safe_float(value)

    if math.isnan(number):
        return "n/a"

    return f"{number:.{digits}f}"


def format_int(value) -> str:
    number = safe_float(value)

    if math.isnan(number):
        return "n/a"

    return f"{int(round(number)):,}"


def normalize_feature_set_name(text: str) -> str:
    return str(text).strip()


def discover_grid_csvs(root: Path) -> list[Path]:
    output_root = root / "outputs"

    if not output_root.exists():
        return []

    candidates = []

    for path in output_root.rglob("*.csv"):
        name = path.name.lower()

        if any(
            token in name
            for token in [
                "frozen",
                "grid",
                "stability",
                "period_summary",
                "raw",
                "ranking",
            ]
        ):
            candidates.append(path)

    # If file names are generic, still allow all CSVs under outputs as fallback.
    if not candidates:
        candidates = list(output_root.rglob("*.csv"))

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def read_grid_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if not REQUIRED_GRID_COLUMNS.issubset(set(df.columns)):
        return None

    df = df.copy()
    df["Source_File"] = str(path)
    df["Source_Modified"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat(
        timespec="seconds"
    )

    return df


def load_grid_data(root: Path, input_path: str | None) -> pd.DataFrame:
    if input_path:
        path = Path(input_path)

        if not path.is_absolute():
            path = root / path

        df = read_grid_csv(path)

        if df is None:
            raise SystemExit(
                f"Input CSV does not look like a frozen grid result: {path}"
            )

        return df

    frames = []

    for path in discover_grid_csvs(root):
        df = read_grid_csv(path)

        if df is not None:
            frames.append(df)

    if not frames:
        raise SystemExit(
            "No frozen grid CSV found. Run frozen_parameter_grid_runner.py first, "
            "or pass --input path/to/frozen_grid.csv"
        )

    combined = pd.concat(frames, ignore_index=True)

    # Prefer the most recently modified source file while still allowing fallback.
    combined = combined.sort_values(["Source_Modified"], ascending=False)

    return combined


def value_matches(series: pd.Series, expected: float | int | None, tol: float = 1e-9) -> pd.Series:
    if expected is None:
        return pd.Series([True] * len(series), index=series.index)

    numeric = series.map(safe_float)

    return (numeric - float(expected)).abs() <= tol


def strict_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    return text in {"true", "1", "yes", "y"}


def select_candidate_row(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    feature_names = {
        normalize_feature_set_name(args.feature_set),
    }

    if args.alias_of:
        feature_names.add(normalize_feature_set_name(args.alias_of))

    if args.alias_name:
        feature_names.add(normalize_feature_set_name(args.alias_name))

    subset = df[df["Feature_Set"].astype(str).map(normalize_feature_set_name).isin(feature_names)].copy()

    subset = subset[
        value_matches(subset["LOOKAHEAD"], args.lookahead)
        & value_matches(subset["TP"], args.tp)
        & value_matches(subset["SL"], args.sl)
        & value_matches(subset["THRESH"], args.threshold)
    ]

    if subset.empty:
        available = (
            df[["Feature_Set", "LOOKAHEAD", "TP", "SL", "THRESH"]]
            .drop_duplicates()
            .head(30)
            .to_string(index=False)
        )

        raise SystemExit(
            "No matching candidate row found in frozen grid output.\n\n"
            f"Requested feature names: {sorted(feature_names)}\n\n"
            "Available examples:\n"
            f"{available}"
        )

    subset = subset.copy()
    subset["_strict_candidate"] = subset.get(
        "Frozen_Strict_Candidate",
        pd.Series([False] * len(subset), index=subset.index),
    ).map(strict_bool)

    for col in ["Pass_Periods", "Avg_Edge", "Min_Edge", "Avg_Return", "Total_Trades"]:
        subset[f"_{col}"] = subset[col].map(safe_float)

    subset = subset.sort_values(
        [
            "_strict_candidate",
            "_Pass_Periods",
            "_Min_Edge",
            "_Avg_Return",
            "_Total_Trades",
            "Source_Modified",
        ],
        ascending=[False, False, False, False, False, False],
    )

    return subset.iloc[0]


def select_baseline_row(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series | None:
    if not args.baseline_feature_set:
        return None

    subset = df[
        df["Feature_Set"].astype(str).map(normalize_feature_set_name)
        == normalize_feature_set_name(args.baseline_feature_set)
    ].copy()

    if subset.empty:
        return None

    for col in ["Pass_Periods", "Avg_Edge", "Min_Edge", "Avg_Return", "Total_Trades"]:
        subset[f"_{col}"] = subset[col].map(safe_float)

    subset = subset.sort_values(
        ["_Pass_Periods", "_Min_Edge", "_Avg_Return", "_Total_Trades"],
        ascending=[False, False, False, False],
    )

    return subset.iloc[0]


def build_acceptance(row: pd.Series, args: argparse.Namespace) -> list[dict[str, str]]:
    pass_periods = safe_float(row.get("Pass_Periods"))
    min_edge = safe_float(row.get("Min_Edge"))
    avg_return = safe_float(row.get("Avg_Return"))
    total_trades = safe_float(row.get("Total_Trades"))

    checks = [
        {
            "Rule": f"Pass_Periods >= {args.min_pass_periods}",
            "Observed": format_int(pass_periods),
            "Status": "PASS" if pass_periods >= args.min_pass_periods else "FAIL",
        },
        {
            "Rule": f"Min_Edge >= {format_percent(args.min_edge)}",
            "Observed": format_percent(min_edge),
            "Status": "PASS" if min_edge >= args.min_edge else "FAIL",
        },
        {
            "Rule": f"Avg_Return >= {format_percent(args.min_avg_return)}",
            "Observed": format_percent(avg_return),
            "Status": "PASS" if avg_return >= args.min_avg_return else "FAIL",
        },
        {
            "Rule": f"Total_Trades >= {args.min_trades}",
            "Observed": format_int(total_trades),
            "Status": "PASS" if total_trades >= args.min_trades else "FAIL",
        },
    ]

    return checks


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []

    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")

    return "\n".join([header, separator] + body)


def infer_decision(row: pd.Series, checks: list[dict[str, str]]) -> tuple[str, str]:
    strict_candidate = strict_bool(row.get("Frozen_Strict_Candidate", False))
    all_checks_pass = all(item["Status"] == "PASS" for item in checks)

    if strict_candidate and all_checks_pass:
        return (
            "PROMOTE_TO_PRODUCTION_CANDIDATE",
            "Frozen grid evidence satisfies the acceptance rules under the selected cost assumption.",
        )

    if strict_candidate:
        return (
            "KEEP_AS_RESEARCH_CANDIDATE",
            "Frozen grid marks this as strict candidate, but at least one explicit acceptance rule failed.",
        )

    return (
        "RESEARCH_ONLY",
        "Evidence is not strong enough to promote this setup.",
    )


def build_repro_command(row: pd.Series, args: argparse.Namespace) -> str:
    feature_set = args.alias_name or args.feature_set
    cost = args.round_trip_cost if args.round_trip_cost is not None else "0.003"

    return (
        "python frozen_parameter_grid_runner.py ^\n"
        f"  --ticker={args.ticker} ^\n"
        f"  --profile={args.profile} ^\n"
        f"  --feature-sets={feature_set} ^\n"
        f"  --lookaheads={int(safe_float(row.get('LOOKAHEAD')))} ^\n"
        f"  --tps={safe_float(row.get('TP')):.2f} ^\n"
        f"  --sls={safe_float(row.get('SL')):.2f} ^\n"
        f"  --thresholds={safe_float(row.get('THRESH')):.2f} ^\n"
        f"  --round-trip-cost={cost} ^\n"
        "  --quiet"
    )


def build_report(
    row: pd.Series,
    baseline_row: pd.Series | None,
    checks: list[dict[str, str]],
    decision: str,
    reason: str,
    args: argparse.Namespace,
) -> str:
    candidate_name = args.candidate_name or args.alias_name or args.feature_set
    display_feature_set = args.alias_name or args.feature_set

    evidence_rows = [
        {"Metric": "Feature Set", "Value": display_feature_set},
        {"Metric": "Frozen Grid Source Set", "Value": str(row.get("Feature_Set"))},
        {"Metric": "LOOKAHEAD", "Value": format_int(row.get("LOOKAHEAD"))},
        {"Metric": "TP", "Value": format_percent(row.get("TP"))},
        {"Metric": "SL", "Value": format_percent(row.get("SL"))},
        {"Metric": "THRESH", "Value": format_percent(row.get("THRESH"))},
        {"Metric": "Round-trip Cost", "Value": format_percent(args.round_trip_cost) if args.round_trip_cost is not None else "n/a"},
        {"Metric": "Periods", "Value": format_int(row.get("Periods"))},
        {"Metric": "Pass Periods", "Value": format_int(row.get("Pass_Periods"))},
        {"Metric": "Pass Rate", "Value": format_percent(row.get("Pass_Rate"))},
        {"Metric": "Avg Edge", "Value": format_percent(row.get("Avg_Edge"))},
        {"Metric": "Min Edge", "Value": format_percent(row.get("Min_Edge"))},
        {"Metric": "Avg Return", "Value": format_percent(row.get("Avg_Return"))},
        {"Metric": "Min Return", "Value": format_percent(row.get("Min_Return"))},
        {"Metric": "Total Trades", "Value": format_int(row.get("Total_Trades"))},
        {"Metric": "Frozen Strict Candidate", "Value": str(row.get("Frozen_Strict_Candidate", "n/a"))},
    ]

    lines = [
        f"# {args.ticker.upper()} {args.profile.upper()} Production Candidate Report",
        "",
        f"**Candidate:** `{candidate_name}`",
        "",
        f"**Decision:** `{decision}`",
        "",
        f"**Reason:** {reason}",
        "",
        "## Recommended setup",
        "",
        markdown_table(evidence_rows, ["Metric", "Value"]),
        "",
        "## Acceptance checks",
        "",
        markdown_table(checks, ["Rule", "Observed", "Status"]),
        "",
        "## Research interpretation",
        "",
    ]

    feature_set_text = str(row.get("Feature_Set", ""))

    if "minus_noise_filter" in feature_set_text or "no_noise" in display_feature_set:
        lines.extend(
            [
                "The strongest frozen-grid setup excludes the noise-filter feature family. "
                "This suggests the production candidate should retain liquidity and recovery-quality features, "
                "while keeping noise-filter features only for research and ablation tests.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The selected setup should be treated as the current best frozen-grid candidate for this ticker/profile.",
                "",
            ]
        )

    if baseline_row is not None:
        comparison_rows = [
            {
                "Metric": "Min Edge",
                "Candidate": format_percent(row.get("Min_Edge")),
                "Baseline": format_percent(baseline_row.get("Min_Edge")),
                "Delta": format_percent(safe_float(row.get("Min_Edge")) - safe_float(baseline_row.get("Min_Edge"))),
            },
            {
                "Metric": "Avg Return",
                "Candidate": format_percent(row.get("Avg_Return")),
                "Baseline": format_percent(baseline_row.get("Avg_Return")),
                "Delta": format_percent(safe_float(row.get("Avg_Return")) - safe_float(baseline_row.get("Avg_Return"))),
            },
            {
                "Metric": "Total Trades",
                "Candidate": format_int(row.get("Total_Trades")),
                "Baseline": format_int(baseline_row.get("Total_Trades")),
                "Delta": format_int(safe_float(row.get("Total_Trades")) - safe_float(baseline_row.get("Total_Trades"))),
            },
        ]

        lines.extend(
            [
                "## Baseline comparison",
                "",
                f"Baseline feature set: `{args.baseline_feature_set}`",
                "",
                markdown_table(
                    comparison_rows,
                    ["Metric", "Candidate", "Baseline", "Delta"],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Reproducibility command",
            "",
            "```cmd",
            build_repro_command(row, args),
            "```",
            "",
            "## Source",
            "",
            f"- Source CSV: `{row.get('Source_File')}`",
            f"- Source modified: `{row.get('Source_Modified')}`",
            f"- Report generated: `{datetime.now().isoformat(timespec='seconds')}`",
            "",
        ]
    )

    return "\n".join(lines)


def write_outputs(
    row: pd.Series,
    baseline_row: pd.Series | None,
    checks: list[dict[str, str]],
    decision: str,
    reason: str,
    report: str,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    root = Path.cwd()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_name = args.candidate_name or args.alias_name or args.feature_set
    safe_name = candidate_name.replace("/", "_").replace("\\", "_")
    stem = f"{args.ticker.upper()}_{args.profile.upper()}_{safe_name}"

    md_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"

    md_path.write_text(report, encoding="utf-8")

    payload = {
        "ticker": args.ticker.upper(),
        "profile": args.profile.upper(),
        "candidate_name": candidate_name,
        "display_feature_set": args.alias_name or args.feature_set,
        "source_feature_set": row.get("Feature_Set"),
        "decision": decision,
        "reason": reason,
        "setup": {
            "lookahead": safe_float(row.get("LOOKAHEAD")),
            "tp": safe_float(row.get("TP")),
            "sl": safe_float(row.get("SL")),
            "threshold": safe_float(row.get("THRESH")),
            "round_trip_cost": args.round_trip_cost,
        },
        "metrics": {
            "periods": safe_float(row.get("Periods")),
            "pass_periods": safe_float(row.get("Pass_Periods")),
            "pass_rate": safe_float(row.get("Pass_Rate")),
            "avg_edge": safe_float(row.get("Avg_Edge")),
            "min_edge": safe_float(row.get("Min_Edge")),
            "avg_return": safe_float(row.get("Avg_Return")),
            "min_return": safe_float(row.get("Min_Return")),
            "total_trades": safe_float(row.get("Total_Trades")),
            "frozen_strict_candidate": strict_bool(row.get("Frozen_Strict_Candidate", False)),
        },
        "acceptance_checks": checks,
        "source": {
            "csv": row.get("Source_File"),
            "modified": row.get("Source_Modified"),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if baseline_row is not None:
        payload["baseline"] = {
            "feature_set": args.baseline_feature_set,
            "metrics": {
                "min_edge": safe_float(baseline_row.get("Min_Edge")),
                "avg_return": safe_float(baseline_row.get("Avg_Return")),
                "total_trades": safe_float(baseline_row.get("Total_Trades")),
            },
        }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return md_path, json_path


def main() -> None:
    args = parse_args()
    root = Path.cwd()

    df = load_grid_data(root, args.input)
    row = select_candidate_row(df, args)
    baseline_row = select_baseline_row(df, args)
    checks = build_acceptance(row, args)
    decision, reason = infer_decision(row, checks)
    report = build_report(row, baseline_row, checks, decision, reason, args)
    md_path, json_path = write_outputs(row, baseline_row, checks, decision, reason, report, args)

    print("=" * 100)
    print("Production candidate report generated")
    print("=" * 100)
    print(f"Decision: {decision}")
    print(f"Markdown: {md_path}")
    print(f"JSON:     {json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
