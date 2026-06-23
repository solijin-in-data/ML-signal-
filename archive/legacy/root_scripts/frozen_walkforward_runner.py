from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_PERIODS = [
    ("P1_2014_2018", "2014-01-01", "2018-12-31"),
    ("P2_2019_2021", "2019-01-01", "2021-12-31"),
    ("P3_2022_2023", "2022-01-01", "2023-12-31"),
    ("P4_2024_2026", "2024-01-01", "2026-12-31"),
]


DEFAULT_FEATURE_SETS = [
    "ablation_trend_recovery_minus_market_regime_v1",
]


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []

    return [item.strip() for item in value.split(",") if item.strip()]


def parse_periods(value: str | None) -> list[tuple[str, str, str]]:
    if not value:
        return DEFAULT_PERIODS

    periods: list[tuple[str, str, str]] = []

    for chunk in value.split(","):
        parts = [part.strip() for part in chunk.split(":")]

        if len(parts) != 3:
            raise ValueError(
                "Invalid period format. Use Label:YYYY-MM-DD:YYYY-MM-DD separated by commas."
            )

        periods.append((parts[0], parts[1], parts[2]))

    return periods


def load_csv_with_date(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()

    if "Date" not in df.columns:
        raise ValueError(f"File must contain Date column: {path}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")

    return df


def filter_by_period(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    return df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].copy()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def backup_files(paths: list[Path], backup_dir: Path) -> dict[Path, Path]:
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_map: dict[Path, Path] = {}

    for path in paths:
        if path.exists():
            backup_path = backup_dir / path.name
            shutil.copy2(path, backup_path)
            backup_map[path] = backup_path

    return backup_map


def restore_files(backup_map: dict[Path, Path]) -> None:
    for original_path, backup_path in backup_map.items():
        shutil.copy2(backup_path, original_path)


def run_fixed_feature_experiment(
    project_root: Path,
    ticker: str,
    profile: str,
    feature_set: str,
    lookahead: int,
    tp: float,
    sl: float,
    threshold: float,
    min_edge: float,
    round_trip_cost: float,
) -> None:
    cmd = [
        sys.executable,
        "feature_experiment_runner.py",
        "--ticker",
        ticker,
        "--profile",
        profile,
        "--mode",
        "fixed",
        "--feature-set",
        feature_set,
        "--lookahead",
        str(lookahead),
        "--tp",
        str(tp),
        "--sl",
        str(sl),
        "--threshold",
        str(threshold),
        "--min-edge-vs-breakeven",
        str(min_edge),
        "--round-trip-cost",
        str(round_trip_cost),
        "--no-diagnostics-fallback",
    ]

    print("")
    print("=" * 120)
    print("Running frozen setup:", " ".join(cmd))
    print("=" * 120)

    subprocess.run(
        cmd,
        cwd=project_root,
        check=True,
    )


def read_latest_comparison(
    project_root: Path,
    ticker: str,
    profile: str,
    feature_set: str,
    period_label: str,
    period_start: str,
    period_end: str,
) -> pd.DataFrame:
    comparison_path = project_root / "outputs" / "feature_experiments" / "feature_set_comparison.csv"

    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing comparison output: {comparison_path}")

    df = pd.read_csv(comparison_path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()

    mask = (
        df["Ticker"].astype(str).str.upper().eq(ticker.upper())
        & df["Profile"].astype(str).str.upper().eq(profile.upper())
        & df["Feature_Set"].astype(str).eq(feature_set)
        & df["Experiment_Mode"].astype(str).eq("fixed_setup")
    )

    out = df[mask].copy()

    if out.empty:
        raise ValueError(
            f"No fixed_setup row found for period={period_label}, feature_set={feature_set}."
        )

    out.insert(0, "Period_Label", period_label)
    out.insert(1, "Period_Start", period_start)
    out.insert(2, "Period_End", period_end)

    return out


def add_period_baseline_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    metric_cols = [
        "Score",
        "Validation_Precision",
        "Excess_Precision",
        "Edge_vs_Breakeven",
        "Avg_Return",
        "Expectancy_Per_Day",
        "Timeout_Rate",
        "Trades",
    ]

    key_cols = ["Period_Label"]

    baseline = out[out["Feature_Set"].eq("baseline")].copy()

    if baseline.empty:
        for metric in metric_cols:
            out[f"{metric}_Delta_vs_Period_Baseline"] = pd.NA
        return out

    baseline = baseline[key_cols + metric_cols].drop_duplicates(subset=key_cols)
    baseline = baseline.rename(
        columns={metric: f"{metric}_Period_Baseline" for metric in metric_cols}
    )

    out = out.merge(baseline, on=key_cols, how="left")

    for metric in metric_cols:
        if metric in out.columns:
            out[f"{metric}_Delta_vs_Period_Baseline"] = (
                out[metric] - out[f"{metric}_Period_Baseline"]
            )

    drop_cols = [f"{metric}_Period_Baseline" for metric in metric_cols]
    out = out.drop(columns=[col for col in drop_cols if col in out.columns])

    return out


def build_stability_score(df: pd.DataFrame, min_edge: float) -> pd.DataFrame:
    candidates = df[~df["Feature_Set"].eq("baseline")].copy()

    if candidates.empty:
        return pd.DataFrame()

    def count_pass(series: pd.Series) -> int:
        return int(series.astype(str).str.lower().isin(["true", "1"]).sum())

    grouped = candidates.groupby("Feature_Set", dropna=False)

    summary = grouped.agg(
        Periods=("Period_Label", "nunique"),
        Pass_Periods=("Pass_Tradeable_Rule", count_pass),
        Avg_Score=("Score", "mean"),
        Min_Score=("Score", "min"),
        Avg_Edge=("Edge_vs_Breakeven", "mean"),
        Min_Edge=("Edge_vs_Breakeven", "min"),
        Avg_Return=("Avg_Return", "mean"),
        Min_Return=("Avg_Return", "min"),
        Avg_Expectancy_Per_Day=("Expectancy_Per_Day", "mean"),
        Avg_Timeout_Rate=("Timeout_Rate", "mean"),
        Total_Trades=("Trades", "sum"),
        Avg_Trades=("Trades", "mean"),
        Avg_Score_Delta=("Score_Delta_vs_Period_Baseline", "mean"),
        Avg_Edge_Delta=("Edge_vs_Breakeven_Delta_vs_Period_Baseline", "mean"),
        Avg_Return_Delta=("Avg_Return_Delta_vs_Period_Baseline", "mean"),
    ).reset_index()

    summary["Pass_Rate"] = summary["Pass_Periods"] / summary["Periods"]

    summary["Frozen_Strict_Candidate"] = (
        (summary["Pass_Rate"] >= 0.75)
        & (summary["Min_Edge"] >= min_edge)
        & (summary["Avg_Return"] > 0)
        & (summary["Total_Trades"] >= 100)
    )

    summary = summary.sort_values(
        ["Frozen_Strict_Candidate", "Pass_Rate", "Min_Edge", "Avg_Edge", "Avg_Score"],
        ascending=[False, False, False, False, False],
    )

    return summary


def build_period_summary(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "Period_Label",
        "Period_Start",
        "Period_End",
        "Feature_Set",
        "LOOKAHEAD",
        "TP",
        "SL",
        "THRESH",
        "Score",
        "Validation_Precision",
        "Base_Rate",
        "Breakeven_Precision",
        "Edge_vs_Breakeven",
        "Avg_Return",
        "Avg_Holding_Days",
        "Expectancy_Per_Day",
        "Timeout_Rate",
        "Trades",
        "Research_Status",
        "Pass_Tradeable_Rule",
        "Score_Delta_vs_Period_Baseline",
        "Edge_vs_Breakeven_Delta_vs_Period_Baseline",
        "Avg_Return_Delta_vs_Period_Baseline",
    ]

    keep_cols = [col for col in keep_cols if col in df.columns]

    out = df[keep_cols].copy()
    out = out.sort_values(["Period_Label", "Feature_Set"])

    return out


def format_pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.{digits}f}%"


def print_stability(stability: pd.DataFrame) -> None:
    print("")
    print("=" * 120)
    print("FROZEN STABILITY RANKING")
    print("=" * 120)

    if stability.empty:
        print("(empty)")
        return

    display = stability.copy()

    for col in [
        "Pass_Rate",
        "Avg_Edge",
        "Min_Edge",
        "Avg_Return",
        "Min_Return",
        "Avg_Expectancy_Per_Day",
        "Avg_Timeout_Rate",
        "Avg_Edge_Delta",
        "Avg_Return_Delta",
    ]:
        if col in display.columns:
            display[col] = display[col].apply(format_pct)

    display_cols = [
        "Feature_Set",
        "Periods",
        "Pass_Periods",
        "Pass_Rate",
        "Avg_Score",
        "Min_Score",
        "Avg_Edge",
        "Min_Edge",
        "Avg_Return",
        "Min_Return",
        "Total_Trades",
        "Frozen_Strict_Candidate",
    ]

    display_cols = [col for col in display_cols if col in display.columns]

    print(display[display_cols].to_string(index=False))


def write_text_summary(path: Path, stability: pd.DataFrame, period_summary: pd.DataFrame) -> None:
    lines = []

    lines.append("# Frozen Walk-forward Summary")
    lines.append("")
    lines.append("## Stability Ranking")
    lines.append("")
    lines.append("```text")
    lines.append(stability.to_string(index=False) if not stability.empty else "(empty)")
    lines.append("```")
    lines.append("")
    lines.append("## Period Results")
    lines.append("")
    lines.append("```text")
    lines.append(period_summary.to_string(index=False) if not period_summary.empty else "(empty)")
    lines.append("```")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_frozen_walkforward(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parent

    ticker = args.ticker.upper()
    profile = args.profile.upper()

    processed_dir = project_root / "data" / "processed_data"
    output_dir = project_root / "outputs" / "walkforward_frozen"
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_path = processed_dir / f"{ticker}_standardized.csv"
    index_path = processed_dir / "VNINDEX_standardized.csv"
    valuation_path = processed_dir / "valuation" / f"{ticker}_valuation_standardized.csv"

    paths_to_backup = [stock_path, index_path]

    if valuation_path.exists():
        paths_to_backup.append(valuation_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / ".tmp_frozen_walkforward_backups" / f"{ticker}_{profile}_{timestamp}"
    backup_map = backup_files(paths_to_backup, backup_dir)

    original_stock = load_csv_with_date(stock_path)
    original_index = load_csv_with_date(index_path)
    original_valuation = load_csv_with_date(valuation_path) if valuation_path.exists() else None

    periods = parse_periods(args.periods)
    feature_sets = parse_csv_list(args.feature_sets) or DEFAULT_FEATURE_SETS.copy()

    if "baseline" not in feature_sets:
        feature_sets = ["baseline"] + feature_sets

    all_rows = []

    try:
        for period_label, period_start, period_end in periods:
            print("")
            print("#" * 120)
            print(f"FROZEN PERIOD: {period_label} | {period_start} to {period_end}")
            print("#" * 120)

            stock_period = filter_by_period(original_stock, period_start, period_end)
            index_period = filter_by_period(original_index, period_start, period_end)

            if len(stock_period) < args.min_rows:
                print(f"Skipping period={period_label}. Stock rows {len(stock_period)} < {args.min_rows}.")
                continue

            if len(index_period) < args.min_rows:
                print(f"Skipping period={period_label}. Index rows {len(index_period)} < {args.min_rows}.")
                continue

            save_csv(stock_period, stock_path)
            save_csv(index_period, index_path)

            if original_valuation is not None:
                valuation_period = filter_by_period(original_valuation, period_start, period_end)
                save_csv(valuation_period, valuation_path)

            for feature_set in feature_sets:
                run_fixed_feature_experiment(
                    project_root=project_root,
                    ticker=ticker,
                    profile=profile,
                    feature_set=feature_set,
                    lookahead=args.lookahead,
                    tp=args.tp,
                    sl=args.sl,
                    threshold=args.threshold,
                    min_edge=args.min_edge_vs_breakeven,
                    round_trip_cost=args.round_trip_cost,
                )

                rows = read_latest_comparison(
                    project_root=project_root,
                    ticker=ticker,
                    profile=profile,
                    feature_set=feature_set,
                    period_label=period_label,
                    period_start=period_start,
                    period_end=period_end,
                )

                all_rows.append(rows)

    finally:
        restore_files(backup_map)
        print("")
        print("=" * 120)
        print("Original data files restored.")
        print(f"Backup folder: {backup_dir}")
        print("=" * 120)

    if not all_rows:
        raise RuntimeError("No frozen walk-forward rows were collected.")

    combined = pd.concat(all_rows, ignore_index=True)
    combined = add_period_baseline_deltas(combined)

    period_summary = build_period_summary(combined)
    stability = build_stability_score(combined, min_edge=args.min_edge_vs_breakeven)

    raw_path = output_dir / f"{ticker}_{profile}_frozen_walkforward_raw.csv"
    period_path = output_dir / f"{ticker}_{profile}_frozen_walkforward_period_summary.csv"
    stability_path = output_dir / f"{ticker}_{profile}_frozen_walkforward_stability_score.csv"
    summary_path = output_dir / f"{ticker}_{profile}_frozen_walkforward_summary.md"

    combined.to_csv(raw_path, index=False, encoding="utf-8-sig")
    period_summary.to_csv(period_path, index=False, encoding="utf-8-sig")
    stability.to_csv(stability_path, index=False, encoding="utf-8-sig")
    write_text_summary(summary_path, stability, period_summary)

    print("")
    print("=" * 120)
    print("Frozen walk-forward outputs saved")
    print("=" * 120)
    print(f"Raw:        {raw_path}")
    print(f"Periods:    {period_path}")
    print(f"Stability:  {stability_path}")
    print(f"Markdown:   {summary_path}")

    print_stability(stability)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen-parameter walk-forward test for feature sets."
    )

    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--profile", type=str, default="SWING")

    parser.add_argument(
        "--feature-sets",
        type=str,
        default=",".join(DEFAULT_FEATURE_SETS),
        help="Comma-separated feature sets. Baseline is added automatically.",
    )

    parser.add_argument("--lookahead", type=int, default=20)
    parser.add_argument("--tp", type=float, default=0.10)
    parser.add_argument("--sl", type=float, default=-0.05)
    parser.add_argument("--threshold", type=float, default=0.60)

    parser.add_argument(
        "--min-edge-vs-breakeven",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--round-trip-cost",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--periods",
        type=str,
        default=None,
        help=(
            "Custom periods. Format: Label:YYYY-MM-DD:YYYY-MM-DD,"
            "Label2:YYYY-MM-DD:YYYY-MM-DD"
        ),
    )

    parser.add_argument(
        "--min-rows",
        type=int,
        default=250,
        help="Minimum rows required for stock and index data in each period.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_frozen_walkforward(parse_args())
