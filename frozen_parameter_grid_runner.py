from __future__ import annotations

import argparse
import itertools
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


def parse_csv_list(value: str | None, cast_type=str) -> list:
    if not value:
        return []
    return [cast_type(item.strip()) for item in value.split(",") if item.strip()]


def parse_periods(value: str | None) -> list[tuple[str, str, str]]:
    if not value:
        return DEFAULT_PERIODS

    periods = []

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


def run_fixed_experiment(
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
    quiet: bool,
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

    if not quiet:
        print("")
        print("=" * 120)
        print("Running:", " ".join(cmd))
        print("=" * 120)

    subprocess.run(
        cmd,
        cwd=project_root,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )


def read_fixed_result(
    project_root: Path,
    ticker: str,
    profile: str,
    feature_set: str,
    period_label: str,
    period_start: str,
    period_end: str,
    lookahead: int,
    tp: float,
    sl: float,
    threshold: float,
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
            f"No fixed_setup row found for feature_set={feature_set}."
        )

    out.insert(0, "Period_Label", period_label)
    out.insert(1, "Period_Start", period_start)
    out.insert(2, "Period_End", period_end)
    out.insert(3, "Frozen_Feature_Set", feature_set)
    out.insert(4, "Frozen_LOOKAHEAD", lookahead)
    out.insert(5, "Frozen_TP", tp)
    out.insert(6, "Frozen_SL", sl)
    out.insert(7, "Frozen_THRESH", threshold)

    out["Setup_ID"] = (
        out["Frozen_Feature_Set"].astype(str)
        + "|L" + out["Frozen_LOOKAHEAD"].astype(str)
        + "|TP" + out["Frozen_TP"].astype(str)
        + "|SL" + out["Frozen_SL"].astype(str)
        + "|TH" + out["Frozen_THRESH"].astype(str)
    )

    return out


def build_period_summary(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Period_Label",
        "Feature_Set",
        "LOOKAHEAD",
        "TP",
        "SL",
        "THRESH",
        "Score",
        "Validation_Precision",
        "Breakeven_Precision",
        "Edge_vs_Breakeven",
        "Avg_Return",
        "Avg_Holding_Days",
        "Expectancy_Per_Day",
        "Timeout_Rate",
        "Trades",
        "Research_Status",
        "Pass_Tradeable_Rule",
        "Setup_ID",
    ]

    cols = [col for col in cols if col in raw.columns]
    out = raw[cols].copy()
    out = out.sort_values(["Period_Label", "Score"], ascending=[True, False])
    return out


def build_stability_score(raw: pd.DataFrame, min_edge: float) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    grouped = raw.groupby(
        ["Setup_ID", "Feature_Set", "LOOKAHEAD", "TP", "SL", "THRESH"],
        dropna=False,
    )

    summary = grouped.agg(
        Periods=("Period_Label", "nunique"),
        Pass_Periods=(
            "Pass_Tradeable_Rule",
            lambda x: int(pd.Series(x).astype(str).str.lower().isin(["true", "1"]).sum()),
        ),
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
    ).reset_index()

    summary["Pass_Rate"] = summary["Pass_Periods"] / summary["Periods"]

    summary["Frozen_Strict_Candidate"] = (
        (summary["Pass_Rate"] >= 0.75)
        & (summary["Min_Edge"] >= min_edge)
        & (summary["Avg_Return"] > 0)
        & (summary["Total_Trades"] >= 100)
    )

    summary = summary.sort_values(
        [
            "Frozen_Strict_Candidate",
            "Pass_Rate",
            "Min_Edge",
            "Avg_Edge",
            "Avg_Score",
            "Total_Trades",
        ],
        ascending=[False, False, False, False, False, False],
    )

    return summary


def format_pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.2f}%"


def print_top(stability: pd.DataFrame, top: int) -> None:
    print("")
    print("=" * 120)
    print("FROZEN PARAMETER GRID RANKING")
    print("=" * 120)

    if stability.empty:
        print("(empty)")
        return

    display = stability.head(top).copy()

    for col in ["Pass_Rate", "Avg_Edge", "Min_Edge", "Avg_Return", "Min_Return"]:
        if col in display.columns:
            display[col] = display[col].apply(format_pct)

    cols = [
        "Feature_Set",
        "LOOKAHEAD",
        "TP",
        "SL",
        "THRESH",
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

    cols = [col for col in cols if col in display.columns]
    print(display[cols].to_string(index=False))


def run_grid(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parent

    ticker = args.ticker.upper()
    profile = args.profile.upper()

    processed_dir = project_root / "data" / "processed_data"
    output_dir = project_root / "outputs" / "walkforward_frozen_grid"
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_path = processed_dir / f"{ticker}_standardized.csv"
    index_path = processed_dir / "VNINDEX_standardized.csv"
    valuation_path = processed_dir / "valuation" / f"{ticker}_valuation_standardized.csv"

    paths_to_backup = [stock_path, index_path]

    if valuation_path.exists():
        paths_to_backup.append(valuation_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / ".tmp_frozen_grid_backups" / f"{ticker}_{profile}_{timestamp}"
    backup_map = backup_files(paths_to_backup, backup_dir)

    original_stock = load_csv_with_date(stock_path)
    original_index = load_csv_with_date(index_path)
    original_valuation = load_csv_with_date(valuation_path) if valuation_path.exists() else None

    periods = parse_periods(args.periods)

    feature_sets = parse_csv_list(args.feature_sets, str) or DEFAULT_FEATURE_SETS
    lookaheads = parse_csv_list(args.lookaheads, int)
    tps = parse_csv_list(args.tps, float)
    sls = parse_csv_list(args.sls, float)
    thresholds = parse_csv_list(args.thresholds, float)

    combos = list(itertools.product(feature_sets, lookaheads, tps, sls, thresholds))

    print("=" * 120)
    print(f"Frozen parameter grid started | ticker={ticker} | profile={profile}")
    print(f"Feature sets: {feature_sets}")
    print(f"Total setups: {len(combos)} | periods={len(periods)} | model runs={len(combos) * len(periods)}")
    print("=" * 120)

    all_rows = []

    try:
        for period_label, period_start, period_end in periods:
            print("")
            print("#" * 120)
            print(f"PERIOD: {period_label} | {period_start} to {period_end}")
            print("#" * 120)

            stock_period = filter_by_period(original_stock, period_start, period_end)
            index_period = filter_by_period(original_index, period_start, period_end)

            if len(stock_period) < args.min_rows:
                print(f"Skipping {period_label}: not enough stock rows.")
                continue

            if len(index_period) < args.min_rows:
                print(f"Skipping {period_label}: not enough index rows.")
                continue

            save_csv(stock_period, stock_path)
            save_csv(index_period, index_path)

            if original_valuation is not None:
                valuation_period = filter_by_period(original_valuation, period_start, period_end)
                save_csv(valuation_period, valuation_path)

            for idx, (feature_set, lookahead, tp, sl, threshold) in enumerate(combos, start=1):
                print(
                    f"[{period_label}] {idx}/{len(combos)} | "
                    f"{feature_set} | L={lookahead} TP={tp} SL={sl} TH={threshold}"
                )

                run_fixed_experiment(
                    project_root=project_root,
                    ticker=ticker,
                    profile=profile,
                    feature_set=feature_set,
                    lookahead=lookahead,
                    tp=tp,
                    sl=sl,
                    threshold=threshold,
                    min_edge=args.min_edge_vs_breakeven,
                    round_trip_cost=args.round_trip_cost,
                    quiet=args.quiet,
                )

                rows = read_fixed_result(
                    project_root=project_root,
                    ticker=ticker,
                    profile=profile,
                    feature_set=feature_set,
                    period_label=period_label,
                    period_start=period_start,
                    period_end=period_end,
                    lookahead=lookahead,
                    tp=tp,
                    sl=sl,
                    threshold=threshold,
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
        raise RuntimeError("No frozen grid rows were collected.")

    raw = pd.concat(all_rows, ignore_index=True)
    period_summary = build_period_summary(raw)
    stability = build_stability_score(raw, min_edge=args.min_edge_vs_breakeven)

    raw_path = output_dir / f"{ticker}_{profile}_frozen_grid_raw.csv"
    period_path = output_dir / f"{ticker}_{profile}_frozen_grid_period_summary.csv"
    stability_path = output_dir / f"{ticker}_{profile}_frozen_grid_stability_score.csv"

    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    period_summary.to_csv(period_path, index=False, encoding="utf-8-sig")
    stability.to_csv(stability_path, index=False, encoding="utf-8-sig")

    print("")
    print("=" * 120)
    print("Frozen parameter grid outputs saved")
    print("=" * 120)
    print(f"Raw:       {raw_path}")
    print(f"Periods:   {period_path}")
    print(f"Stability: {stability_path}")

    print_top(stability, top=args.top)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for a single frozen setup that is stable across periods."
    )

    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--profile", type=str, default="SWING")

    parser.add_argument(
        "--feature-sets",
        type=str,
        default=",".join(DEFAULT_FEATURE_SETS),
        help="Comma-separated feature sets.",
    )

    parser.add_argument("--lookaheads", type=str, default="20,30,40")
    parser.add_argument("--tps", type=str, default="0.08,0.10")
    parser.add_argument("--sls", type=str, default="-0.05,-0.07")
    parser.add_argument("--thresholds", type=str, default="0.55,0.60,0.65,0.70")

    parser.add_argument(
        "--periods",
        type=str,
        default=None,
        help=(
            "Custom periods. Format: Label:YYYY-MM-DD:YYYY-MM-DD,"
            "Label2:YYYY-MM-DD:YYYY-MM-DD"
        ),
    )

    parser.add_argument("--min-edge-vs-breakeven", type=float, default=0.02)
    parser.add_argument("--round-trip-cost", type=float, default=0.0)
    parser.add_argument("--min-rows", type=int, default=250)
    parser.add_argument("--top", type=int, default=20)

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide the full feature_experiment_runner logs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    run_grid(parsed_args)
