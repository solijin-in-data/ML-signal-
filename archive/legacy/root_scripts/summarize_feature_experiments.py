from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/feature_experiments")


def pct(x: float | int | str | None, digits: int = 2) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return ""


def num(x: float | int | str | None, digits: int = 4) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{digits}f}"
    except Exception:
        return ""


def setup_label(row: pd.Series) -> str:
    tp = pct(row.get("TP"), 0)
    sl = pct(row.get("SL"), 0)
    threshold = num(row.get("THRESH"), 2)
    lookahead = row.get("LOOKAHEAD", "")
    return f"{lookahead}D | TP {tp} | SL {sl} | TH {threshold}"


def short_mode(value: str) -> str:
    value = str(value)
    if value == "reoptimized":
        return "reopt"
    if value == "fixed_setup":
        return "fixed"
    return value


def load_reports(output_dir: Path) -> pd.DataFrame:
    comparison_path = output_dir / "feature_set_comparison.csv"
    decision_path = output_dir / "feature_set_decision_report.csv"

    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing file: {comparison_path}")

    comparison = pd.read_csv(comparison_path, encoding="utf-8-sig")

    if decision_path.exists():
        decision = pd.read_csv(decision_path, encoding="utf-8-sig")
        join_cols = ["Ticker", "Profile", "Experiment_Mode", "Feature_Set"]
        available_join_cols = [col for col in join_cols if col in comparison.columns and col in decision.columns]

        if available_join_cols:
            keep_cols = available_join_cols + [
                col for col in ["Decision", "Reason"] if col in decision.columns
            ]
            decision = decision[keep_cols].drop_duplicates(subset=available_join_cols)
            comparison = comparison.merge(
                decision,
                on=available_join_cols,
                how="left",
                suffixes=("", "_DecisionReport"),
            )

    return comparison


def filter_report(
    df: pd.DataFrame,
    ticker: str | None,
    profile: str | None,
    mode: str,
    only_tradeable: bool,
) -> pd.DataFrame:
    out = df.copy()

    if ticker:
        out = out[out["Ticker"].astype(str).str.upper().eq(ticker.upper())]

    if profile:
        out = out[out["Profile"].astype(str).str.upper().eq(profile.upper())]

    if mode != "all":
        if mode == "reopt":
            out = out[out["Experiment_Mode"].astype(str).eq("reoptimized")]
        elif mode == "fixed":
            out = out[out["Experiment_Mode"].astype(str).eq("fixed_setup")]

    if only_tradeable and "Pass_Tradeable_Rule" in out.columns:
        out = out[out["Pass_Tradeable_Rule"].astype(str).str.lower().isin(["true", "1"])]

    return out.copy()


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Mode"] = out["Experiment_Mode"].map(short_mode)
    out["Setup"] = out.apply(setup_label, axis=1)
    out["Precision"] = out["Validation_Precision"].apply(pct)
    out["BE"] = out["Breakeven_Precision"].apply(pct)
    out["Edge"] = out["Edge_vs_Breakeven"].apply(pct)
    out["AvgRet"] = out["Avg_Return"].apply(pct)
    out["Exp/Day"] = out["Expectancy_Per_Day"].apply(pct)
    out["Timeout"] = out["Timeout_Rate"].apply(pct)
    out["ScoreText"] = out["Score"].apply(lambda x: num(x, 4))
    out["TradesText"] = out["Trades"].apply(lambda x: "" if pd.isna(x) else str(int(x)))

    if "Decision" not in out.columns:
        out["Decision"] = ""

    out["Decision"] = out["Decision"].fillna("")
    out["Research_Status"] = out["Research_Status"].fillna("")

    return out


def compact_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "(empty)"

    display = df[columns].copy()
    return display.to_string(index=False)


def build_summary_text(df: pd.DataFrame, top: int, sort_by: str) -> str:
    if df.empty:
        return "No rows matched the selected filters."

    sort_map = {
        "score": "Score",
        "edge": "Edge_vs_Breakeven",
        "return": "Avg_Return",
        "expectancy": "Expectancy_Per_Day",
        "trades": "Trades",
    }

    sort_col = sort_map.get(sort_by, "Score")

    df = df.sort_values(sort_col, ascending=False).head(top).copy()
    df = add_display_columns(df)

    main_cols = [
        "Feature_Set",
        "Mode",
        "Setup",
        "ScoreText",
        "Edge",
        "AvgRet",
        "Exp/Day",
        "Timeout",
        "TradesText",
        "Research_Status",
        "Decision",
    ]

    risk_cols = [
        "Feature_Set",
        "Mode",
        "Precision",
        "BE",
        "Edge",
        "Avg_Holding_Days",
        "Timeout",
        "TradesText",
    ]

    # Rename for cleaner console output.
    main = df[main_cols].rename(
        columns={
            "Feature_Set": "Feature",
            "ScoreText": "Score",
            "TradesText": "Trades",
            "Research_Status": "Status",
        }
    )

    risk = df[risk_cols].rename(
        columns={
            "Feature_Set": "Feature",
            "Avg_Holding_Days": "HoldD",
            "TradesText": "Trades",
        }
    )

    if "HoldD" in risk.columns:
        risk["HoldD"] = risk["HoldD"].apply(lambda x: num(x, 2))

    lines = []
    lines.append("=" * 120)
    lines.append(f"TOP FEATURE SETS | sorted by {sort_by}")
    lines.append("=" * 120)
    lines.append(main.to_string(index=False))
    lines.append("")
    lines.append("=" * 120)
    lines.append("RISK / HIT-RATE VIEW")
    lines.append("=" * 120)
    lines.append(risk.to_string(index=False))

    return "\n".join(lines)


def save_markdown_summary(text: str, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    content = "# Feature Experiment Console Summary\n\n"
    content += "```text\n"
    content += text
    content += "\n```\n"

    path.write_text(content, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print compact feature experiment summaries from CSV outputs."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder containing feature_set_comparison.csv and feature_set_decision_report.csv.",
    )

    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)

    parser.add_argument(
        "--mode",
        type=str,
        default="reopt",
        choices=["reopt", "fixed", "all"],
        help="Which experiment mode to display. Default: reopt.",
    )

    parser.add_argument(
        "--sort-by",
        type=str,
        default="score",
        choices=["score", "edge", "return", "expectancy", "trades"],
        help="Sort key for ranking. Default: score.",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of rows to show. Default: 10.",
    )

    parser.add_argument(
        "--only-tradeable",
        action="store_true",
        help="Show only rows where Pass_Tradeable_Rule is true.",
    )

    parser.add_argument(
        "--save-md",
        action="store_true",
        help="Save the compact summary to markdown.",
    )

    parser.add_argument(
        "--md-name",
        type=str,
        default="feature_experiment_console_summary.md",
        help="Markdown output filename.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)

    report = load_reports(output_dir)

    report = filter_report(
        df=report,
        ticker=args.ticker,
        profile=args.profile,
        mode=args.mode,
        only_tradeable=args.only_tradeable,
    )

    summary_text = build_summary_text(
        df=report,
        top=args.top,
        sort_by=args.sort_by,
    )

    print(summary_text)

    if args.save_md:
        md_path = save_markdown_summary(
            text=summary_text,
            output_dir=output_dir,
            filename=args.md_name,
        )
        print("")
        print(f"Saved markdown summary: {md_path}")
