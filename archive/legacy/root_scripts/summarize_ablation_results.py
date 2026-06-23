from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/feature_experiments")


FAMILY_CONFIG = {
    "trend_recovery": {
        "full": "ablation_trend_recovery_full_v1",
        "fallback_full": "candidate_trend_recovery_v1",
        "members": [
            "ablation_trend_recovery_full_v1",
            "ablation_trend_recovery_minus_trend_v1",
            "ablation_trend_recovery_minus_recovery_v1",
            "ablation_trend_recovery_minus_rsi_v1",
            "ablation_trend_recovery_minus_drawdown_recovery_v1",
            "ablation_trend_recovery_minus_market_regime_v1",
            "candidate_trend_recovery_v1",
        ],
    },
    "recovery_valuation": {
        "full": "ablation_recovery_valuation_full_v1",
        "fallback_full": "candidate_recovery_valuation_v1",
        "members": [
            "ablation_recovery_valuation_full_v1",
            "ablation_recovery_valuation_minus_recovery_v1",
            "ablation_recovery_valuation_minus_valuation_v1",
            "candidate_recovery_valuation_v1",
        ],
    },
}


def pct(x, digits: int = 2) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return ""


def num(x, digits: int = 4) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{digits}f}"
    except Exception:
        return ""


def short_name(feature_set: str) -> str:
    replacements = {
        "ablation_trend_recovery_full_v1": "full",
        "candidate_trend_recovery_v1": "candidate_full",
        "ablation_trend_recovery_minus_trend_v1": "minus_trend",
        "ablation_trend_recovery_minus_recovery_v1": "minus_recovery",
        "ablation_trend_recovery_minus_rsi_v1": "minus_rsi",
        "ablation_trend_recovery_minus_drawdown_recovery_v1": "minus_drawdown_recovery",
        "ablation_trend_recovery_minus_market_regime_v1": "minus_market_regime",
        "ablation_recovery_valuation_full_v1": "full",
        "candidate_recovery_valuation_v1": "candidate_full",
        "ablation_recovery_valuation_minus_recovery_v1": "minus_recovery",
        "ablation_recovery_valuation_minus_valuation_v1": "minus_valuation",
    }
    return replacements.get(feature_set, feature_set)


def load_comparison(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "feature_set_comparison.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()

    return df


def prepare_family(df: pd.DataFrame, family: str, mode: str) -> pd.DataFrame:
    config = FAMILY_CONFIG[family]
    members = set(config["members"])
    full_name = config["full"]
    fallback_full = config["fallback_full"]

    out = df[df["Feature_Set"].isin(members)].copy()

    if mode == "reopt":
        out = out[out["Experiment_Mode"].astype(str).eq("reoptimized")]
    elif mode == "fixed":
        out = out[out["Experiment_Mode"].astype(str).eq("fixed_setup")]
    elif mode == "all":
        pass
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if out.empty:
        return out

    reference = out[out["Feature_Set"].eq(full_name)].copy()

    if reference.empty:
        reference = out[out["Feature_Set"].eq(fallback_full)].copy()

    if reference.empty:
        raise ValueError(
            f"No full reference row found for family={family}. "
            f"Expected {full_name} or {fallback_full}."
        )

    metric_cols = [
        "Score",
        "Edge_vs_Breakeven",
        "Avg_Return",
        "Expectancy_Per_Day",
        "Timeout_Rate",
        "Trades",
    ]

    ref_map = reference.set_index("Experiment_Mode")[metric_cols].to_dict(orient="index")
    default_ref = ref_map[next(iter(ref_map))]

    for metric in metric_cols:
        out[f"{metric}_Delta_vs_Full"] = out.apply(
            lambda row: row[metric] - ref_map.get(row["Experiment_Mode"], default_ref)[metric],
            axis=1,
        )

    out["Short_Feature"] = out["Feature_Set"].apply(short_name)

    return out


def build_display(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if df.empty:
        return df

    sort_map = {
        "score": "Score",
        "edge": "Edge_vs_Breakeven",
        "return": "Avg_Return",
        "expectancy": "Expectancy_Per_Day",
    }

    sort_col = sort_map.get(sort_by, "Score")
    out = df.sort_values(sort_col, ascending=False).copy()

    display = pd.DataFrame(
        {
            "Feature": out["Short_Feature"],
            "Mode": out["Experiment_Mode"].replace(
                {"reoptimized": "reopt", "fixed_setup": "fixed"}
            ),
            "Setup": (
                out["LOOKAHEAD"].astype(str)
                + "D | TP "
                + out["TP"].apply(lambda x: pct(x, 0))
                + " | SL "
                + out["SL"].apply(lambda x: pct(x, 0))
                + " | TH "
                + out["THRESH"].apply(lambda x: num(x, 2))
            ),
            "Score": out["Score"].apply(lambda x: num(x, 4)),
            "Score_D": out["Score_Delta_vs_Full"].apply(lambda x: num(x, 4)),
            "Edge": out["Edge_vs_Breakeven"].apply(pct),
            "Edge_D": out["Edge_vs_Breakeven_Delta_vs_Full"].apply(pct),
            "AvgRet": out["Avg_Return"].apply(pct),
            "AvgRet_D": out["Avg_Return_Delta_vs_Full"].apply(pct),
            "ExpDay": out["Expectancy_Per_Day"].apply(pct),
            "Timeout": out["Timeout_Rate"].apply(pct),
            "Trades": out["Trades"].apply(lambda x: "" if pd.isna(x) else str(int(x))),
            "Status": out["Research_Status"],
            "Pass": out["Pass_Tradeable_Rule"].astype(str),
        }
    )

    return display


def print_family_summary(df: pd.DataFrame, family: str, mode: str, sort_by: str) -> None:
    prepared = prepare_family(df, family=family, mode=mode)

    print("=" * 130)
    print(f"ABLATION SUMMARY | family={family} | mode={mode} | sorted by {sort_by}")
    print("=" * 130)

    if prepared.empty:
        print("No matching rows found. Run feature_experiment_runner after applying the ablation patch.")
        return

    display = build_display(prepared, sort_by=sort_by)
    print(display.to_string(index=False))

    print("")
    print("Reading guide:")
    print("- Negative Delta means the ablated version is worse than the full version.")
    print("- If minus_* drops Edge/Score materially, that removed group is important.")
    print("- If minus_* improves Edge/Score, that removed group may be noisy.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ablation results from feature_set_comparison.csv."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--family",
        type=str,
        default="trend_recovery",
        choices=["trend_recovery", "recovery_valuation", "all"],
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="reopt",
        choices=["reopt", "fixed", "all"],
    )

    parser.add_argument(
        "--sort-by",
        type=str,
        default="score",
        choices=["score", "edge", "return", "expectancy"],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    comparison = load_comparison(Path(args.output_dir))

    if args.family == "all":
        print_family_summary(comparison, family="trend_recovery", mode=args.mode, sort_by=args.sort_by)
        print("")
        print_family_summary(comparison, family="recovery_valuation", mode=args.mode, sort_by=args.sort_by)
    else:
        print_family_summary(comparison, family=args.family, mode=args.mode, sort_by=args.sort_by)
