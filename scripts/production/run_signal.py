from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

for path in [PROJECT_ROOT, SRC_ROOT]:
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


from ml_signal.production.config_loader import (  # noqa: E402
    apply_overrides,
    load_production_signal_config,
)
from ml_signal.production.signal_engine import run_latest_signal  # noqa: E402
from ml_signal.production.signal_report import write_signal_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the latest production-candidate signal for one ticker/profile."
    )

    parser.add_argument("--ticker", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--config", default=None)

    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--lookahead", type=int, default=None)
    parser.add_argument("--tp", type=float, default=None)
    parser.add_argument("--sl", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--round-trip-cost", type=float, default=None)
    parser.add_argument("--min-edge-vs-breakeven", type=float, default=None)
    parser.add_argument("--watch-margin", type=float, default=None)

    parser.add_argument(
        "--output-dir",
        default="reports/signals",
        help="Directory where latest signal markdown/json files are written.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_production_signal_config(
        project_root=PROJECT_ROOT,
        ticker=args.ticker,
        profile=args.profile,
        config_path=args.config,
    )

    config = apply_overrides(
        config,
        candidate_name=args.candidate_name,
        feature_set=args.feature_set,
        lookahead=args.lookahead,
        tp=args.tp,
        sl=args.sl,
        threshold=args.threshold,
        round_trip_cost=args.round_trip_cost,
        min_edge_vs_breakeven=args.min_edge_vs_breakeven,
        watch_margin=args.watch_margin,
    )

    payload = run_latest_signal(config)

    md_path, json_path = write_signal_outputs(
        payload,
        output_dir=PROJECT_ROOT / args.output_dir,
    )

    print("=" * 100)
    print("Latest production-candidate signal generated")
    print("=" * 100)
    print(f"Ticker/Profile: {payload['ticker']} {payload['profile']}")
    print(f"Candidate:      {payload['candidate_name']}")
    print(f"Probability:    {payload['signal']['probability']:.4f}")
    print(f"Action:         {payload['signal']['action']}")
    print(f"Markdown:       {md_path}")
    print(f"JSON:           {json_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
