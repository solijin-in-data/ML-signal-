from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from datetime import date

from data_vendors.sstock_client import SStockClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_symbols(symbols: str) -> list[str]:
    return [
        symbol.strip().upper()
        for symbol in symbols.split(",")
        if symbol.strip()
    ]


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch price history and market cap data from SStock."
    )

    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated ticker list. Example: CTD,VPB,STB",
    )

    parser.add_argument(
        "--from-date",
        type=str,
        default="2000-01-01",
        help="Start date in YYYY-MM-DD format. Default: 2000-01-01.",
    )

    parser.add_argument(
        "--to-date",
        type=str,
        default=date.today().isoformat(),
        help="End date in YYYY-MM-DD format. Default: today.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/vendor_data/sstock",
        help="Output directory.",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "xlsx"],
        help="Output file format.",
    )

    args = parser.parse_args()

    client = SStockClient()
    output_dir = Path(args.output_dir)

    for symbol in parse_symbols(args.symbols):
        result = client.fetch_price_history(
            symbol=symbol,
            start_date=args.from_date,
            end_date=args.to_date,
        )

        suffix = "csv" if args.format == "csv" else "xlsx"
        output_path = output_dir / f"{symbol}_sstock_price_history.{suffix}"

        client.save_result(
            result=result,
            output_path=output_path,
            file_format=args.format,
        )


if __name__ == "__main__":
    run()