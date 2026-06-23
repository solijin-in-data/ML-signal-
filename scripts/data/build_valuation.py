from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def parse_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def default_marketcap_path(ticker: str) -> Path:
    return Path("data/vendor_data/sstock") / f"{ticker.upper()}_sstock_price_history.csv"


def default_book_equity_path(ticker: str) -> Path:
    return Path("data/processed_data/fundamental") / f"{ticker.upper()}_book_equity_standardized.csv"


def default_output_path(ticker: str) -> Path:
    return Path("data/processed_data/valuation") / f"{ticker.upper()}_valuation_standardized.csv"


def default_audit_path(ticker: str) -> Path:
    return Path("data/processed_data/valuation") / f"{ticker.upper()}_valuation_audit.csv"


def rolling_percentile(series: pd.Series) -> float:
    current_value = series.iloc[-1]
    historical = series.dropna()

    if len(historical) == 0 or pd.isna(current_value):
        return np.nan

    return float((historical <= current_value).mean())


def load_marketcap_data(path: str | Path, ticker: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Market-cap file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()

    rename_map = {
        "date": "Date",
        "symbol": "Symbol",
        "close": "Close",
        "marketCap": "MarketCap",
        "market_cap": "MarketCap",
        "marketCapitalization": "MarketCap",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "Date" not in df.columns:
        raise ValueError("Market-cap file must contain Date/date column.")

    if "MarketCap" not in df.columns:
        raise ValueError("Market-cap file must contain MarketCap/marketCap column.")

    if "Symbol" not in df.columns:
        df["Symbol"] = ticker.upper()

    if "Close" not in df.columns:
        df["Close"] = np.nan

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Symbol"] = df["Symbol"].astype(str).str.upper()
    df["Close"] = parse_number(df["Close"])
    df["MarketCap"] = parse_number(df["MarketCap"])

    df = df[df["Symbol"].eq(ticker.upper())].copy()
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date", "Symbol"], keep="last")

    median_marketcap = df["MarketCap"].median(skipna=True)

    if pd.notna(median_marketcap) and median_marketcap < 1_000_000_000:
        logger.warning(
            "MarketCap median looks small. Check whether MarketCap is already in billion VND. median=%s",
            median_marketcap,
        )

    return df[["Date", "Symbol", "Close", "MarketCap"]].reset_index(drop=True)


def load_book_equity_data(path: str | Path, ticker: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Book-equity file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()

    required_cols = [
        "Ticker",
        "Calendar_Year",
        "Calendar_Quarter",
        "Quarter_End_Date",
        "Effective_Date",
        "Book_Equity_Parent",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing from book-equity file: {col}")

    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    df = df[df["Ticker"].eq(ticker.upper())].copy()

    df["Quarter_End_Date"] = pd.to_datetime(df["Quarter_End_Date"], errors="coerce")
    df["Effective_Date"] = pd.to_datetime(df["Effective_Date"], errors="coerce")
    df["Book_Equity_Parent"] = parse_number(df["Book_Equity_Parent"])

    for col in ["Book_Equity_Total", "Non_Controlling_Interest", "Total_Assets", "Total_Sources"]:
        if col in df.columns:
            df[col] = parse_number(df[col])

    df = df.dropna(subset=["Effective_Date", "Book_Equity_Parent"])
    df = df.sort_values("Effective_Date")
    df = df.drop_duplicates(subset=["Effective_Date"], keep="last")

    keep_cols = [
        "Ticker",
        "Period_Label",
        "Calendar_Year",
        "Calendar_Quarter",
        "Quarter_End_Date",
        "Effective_Date",
        "Book_Equity_Total",
        "Non_Controlling_Interest",
        "Book_Equity_Parent",
    ]

    keep_cols = [col for col in keep_cols if col in df.columns]

    return df[keep_cols].reset_index(drop=True)


def build_valuation_dataframe(
    marketcap_df: pd.DataFrame,
    book_equity_df: pd.DataFrame,
    ticker: str,
    rolling_window: int = 756,
    min_periods: int = 120,
) -> pd.DataFrame:
    left = marketcap_df.sort_values("Date").copy()
    right = book_equity_df.sort_values("Effective_Date").copy()

    valuation = pd.merge_asof(
        left,
        right,
        left_on="Date",
        right_on="Effective_Date",
        direction="backward",
    )

    valuation["Ticker"] = ticker.upper()

    valuation["PB"] = valuation["MarketCap"] / (valuation["Book_Equity_Parent"] + 1e-9)

    valuation["Book_to_Market"] = np.where(
        valuation["PB"] > 0,
        1 / (valuation["PB"] + 1e-9),
        np.nan,
    )

    valuation["Log_PB"] = np.where(
        valuation["PB"] > 0,
        np.log(valuation["PB"]),
        np.nan,
    )

    pb_mean = valuation["PB"].rolling(rolling_window, min_periods=min_periods).mean()
    pb_std = valuation["PB"].rolling(rolling_window, min_periods=min_periods).std()

    valuation[f"PB_Z_{rolling_window}"] = (valuation["PB"] - pb_mean) / (pb_std + 1e-9)

    valuation[f"PB_Percentile_{rolling_window}"] = (
        valuation["PB"]
        .rolling(rolling_window, min_periods=min_periods)
        .apply(rolling_percentile, raw=False)
    )

    valuation["MarketCap_TyDong"] = valuation["MarketCap"] / 1_000_000_000
    valuation["Book_Equity_Parent_TyDong"] = valuation["Book_Equity_Parent"] / 1_000_000_000

    output_cols = [
        "Date",
        "Ticker",
        "Close",
        "MarketCap",
        "MarketCap_TyDong",
        "Calendar_Year",
        "Calendar_Quarter",
        "Period_Label",
        "Quarter_End_Date",
        "Effective_Date",
        "Book_Equity_Parent",
        "Book_Equity_Parent_TyDong",
        "Book_Equity_Total",
        "Non_Controlling_Interest",
        "PB",
        "Book_to_Market",
        "Log_PB",
        f"PB_Z_{rolling_window}",
        f"PB_Percentile_{rolling_window}",
    ]

    output_cols = [col for col in output_cols if col in valuation.columns]

    valuation = valuation[output_cols].copy()
    valuation = valuation.replace([np.inf, -np.inf], np.nan)
    valuation = valuation.sort_values("Date")
    valuation = valuation.reset_index(drop=True)

    return valuation


def create_audit_report(valuation: pd.DataFrame, ticker: str) -> pd.DataFrame:
    rows = []

    total_rows = len(valuation)
    valuation_rows = int(valuation["PB"].notna().sum()) if "PB" in valuation.columns else 0
    missing_pb_rows = total_rows - valuation_rows

    rows.append({"Check": "rows_total", "Value": total_rows, "Status": "ok" if total_rows > 0 else "empty"})
    rows.append({"Check": "rows_with_pb", "Value": valuation_rows, "Status": "ok" if valuation_rows > 0 else "empty"})
    rows.append({"Check": "rows_missing_pb", "Value": missing_pb_rows, "Status": "review" if missing_pb_rows > 0 else "ok"})

    if total_rows > 0:
        rows.append({"Check": "start_date", "Value": str(valuation["Date"].min().date()), "Status": "ok"})
        rows.append({"Check": "end_date", "Value": str(valuation["Date"].max().date()), "Status": "ok"})

    if "PB" in valuation.columns and valuation["PB"].notna().any():
        rows.extend(
            [
                {"Check": "pb_min", "Value": float(valuation["PB"].min(skipna=True)), "Status": "ok"},
                {"Check": "pb_median", "Value": float(valuation["PB"].median(skipna=True)), "Status": "ok"},
                {"Check": "pb_max", "Value": float(valuation["PB"].max(skipna=True)), "Status": "review" if valuation["PB"].max(skipna=True) > 20 else "ok"},
                {"Check": "latest_pb", "Value": float(valuation["PB"].dropna().iloc[-1]), "Status": "ok"},
            ]
        )

    if "Effective_Date" in valuation.columns:
        leakage_mask = valuation["Effective_Date"].notna() & (valuation["Effective_Date"] > valuation["Date"])

        rows.append(
            {
                "Check": "lookahead_leakage_rows",
                "Value": int(leakage_mask.sum()),
                "Status": "ok" if int(leakage_mask.sum()) == 0 else "fail",
            }
        )

    rows.append({"Check": "ticker", "Value": ticker.upper(), "Status": "ok"})

    return pd.DataFrame(rows)


def build_valuation_file(
    ticker: str,
    marketcap_path: str | Path | None = None,
    book_equity_path: str | Path | None = None,
    output_path: str | Path | None = None,
    audit_path: str | Path | None = None,
    rolling_window: int = 756,
    min_periods: int = 120,
) -> dict[str, Path]:
    ticker = ticker.upper()

    marketcap_path = Path(marketcap_path) if marketcap_path else default_marketcap_path(ticker)
    book_equity_path = Path(book_equity_path) if book_equity_path else default_book_equity_path(ticker)
    output_path = Path(output_path) if output_path else default_output_path(ticker)
    audit_path = Path(audit_path) if audit_path else default_audit_path(ticker)

    marketcap_df = load_marketcap_data(marketcap_path, ticker)
    book_equity_df = load_book_equity_data(book_equity_path, ticker)

    valuation = build_valuation_dataframe(
        marketcap_df=marketcap_df,
        book_equity_df=book_equity_df,
        ticker=ticker,
        rolling_window=rolling_window,
        min_periods=min_periods,
    )

    audit = create_audit_report(valuation, ticker)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    valuation.to_csv(output_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    logger.info("Valuation file saved | ticker=%s | path=%s | rows=%d", ticker, output_path, len(valuation))
    logger.info("Valuation audit saved | ticker=%s | path=%s | rows=%d", ticker, audit_path, len(audit))

    return {"valuation": output_path, "audit": audit_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build daily valuation features using SStock MarketCap and standardized book equity."
    )

    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol. Example: CTD")

    parser.add_argument(
        "--marketcap-path",
        type=str,
        default=None,
        help="Path to SStock price-history file. Default: data/vendor_data/sstock/{TICKER}_sstock_price_history.csv",
    )

    parser.add_argument(
        "--book-equity-path",
        type=str,
        default=None,
        help="Path to book-equity file. Default: data/processed_data/fundamental/{TICKER}_book_equity_standardized.csv",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output valuation CSV path. Default: data/processed_data/valuation/{TICKER}_valuation_standardized.csv",
    )

    parser.add_argument(
        "--audit-path",
        type=str,
        default=None,
        help="Output audit CSV path. Default: data/processed_data/valuation/{TICKER}_valuation_audit.csv",
    )

    parser.add_argument("--rolling-window", type=int, default=756, help="Rolling window for P/B z-score and percentile.")
    parser.add_argument("--min-periods", type=int, default=120, help="Minimum observations for rolling features.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_valuation_file(
        ticker=args.ticker,
        marketcap_path=args.marketcap_path,
        book_equity_path=args.book_equity_path,
        output_path=args.output_path,
        audit_path=args.audit_path,
        rolling_window=args.rolling_window,
        min_periods=args.min_periods,
    )
