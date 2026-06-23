from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# =============================================================================
# ACCOUNT ALIASES
# =============================================================================

ACCOUNT_ALIASES = {
    "total_assets": [
        "TOTAL ASSETS",
        "Total assets",
        "TỔNG TÀI SẢN",
        "Tổng tài sản",
    ],
    "total_sources": [
        "TOTAL RESOURCES",
        "Total resources",
        "TOTAL LIABILITIES AND OWNER'S EQUITY",
        "TOTAL LIABILITIES AND OWNERS EQUITY",
        "TOTAL LIABILITIES AND SHAREHOLDERS' EQUITY",
        "TỔNG CỘNG NGUỒN VỐN",
        "Tổng cộng nguồn vốn",
    ],
    "book_equity_total": [
        "OWNER'S EQUITY",
        "OWNERS EQUITY",
        "SHAREHOLDERS' EQUITY",
        "SHAREHOLDERS EQUITY",
        "STOCKHOLDERS' EQUITY",
        "STOCKHOLDERS EQUITY",
        "TOTAL EQUITY",
        "VỐN CHỦ SỞ HỮU",
        "Vốn chủ sở hữu",
    ],
    "non_controlling_interest": [
        "Minority interests",
        "Minority interest",
        "Minority Interest",
        "Non-controlling interests",
        "Non controlling interests",
        "Non-controlling interest",
        "Non controlling interest",
        "Lợi ích cổ đông không kiểm soát",
        "Lợi ích của cổ đông thiểu số",
    ],
}


# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = text.replace("đ", "d").replace("Đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_account_name(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def parse_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    if text in ["", "-", "--", "nan", "NaN", "None"]:
        return np.nan

    negative = False

    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    text = text.replace(",", "")
    text = text.replace(" ", "")

    try:
        number = float(text)
        return -number if negative else number
    except ValueError:
        return np.nan


# =============================================================================
# INPUT PATH RESOLUTION
# =============================================================================

def resolve_input_path(input_path: str | None) -> Path:
    if input_path:
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"Input Excel file not found: {path}")

        return path

    candidates = [
        Path("data/raw_fundamental/CTD_financials.xlsx"),
        Path("data/raw_fundamental/Balance sheet CTD.xlsx"),
        Path("CTD_financials.xlsx"),
        Path("Balance sheet CTD.xlsx"),
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.info("Input file auto-detected | path=%s", candidate)
            return candidate

    raise FileNotFoundError(
        "No input file was provided and no default input file was found. "
        "Place CTD_financials.xlsx in data/raw_fundamental/ or pass --input."
    )


# =============================================================================
# PERIOD PARSING
# =============================================================================

def parse_period_label(value: Any) -> tuple[int, int] | None:
    if pd.isna(value):
        return None

    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)

    patterns = [
        r"Q(?P<q>[1-4])\s*[/\-_ ]?\s*(?P<year>\d{4})",
        r"(?P<year>\d{4})\s*[/\-_ ]?\s*Q(?P<q>[1-4])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return int(match.group("year")), int(match.group("q"))

    return None


def quarter_end_date(calendar_year: int, calendar_quarter: int) -> pd.Timestamp:
    quarter_end_map = {
        1: (3, 31),
        2: (6, 30),
        3: (9, 30),
        4: (12, 31),
    }

    month, day = quarter_end_map[int(calendar_quarter)]

    return pd.Timestamp(year=int(calendar_year), month=month, day=day)


def fallback_effective_date(
    quarter_end: pd.Timestamp,
    calendar_quarter: int,
    quarterly_lag_days: int = 45,
    q4_lag_days: int = 90,
) -> pd.Timestamp:
    lag_days = q4_lag_days if int(calendar_quarter) == 4 else quarterly_lag_days

    return pd.Timestamp(quarter_end) + pd.Timedelta(days=lag_days)


# =============================================================================
# EXCEL PARSING
# =============================================================================

def find_header_row(raw: pd.DataFrame, max_scan_rows: int = 40) -> int:
    best_row = None
    best_period_count = 0

    scan_limit = min(max_scan_rows, len(raw))

    for row_idx in range(scan_limit):
        row_values = raw.iloc[row_idx].tolist()

        normalized_values = [normalize_text(value) for value in row_values]

        has_account_label = any(
            value in ["items", "item", "account", "accounts", "chi tieu", "chi tieu bao cao"]
            or "chi tieu" in value
            for value in normalized_values
        )

        period_count = sum(
            parse_period_label(value) is not None
            for value in row_values
        )

        if has_account_label and period_count >= 4:
            return row_idx

        if period_count > best_period_count:
            best_period_count = period_count
            best_row = row_idx

    if best_row is not None and best_period_count >= 4:
        logger.warning(
            "Header row inferred by period-count fallback | row=%d | periods=%d",
            best_row + 1,
            best_period_count,
        )
        return best_row

    raise ValueError(
        "Could not identify header row. Expected a row with ITEMS and quarter labels like Q1 2024."
    )


def read_balance_sheet_wide(
    excel_path: str | Path,
    sheet_name: str | int = 0,
    header_row: int | None = None,
) -> tuple[pd.DataFrame, int, str]:
    excel_path = Path(excel_path)

    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    used_sheet_name = str(sheet_name)

    if header_row is None:
        header_row = find_header_row(raw)

    headers = raw.iloc[header_row].tolist()
    data = raw.iloc[header_row + 1:].copy()
    data.columns = headers

    data = data.dropna(how="all")
    data = data.dropna(axis=1, how="all")

    return data, header_row, used_sheet_name


def identify_account_column(wide: pd.DataFrame) -> Any:
    for col in wide.columns:
        normalized = normalize_text(col)

        if normalized in ["items", "item", "account", "accounts", "chi tieu", "chi tieu bao cao"]:
            return col

        if "chi tieu" in normalized:
            return col

    for col in wide.columns:
        if parse_period_label(col) is None:
            return col

    raise ValueError("Could not identify the account-name column.")


def identify_period_columns(wide: pd.DataFrame) -> dict[Any, dict[str, Any]]:
    period_columns = {}

    for col in wide.columns:
        parsed = parse_period_label(col)

        if parsed is None:
            continue

        calendar_year, calendar_quarter = parsed
        q_end = quarter_end_date(calendar_year, calendar_quarter)

        period_columns[col] = {
            "Period_Label": str(col).strip(),
            "Calendar_Year": calendar_year,
            "Calendar_Quarter": calendar_quarter,
            "Quarter_End_Date": q_end,
        }

    if not period_columns:
        raise ValueError("No period columns were found. Expected labels such as Q1 2024.")

    return period_columns


# =============================================================================
# STANDARDIZATION
# =============================================================================

def balance_sheet_wide_to_long(
    wide: pd.DataFrame,
    header_row: int,
    source_file: str,
    source_sheet: str,
    ticker: str = "CTD",
    quarterly_lag_days: int = 45,
    q4_lag_days: int = 90,
    keep_missing_values: bool = True,
) -> pd.DataFrame:
    account_col = identify_account_column(wide)
    period_columns = identify_period_columns(wide)

    records = []
    account_counter = 0

    for row_pos, (_, row) in enumerate(wide.iterrows(), start=1):
        account_name = clean_account_name(row.get(account_col))

        if account_name == "":
            continue

        account_counter += 1
        account_id = f"BS_{account_counter:03d}"

        original_excel_row = header_row + 1 + row_pos

        for period_col, period_meta in period_columns.items():
            value = parse_number(row.get(period_col))

            if not keep_missing_values and pd.isna(value):
                continue

            q_end = period_meta["Quarter_End_Date"]
            cal_q = period_meta["Calendar_Quarter"]
            effective_date = fallback_effective_date(
                q_end,
                cal_q,
                quarterly_lag_days=quarterly_lag_days,
                q4_lag_days=q4_lag_days,
            )

            records.append(
                {
                    "Ticker": ticker.upper(),
                    "Statement": "Balance Sheet",
                    "Source_File": source_file,
                    "Source_Sheet": source_sheet,
                    "Account_ID": account_id,
                    "Original_Excel_Row": original_excel_row,
                    "Account_Name": account_name,
                    "Account_Name_Normalized": normalize_text(account_name),
                    "Period_Label": period_meta["Period_Label"],
                    "Calendar_Year": period_meta["Calendar_Year"],
                    "Calendar_Quarter": cal_q,
                    "Quarter_End_Date": q_end,
                    "Effective_Date": effective_date,
                    "Value": value,
                }
            )

    long_df = pd.DataFrame(records)

    if long_df.empty:
        raise ValueError("Long balance sheet dataframe is empty after standardization.")

    long_df = long_df.sort_values(
        [
            "Calendar_Year",
            "Calendar_Quarter",
            "Original_Excel_Row",
        ]
    ).reset_index(drop=True)

    return long_df


# =============================================================================
# ACCOUNT EXTRACTION
# =============================================================================

def get_account_map(long_df: pd.DataFrame) -> pd.DataFrame:
    return (
        long_df[
            [
                "Account_ID",
                "Account_Name",
                "Account_Name_Normalized",
                "Original_Excel_Row",
            ]
        ]
        .drop_duplicates()
        .sort_values("Original_Excel_Row")
        .reset_index(drop=True)
    )


def find_first_account_id(
    long_df: pd.DataFrame,
    aliases: list[str],
    required: bool = True,
) -> tuple[str | None, str | None]:
    account_map = get_account_map(long_df)
    alias_norm = [normalize_text(alias) for alias in aliases]

    exact_matches = account_map[
        account_map["Account_Name_Normalized"].isin(alias_norm)
    ]

    if not exact_matches.empty:
        row = exact_matches.iloc[0]
        return str(row["Account_ID"]), str(row["Account_Name"])

    if required:
        available_preview = account_map["Account_Name"].head(20).tolist()

        raise ValueError(
            "Required account was not found. "
            f"Aliases={aliases}. Available preview={available_preview}"
        )

    return None, None


def find_all_account_ids(
    long_df: pd.DataFrame,
    aliases: list[str],
) -> list[tuple[str, str]]:
    account_map = get_account_map(long_df)
    alias_norm = [normalize_text(alias) for alias in aliases]

    matches = account_map[
        account_map["Account_Name_Normalized"].isin(alias_norm)
    ].copy()

    return [
        (str(row["Account_ID"]), str(row["Account_Name"]))
        for _, row in matches.iterrows()
    ]


def extract_account_series(
    long_df: pd.DataFrame,
    account_id: str | None,
    value_col_name: str,
) -> pd.DataFrame:
    period_cols = [
        "Ticker",
        "Period_Label",
        "Calendar_Year",
        "Calendar_Quarter",
        "Quarter_End_Date",
        "Effective_Date",
    ]

    if account_id is None:
        base = long_df[period_cols].drop_duplicates().copy()
        base[value_col_name] = np.nan

        return base

    output = long_df.loc[
        long_df["Account_ID"].eq(account_id),
        period_cols + ["Value"],
    ].copy()

    output = output.rename(columns={"Value": value_col_name})

    return output


def merge_period_frame(base: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    merge_keys = [
        "Ticker",
        "Period_Label",
        "Calendar_Year",
        "Calendar_Quarter",
        "Quarter_End_Date",
        "Effective_Date",
    ]

    return base.merge(frame, on=merge_keys, how="left")


def coalesce_nonzero_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """
    Pick the first non-missing, non-zero value across candidate columns.
    If all values are missing or zero, return 0.
    """
    result = pd.Series(np.nan, index=df.index, dtype="float64")

    for col in columns:
        if col not in df.columns:
            continue

        candidate = pd.to_numeric(df[col], errors="coerce")
        candidate_is_useful = candidate.notna() & (candidate != 0)
        result = result.where(result.notna(), candidate.where(candidate_is_useful))

    result = result.fillna(0.0)

    return result


def create_book_equity_table(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_equity_id, total_equity_name = find_first_account_id(
        long_df,
        aliases=ACCOUNT_ALIASES["book_equity_total"],
        required=True,
    )

    total_assets_id, total_assets_name = find_first_account_id(
        long_df,
        aliases=ACCOUNT_ALIASES["total_assets"],
        required=False,
    )

    total_sources_id, total_sources_name = find_first_account_id(
        long_df,
        aliases=ACCOUNT_ALIASES["total_sources"],
        required=False,
    )

    nci_matches = find_all_account_ids(
        long_df,
        aliases=ACCOUNT_ALIASES["non_controlling_interest"],
    )

    total_equity = extract_account_series(
        long_df,
        total_equity_id,
        "Book_Equity_Total",
    )

    book = total_equity.copy()

    nci_cols = []

    for i, (account_id, account_name) in enumerate(nci_matches, start=1):
        col_name = f"NCI_Candidate_{i}"
        nci_cols.append(col_name)

        nci_series = extract_account_series(
            long_df,
            account_id,
            col_name,
        )

        book = merge_period_frame(book, nci_series)

    total_assets = extract_account_series(
        long_df,
        total_assets_id,
        "Total_Assets",
    )

    total_sources = extract_account_series(
        long_df,
        total_sources_id,
        "Total_Sources",
    )

    book = merge_period_frame(book, total_assets)
    book = merge_period_frame(book, total_sources)

    book["Non_Controlling_Interest"] = coalesce_nonzero_columns(book, nci_cols)

    book["Book_Equity_Parent"] = (
        book["Book_Equity_Total"]
        - book["Non_Controlling_Interest"]
    )

    book["Balance_Difference"] = book["Total_Assets"] - book["Total_Sources"]

    book["Balance_Difference_Pct"] = np.where(
        book["Total_Assets"].abs() > 0,
        book["Balance_Difference"] / book["Total_Assets"].abs(),
        np.nan,
    )

    book["Book_Equity_Total_Account"] = total_equity_name
    book["NCI_Accounts_Found"] = "; ".join(name for _, name in nci_matches)
    book["Total_Assets_Account"] = total_assets_name
    book["Total_Sources_Account"] = total_sources_name

    output_cols = [
        "Ticker",
        "Period_Label",
        "Calendar_Year",
        "Calendar_Quarter",
        "Quarter_End_Date",
        "Effective_Date",
        "Book_Equity_Total",
        "Non_Controlling_Interest",
        "Book_Equity_Parent",
        "Total_Assets",
        "Total_Sources",
        "Balance_Difference",
        "Balance_Difference_Pct",
        "Book_Equity_Total_Account",
        "NCI_Accounts_Found",
        "Total_Assets_Account",
        "Total_Sources_Account",
    ]

    book = book[output_cols].copy()

    book = book.sort_values(
        ["Calendar_Year", "Calendar_Quarter"]
    ).reset_index(drop=True)

    max_abs_balance_diff = book["Balance_Difference"].abs().max(skipna=True)

    audit_records = [
        {
            "Check": "book_equity_total_account",
            "Value": total_equity_name,
            "Status": "ok" if total_equity_name else "missing",
        },
        {
            "Check": "nci_accounts_found",
            "Value": "; ".join(name for _, name in nci_matches),
            "Status": "ok" if nci_matches else "missing_optional",
        },
        {
            "Check": "total_assets_account",
            "Value": total_assets_name,
            "Status": "ok" if total_assets_name else "missing_optional",
        },
        {
            "Check": "total_sources_account",
            "Value": total_sources_name,
            "Status": "ok" if total_sources_name else "missing_optional",
        },
        {
            "Check": "max_abs_balance_difference",
            "Value": float(max_abs_balance_diff) if pd.notna(max_abs_balance_diff) else np.nan,
            "Status": "ok" if pd.notna(max_abs_balance_diff) and max_abs_balance_diff <= 1 else "review",
        },
        {
            "Check": "rows",
            "Value": int(len(book)),
            "Status": "ok",
        },
        {
            "Check": "start_period",
            "Value": str(book["Period_Label"].iloc[0]) if len(book) else "",
            "Status": "ok" if len(book) else "empty",
        },
        {
            "Check": "end_period",
            "Value": str(book["Period_Label"].iloc[-1]) if len(book) else "",
            "Status": "ok" if len(book) else "empty",
        },
    ]

    audit_df = pd.DataFrame(audit_records)

    return book, audit_df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def standardize_balance_sheet_file(
    input_path: str | Path | None,
    output_dir: str | Path,
    ticker: str = "CTD",
    sheet_name: str | int = 0,
    header_row: int | None = None,
    quarterly_lag_days: int = 45,
    q4_lag_days: int = 90,
    keep_missing_values: bool = True,
) -> dict[str, Path]:
    input_path = resolve_input_path(str(input_path) if input_path else None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wide, used_header_row, used_sheet_name = read_balance_sheet_wide(
        excel_path=input_path,
        sheet_name=sheet_name,
        header_row=header_row,
    )

    long_df = balance_sheet_wide_to_long(
        wide=wide,
        header_row=used_header_row,
        source_file=input_path.name,
        source_sheet=used_sheet_name,
        ticker=ticker,
        quarterly_lag_days=quarterly_lag_days,
        q4_lag_days=q4_lag_days,
        keep_missing_values=keep_missing_values,
    )

    book_equity_df, audit_df = create_book_equity_table(long_df)

    ticker_upper = ticker.upper()

    long_path = output_dir / f"{ticker_upper}_balance_sheet_long.csv"
    book_path = output_dir / f"{ticker_upper}_book_equity_standardized.csv"
    audit_path = output_dir / f"{ticker_upper}_book_equity_audit.csv"

    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    book_equity_df.to_csv(book_path, index=False, encoding="utf-8-sig")
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")

    logger.info(
        "Balance sheet long file saved | path=%s | rows=%d",
        long_path,
        len(long_df),
    )

    logger.info(
        "Book equity file saved | path=%s | rows=%d",
        book_path,
        len(book_equity_df),
    )

    logger.info(
        "Audit file saved | path=%s | rows=%d",
        audit_path,
        len(audit_df),
    )

    return {
        "long": long_path,
        "book_equity": book_path,
        "audit": audit_path,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize CTD balance sheet Excel into long format and book-equity table."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to the input balance sheet Excel file. "
            "If omitted, the script searches common project paths."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed_data/fundamental",
        help="Output directory for standardized CSV files.",
    )

    parser.add_argument(
        "--ticker",
        type=str,
        default="CTD",
        help="Ticker symbol.",
    )

    parser.add_argument(
        "--sheet-name",
        type=str,
        default="0",
        help="Sheet name or sheet index. Default is 0, meaning the first sheet.",
    )

    parser.add_argument(
        "--header-row",
        type=int,
        default=None,
        help="Optional zero-based header row index. Leave blank for auto-detection.",
    )

    parser.add_argument(
        "--quarterly-lag-days",
        type=int,
        default=45,
        help="Fallback reporting lag for Q1/Q2/Q3 when announcement date is not available.",
    )

    parser.add_argument(
        "--q4-lag-days",
        type=int,
        default=90,
        help="Fallback reporting lag for Q4 when announcement date is not available.",
    )

    parser.add_argument(
        "--drop-missing-values",
        action="store_true",
        help="Drop missing values from the long-format balance sheet output.",
    )

    return parser.parse_args()


def normalize_sheet_name_arg(sheet_name: str) -> str | int:
    if str(sheet_name).isdigit():
        return int(sheet_name)

    return sheet_name


if __name__ == "__main__":
    args = parse_args()

    standardize_balance_sheet_file(
        input_path=args.input,
        output_dir=args.output_dir,
        ticker=args.ticker,
        sheet_name=normalize_sheet_name_arg(args.sheet_name),
        header_row=args.header_row,
        quarterly_lag_days=args.quarterly_lag_days,
        q4_lag_days=args.q4_lag_days,
        keep_missing_values=not args.drop_missing_values,
    )
