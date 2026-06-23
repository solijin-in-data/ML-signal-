from __future__ import annotations

from pathlib import Path
import argparse
import re
import warnings

import numpy as np
import pandas as pd

from typing import cast


# =============================================================================
# PROJECT PATH CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Support both cases:
# 1) data_standardizer.py is in project root
# 2) data_standardizer.py is inside src/
if SCRIPT_DIR.name.lower() == "src":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_INPUT_DIR = DATA_DIR / "unprocessed_data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "processed_data"

FALLBACK_INPUT_DIRS = [
    DATA_DIR / "unprocessed_data",
    DATA_DIR / "unprocessed data",
    DATA_DIR / "raw",
]


# =============================================================================
# STANDARD SCHEMAS
# =============================================================================

STOCK_SCHEMA = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Change_Pct",
    "Net_Volume_Foreign",
    "Net_Value_Foreign",
    "Buy_Volume_Foreign",
    "Buy_Value_Foreign",
    "Sell_Volume_Foreign",
    "Sell_Value_Foreign",
    "Foreign_room_Remain",
    "Foreign_Ownership_Pct",
]

VNINDEX_SCHEMA = [
    "Date",
    "VN_Open",
    "VN_High",
    "VN_Low",
    "VN_Close",
    "VN_Volume",
    "VN_Change_Pct",
]


# =============================================================================
# COLUMN ALIASES
# =============================================================================

COLUMN_ALIASES = {
    "Date": [
        "Date", "TradingDate", "Trading Date", "Ngày", "Ngay",
        "VN_Date", "time", "datetime", "Ngày giao dịch", "Ngay giao dich"
    ],

    "Open": [
        "Open", "Open Price", "Mở", "Mo", "Gia mo cua", "Giá mở cửa",
        "VN_Open"
    ],

    "High": [
        "High", "High Price", "Cao", "Gia cao nhat", "Giá cao nhất",
        "VN_High"
    ],

    "Low": [
        "Low", "Low Price", "Thấp", "Thap", "Gia thap nhat", "Giá thấp nhất",
        "VN_Low"
    ],

    "Close": [
        "Close", "Last", "Price", "Lần cuối", "Lan cuoi",
        "Đóng cửa", "Dong cua", "Giá đóng cửa", "Gia dong cua",
        "VN_Close"
    ],

    "Volume": [
        "Volume", "Vol", "Vol.", "KL", "KLGD", "Khối lượng",
        "Khoi luong", "Khối lượng giao dịch", "Khoi luong giao dich",
        "VN_Volume"
    ],

    "Change_Pct": [
        "Change %", "% Change", "Change_Pct", "ChangePct",
        "% Thay đổi", "% Thay doi", "Change",
        "Thay đổi %", "Thay doi %", "VN_Change_Pct"
    ],

    "Net_Volume_Foreign": [
        "Net_Volume_Foreign", "KLGD ròng", "KLGD rong",
        "NN mua/bán ròng", "NN mua ban rong",
        "Net Foreign Volume", "Foreign Net Volume",
        "Foreign_Net_Volume"
    ],

    "Net_Value_Foreign": [
        "Net_Value_Foreign", "GTGD ròng", "GTGD rong",
        "GTGD NN ròng", "GTGD NN rong",
        "Net Foreign Value", "Foreign Net Value",
        "Foreign_Net_Value"
    ],

    "Buy_Volume_Foreign": [
        "Buy_Volume_Foreign", "NN mua", "Foreign Buy Volume",
        "Foreign_Buy_Volume"
    ],

    "Buy_Value_Foreign": [
        "Buy_Value_Foreign", "GT NN mua", "GTGD NN mua",
        "Foreign Buy Value", "Foreign_Buy_Value"
    ],

    "Sell_Volume_Foreign": [
        "Sell_Volume_Foreign", "NN bán", "NN ban",
        "Foreign Sell Volume", "Foreign_Sell_Volume"
    ],

    "Sell_Value_Foreign": [
        "Sell_Value_Foreign", "GT NN bán", "GT NN ban",
        "GTGD NN bán", "GTGD NN ban",
        "Foreign Sell Value", "Foreign_Sell_Value"
    ],

    "Foreign_room_Remain": [
        "Foreign_room_Remain", "Room còn lại", "Room con lai",
        "Foreign Room Remaining", "Room NN còn lại", "Room NN con lai"
    ],

    "Foreign_Ownership_Pct": [
        "Foreign_Ownership_Pct", "Tỷ lệ sở hữu NN", "Ty le so huu NN",
        "Foreign Ownership %", "Foreign Ownership", "Sở hữu NN", "So huu NN"
    ],
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def normalize_col_name(col: str) -> str:
    col = str(col).strip().lower()
    col = col.replace("\ufeff", "")
    col = col.replace(".", "")
    col = re.sub(r"[\s_\-\/]+", "", col)
    return col


def build_alias_lookup() -> dict[str, str]:
    lookup = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            lookup[normalize_col_name(alias)] = canonical

    return lookup


ALIAS_LOOKUP = build_alias_lookup()


def read_csv_smart(file_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1258", "cp1252", "latin1"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as e:
            last_error = e

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, sep=None, engine="python")
        except Exception as e:
            last_error = e

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"Could not read CSV file: {file_path}")


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nếu nhiều cột raw bị map về cùng một tên chuẩn,
    lấy giá trị non-null đầu tiên theo từng dòng.
    """
    if not df.columns.duplicated().any():
        return df

    result = pd.DataFrame(index=df.index)

    for col in pd.unique(df.columns):
        matching_cols = [column for column in df.columns if column == col]
        subset = df.loc[:, matching_cols].copy()

        if subset.shape[1] == 1:
            first_col = subset.columns[0]
            result[str(col)] = subset[first_col]
        else:
            filled = subset.bfill(axis="columns")
            first_col = filled.columns[0]
            result[str(col)] = filled[first_col]

    return result


def rename_columns_to_standard(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for col in df.columns:
        normalized = normalize_col_name(col)

        if normalized in ALIAS_LOOKUP:
            rename_map[col] = ALIAS_LOOKUP[normalized]

    df = df.rename(columns=rename_map)
    df = coalesce_duplicate_columns(df)
    return df


def infer_file_type(file_path: Path, df: pd.DataFrame) -> str:
    """
    Tự nhận diện file là:
    - vnindex: file benchmark thị trường
    - stock: file cổ phiếu
    """
    filename = (
        file_path.stem
        .upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )

    if "VNINDEX" in filename or "VN30" in filename or "HNXINDEX" in filename:
        return "vnindex"

    normalized_columns = {normalize_col_name(col) for col in df.columns}

    vnindex_markers = {
        "vnopen",
        "vnhigh",
        "vnlow",
        "vnclose",
        "vnvolume",
        "vnchangepct",
        "vndate",
    }

    if normalized_columns.intersection(vnindex_markers):
        return "vnindex"

    return "stock"


# =============================================================================
# NUMBER PARSING
# =============================================================================

def normalize_numeric_string(text: str) -> str:
    text = str(text).strip()

    if "," in text and "." in text:
        last_comma = text.rfind(",")
        last_dot = text.rfind(".")

        if last_comma > last_dot:
            # European format: 1.234,56
            text = text.replace(".", "")
            text = text.replace(",", ".")
        else:
            # US format: 1,234.56
            text = text.replace(",", "")

        return text

    if "," in text and "." not in text:
        parts = text.split(",")

        # 1,234 or 1,234,567
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            return text.replace(",", "")

        # 72,70
        return text.replace(",", ".")

    if "." in text and "," not in text:
        parts = text.split(".")

        # 1.234.567
        if len(parts) > 2 and all(len(p) == 3 for p in parts[1:]):
            return text.replace(".", "")

        return text

    return text


def parse_number(value):
    """
    Parse các format:
    - 72,700.00
    - 72,70
    - 1.341,86
    - 569.40K
    - 573.80M
    - 1.2B
    - 0.55%
    - 45,21%
    - (1,000)
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    if text in ["", "-", "--", "nan", "NaN", "None", "null"]:
        return np.nan

    text = (
        text.replace("\xa0", "")
        .replace(" ", "")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )

    is_negative = False

    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1]

    if text.startswith("-"):
        is_negative = True
        text = text[1:]

    is_percent = "%" in text
    text = text.replace("%", "")

    multiplier = 1.0

    if len(text) > 0 and text[-1].upper() in ["K", "M", "B"]:
        suffix = text[-1].upper()
        text = text[:-1]

        if suffix == "K":
            multiplier = 1_000
        elif suffix == "M":
            multiplier = 1_000_000
        elif suffix == "B":
            multiplier = 1_000_000_000

    text = re.sub(r"[^0-9,\.]", "", text)

    if text == "":
        return np.nan

    text = normalize_numeric_string(text)

    try:
        number = float(text) * multiplier

        if is_percent:
            number = number / 100

        if is_negative:
            number = -number

        return number

    except Exception:
        return np.nan


# =============================================================================
# DATE PARSING
# =============================================================================

def detect_date_order(date_series: pd.Series) -> str:
    """
    Tự phát hiện ngày:
    - 16/06/2026 -> dayfirst
    - 06/16/2026 -> monthfirst
    - 2026-06-16 -> iso
    """
    sample = date_series.dropna().astype(str).head(300)

    first_gt_12 = 0
    second_gt_12 = 0
    iso_like = 0

    for value in sample:
        value = value.strip()

        if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", value):
            iso_like += 1
            continue

        match = re.match(r"^(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})", value)

        if not match:
            continue

        first = int(match.group(1))
        second = int(match.group(2))

        if first > 12:
            first_gt_12 += 1

        if second > 12:
            second_gt_12 += 1

    if iso_like > first_gt_12 and iso_like > second_gt_12:
        return "iso"

    if second_gt_12 > first_gt_12:
        return "monthfirst"

    return "dayfirst"


def parse_date_column(df: pd.DataFrame, audit: dict | None = None) -> pd.DataFrame:
    if "Date" not in df.columns:
        return df

    date_order = detect_date_order(df["Date"])

    if audit is not None:
        audit["date_order_detected"] = date_order

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        if date_order == "monthfirst":
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
        elif date_order == "iso":
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    return df


# =============================================================================
# QUALITY CHECKS
# =============================================================================

def count_ohlc_invalid(
    df: pd.DataFrame,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str
) -> int:
    required = [open_col, high_col, low_col, close_col]

    if any(col not in df.columns for col in required):
        return 0

    invalid = (
        (df[high_col] < df[[open_col, close_col, low_col]].max(axis=1)) |
        (df[low_col] > df[[open_col, close_col, high_col]].min(axis=1))
    )

    return int(invalid.sum())


def clean_sort_deduplicate(df: pd.DataFrame, audit: dict) -> pd.DataFrame:
    duplicate_count = int(df["Date"].duplicated().sum())

    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.reset_index(drop=True)

    audit["duplicate_dates_removed"] = duplicate_count

    return df


# =============================================================================
# STANDARDIZE STOCK
# =============================================================================

def standardize_stock(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    audit = {}

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]

    missing_cols = [col for col in required if col not in df.columns]
    audit["missing_required_columns"] = ", ".join(missing_cols)

    if missing_cols:
        audit["error_detail"] = (
            f"Stock missing required columns: {missing_cols}. "
            f"Current columns: {list(df.columns)}"
        )
        return pd.DataFrame(), audit

    keep_cols = [col for col in STOCK_SCHEMA if col in df.columns]
    df = df[keep_cols].copy()

    df = parse_date_column(df, audit=audit)

    numeric_cols = [col for col in df.columns if col != "Date"]

    for col in numeric_cols:
        df[col] = df[col].apply(parse_number)

    rows_before_drop = len(df)

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])

    audit["dropped_missing_core_rows"] = int(rows_before_drop - len(df))

    if df.empty:
        return df, audit

    df = clean_sort_deduplicate(df, audit)

    audit["ohlc_invalid_rows"] = count_ohlc_invalid(
        df,
        open_col="Open",
        high_col="High",
        low_col="Low",
        close_col="Close",
    )

    return df, audit


# =============================================================================
# STANDARDIZE VNINDEX
# =============================================================================

def standardize_vnindex(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    VNINDEX là file benchmark thị trường, không phải file cổ phiếu.

    Input có thể là:
    - Date, VN_Open, VN_High, VN_Low, VN_Close, VN_Volume
    - Date, Price, Open, High, Low, Vol., Change %

    Output:
    - Date, VN_Open, VN_High, VN_Low, VN_Close, VN_Volume, VN_Change_Pct
    """
    audit = {}

    vnindex_column_map = {
        "VN_Date": "Date",
        "VN_Open": "Open",
        "VN_High": "High",
        "VN_Low": "Low",
        "VN_Close": "Close",
        "VN_Volume": "Volume",
        "VN_Change_Pct": "Change_Pct",
    }

    df = df.rename(
        columns={k: v for k, v in vnindex_column_map.items() if k in df.columns}
    )

    df = coalesce_duplicate_columns(df)

    required = ["Date", "Open", "High", "Low", "Close"]

    missing_cols = [col for col in required if col not in df.columns]
    audit["missing_required_columns"] = ", ".join(missing_cols)

    if missing_cols:
        audit["error_detail"] = (
            f"VNINDEX missing required columns: {missing_cols}. "
            f"Current columns: {list(df.columns)}"
        )
        return pd.DataFrame(), audit

    keep_cols = [
        col for col in [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Change_Pct",
        ]
        if col in df.columns
    ]

    df = df[keep_cols].copy()

    df = parse_date_column(df, audit=audit)

    numeric_cols = [col for col in df.columns if col != "Date"]

    for col in numeric_cols:
        df[col] = df[col].apply(parse_number)

    rows_before_drop = len(df)

    # VNINDEX có thể thiếu volume ở một số giai đoạn cũ,
    # nên không bắt buộc Volume.
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])

    audit["dropped_missing_core_rows"] = int(rows_before_drop - len(df))

    if df.empty:
        return df, audit

    df = clean_sort_deduplicate(df, audit)

    df = df.rename(
        columns={
            "Open": "VN_Open",
            "High": "VN_High",
            "Low": "VN_Low",
            "Close": "VN_Close",
            "Volume": "VN_Volume",
            "Change_Pct": "VN_Change_Pct",
        }
    )

    final_cols = [col for col in VNINDEX_SCHEMA if col in df.columns]
    df = df[final_cols].copy()

    audit["ohlc_invalid_rows"] = count_ohlc_invalid(
        df,
        open_col="VN_Open",
        high_col="VN_High",
        low_col="VN_Low",
        close_col="VN_Close",
    )

    return df, audit


# =============================================================================
# FILE STANDARDIZATION
# =============================================================================

def standardize_one_file(file_path: Path, output_dir: Path) -> dict:
    audit = {
        "input_file": file_path.name,
        "status": "FAILED",
        "file_type": None,
        "raw_rows": 0,
        "clean_rows": 0,
        "dropped_rows": 0,
        "date_min": None,
        "date_max": None,
        "date_order_detected": None,
        "missing_required_columns": None,
        "dropped_missing_core_rows": None,
        "duplicate_dates_removed": None,
        "ohlc_invalid_rows": None,
        "output_file": None,
        "error": None,
        "error_detail": None,
    }

    try:
        df_raw = read_csv_smart(file_path)
        audit["raw_rows"] = int(len(df_raw))

        df_raw.columns = df_raw.columns.astype(str).str.strip()

        file_type_before = infer_file_type(file_path, df_raw)

        df_renamed = rename_columns_to_standard(df_raw)

        file_type_after = infer_file_type(file_path, df_renamed)

        if "vnindex" in [file_type_before, file_type_after]:
            file_type = "vnindex"
        else:
            file_type = "stock"

        audit["file_type"] = file_type

        if file_type == "vnindex":
            df_clean, extra_audit = standardize_vnindex(df_renamed)
        else:
            df_clean, extra_audit = standardize_stock(df_renamed)

        audit.update(extra_audit)

        if df_clean.empty:
            audit["error"] = "No usable rows after standardization."
            return audit

        output_dir.mkdir(parents=True, exist_ok=True)

        output_name = f"{file_path.stem.upper()}_standardized.csv"
        output_path = output_dir / output_name

        df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

        audit["status"] = "SUCCESS"
        audit["clean_rows"] = int(len(df_clean))
        audit["dropped_rows"] = int(audit["raw_rows"] - audit["clean_rows"])
        audit["date_min"] = df_clean["Date"].min()
        audit["date_max"] = df_clean["Date"].max()
        audit["output_file"] = str(output_path)

        return audit

    except Exception as e:
        audit["error"] = str(e)
        return audit


def resolve_input_dir(input_dir: Path | None = None) -> Path:
    if input_dir is not None:
        return Path(input_dir)

    for candidate in FALLBACK_INPUT_DIRS:
        if candidate.exists():
            return candidate

    return DEFAULT_INPUT_DIR


def resolve_output_dir(output_dir: Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)

    return DEFAULT_OUTPUT_DIR


def standardize_all_files(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    input_dir = resolve_input_dir(input_dir)
    output_dir = resolve_output_dir(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Không tìm thấy file CSV trong folder: {input_dir}\n"
            f"Hãy đặt file raw vào: {DEFAULT_INPUT_DIR}"
        )

    audit_records = []

    print("=" * 100)
    print("START DATA STANDARDIZATION")
    print("=" * 100)
    print(f"Input folder : {input_dir}")
    print(f"Output folder: {output_dir}")
    print("=" * 100)

    for file_path in csv_files:
        print(f"[PROCESSING] {file_path.name}")

        audit = standardize_one_file(file_path, output_dir)
        audit_records.append(audit)

        if audit["status"] == "SUCCESS":
            print(
                f"  -> SUCCESS | type: {audit['file_type']} | "
                f"rows: {audit['raw_rows']} -> {audit['clean_rows']} | "
                f"date: {audit['date_min']} -> {audit['date_max']}"
            )
        else:
            print(f"  -> FAILED  | error: {audit['error']}")

            if audit.get("error_detail"):
                print(f"             | detail: {audit['error_detail']}")

    audit_df = pd.DataFrame(audit_records)

    audit_path = output_dir / "standardization_audit.csv"
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print(f"AUDIT SAVED TO: {audit_path}")
    print("=" * 100)

    return audit_df


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Standardize stock and VNINDEX CSV files for quant trading model."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input folder containing raw CSV files. Default: data/unprocessed_data",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output folder for standardized CSV files. Default: data/processed_data",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args.input) if args.input else None
    output_path = Path(args.output) if args.output else None

    standardize_all_files(
        input_dir=input_path,
        output_dir=output_path,
    )