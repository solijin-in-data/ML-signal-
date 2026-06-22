from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from data_vendors.base import VendorFetchResult, ensure_parent_dir


logger = logging.getLogger(__name__)


class SStockClient:
    """
    SStock price-history connector.

    Do not hard-code cookies or session tokens in this file.
    Put your own authenticated cookie in the SSTOCK_COOKIE environment variable.
    """

    BASE_URL = "https://api-feature.sstock.vn/api/v1/prices/history"
    VENDOR_NAME = "sstock"

    def __init__(
        self,
        cookie: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
        sleep_seconds: float = 0.7,
    ) -> None:
        self.cookie = cookie or os.getenv("SSTOCK_COOKIE", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds

        self.session = requests.Session()
        self.session.headers.update(self._build_headers())

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://sstock.vn/",
            "Origin": "https://sstock.vn",
        }

        if self.cookie:
            headers["Cookie"] = self.cookie

        return headers

    def fetch_price_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> VendorFetchResult:
        symbol = symbol.upper().strip()

        params = {
            "symbol": symbol,
            "from": start_date,
            "to": end_date,
        }

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Fetching SStock price history | symbol=%s | from=%s | to=%s | attempt=%d",
                    symbol,
                    start_date,
                    end_date,
                    attempt,
                )

                response = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    data = self._extract_data_list(response.json())
                    df = self._standardize_price_history(data, symbol)

                    metadata = {
                        "vendor": self.VENDOR_NAME,
                        "symbol": symbol,
                        "start_date": start_date,
                        "end_date": end_date,
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "source_url": response.url,
                        "rows": int(len(df)),
                    }

                    return VendorFetchResult(
                        vendor=self.VENDOR_NAME,
                        symbol=symbol,
                        dataframe=df,
                        metadata=metadata,
                    )

                if response.status_code in [401, 403]:
                    raise PermissionError(
                        "SStock authentication failed. Refresh your own browser cookie "
                        "and set it in the SSTOCK_COOKIE environment variable."
                    )

                response.raise_for_status()

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "SStock fetch failed | symbol=%s | attempt=%d | error=%s",
                    symbol,
                    attempt,
                    exc,
                )

                if attempt < self.max_retries:
                    time.sleep(self.sleep_seconds * attempt)

        raise RuntimeError(
            f"SStock fetch failed after {self.max_retries} attempts for {symbol}."
        ) from last_error

    @staticmethod
    def _extract_data_list(json_data: Any) -> list[dict[str, Any]]:
        if isinstance(json_data, list):
            return json_data

        if isinstance(json_data, dict):
            for key in ["data", "items", "result", "prices"]:
                value = json_data.get(key)
                if isinstance(value, list):
                    return value

            data = json_data.get("data")
            if isinstance(data, dict):
                for key in ["items", "result", "prices"]:
                    value = data.get(key)
                    if isinstance(value, list):
                        return value

        raise ValueError("Unexpected SStock JSON response format.")

    @staticmethod
    def _standardize_price_history(
        data: list[dict[str, Any]],
        symbol: str,
    ) -> pd.DataFrame:
        df = pd.DataFrame(data)

        output_cols = [
            "Date",
            "Symbol",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "MarketCap",
            "MarketCap_TyDong",
            "Vendor",
        ]

        if df.empty:
            return pd.DataFrame(columns=output_cols)

        rename_map = {
            "date": "Date",
            "symbol": "Symbol",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "marketCap": "MarketCap",
            "market_cap": "MarketCap",
            "marketCapitalization": "MarketCap",
        }

        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if "Date" not in df.columns:
            raise ValueError("SStock response does not contain a date column.")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Symbol"] = symbol

        for col in ["Open", "High", "Low", "Close", "Volume", "MarketCap"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "MarketCap" in df.columns:
            df["MarketCap_TyDong"] = df["MarketCap"] / 1_000_000_000
        else:
            df["MarketCap"] = pd.NA
            df["MarketCap_TyDong"] = pd.NA

        df["Vendor"] = SStockClient.VENDOR_NAME

        for col in output_cols:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[output_cols].copy()
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date")
        df = df.drop_duplicates(subset=["Date", "Symbol"], keep="last")
        df = df.reset_index(drop=True)

        return df

    @staticmethod
    def save_result(
        result: VendorFetchResult,
        output_path: str | Path,
        file_format: str = "csv",
    ) -> Path:
        output_path = ensure_parent_dir(output_path)
        df = result.dataframe.copy()

        if file_format.lower() == "csv":
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        elif file_format.lower() in ["xlsx", "excel"]:
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {file_format}")

        logger.info(
            "SStock data saved | symbol=%s | path=%s | rows=%d",
            result.symbol,
            output_path,
            len(df),
        )

        return output_path
