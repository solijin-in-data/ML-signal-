from __future__ import annotations


def get_valuation_feature_columns() -> list[str]:
    return [
        "PB",
        "Book_to_Market",
        "Log_PB",
        "PB_Z_756",
        "PB_Percentile_756",
    ]


def get_extended_valuation_columns() -> list[str]:
    return [
        "MarketCap_TyDong",
        "Book_Equity_Parent_TyDong",
    ]
