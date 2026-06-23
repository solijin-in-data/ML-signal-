from __future__ import annotations

from ml_signal import core_compat as radar
from ml_signal.features.valuation import get_valuation_feature_columns
from ml_signal.features.liquidity import LIQUIDITY_COST_COLUMNS
from ml_signal.features.recovery import RECOVERY_QUALITY_COLUMNS
from ml_signal.features.noise import NOISE_FILTER_COLUMNS


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    output = []

    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)

    return output


def subtract_columns(source_cols: list[str], remove_cols: list[str]) -> list[str]:
    remove_set = set(remove_cols)
    return [col for col in source_cols if col not in remove_set]


def get_feature_sets() -> dict[str, list[str]]:
    # Central feature-set registry.
    # This registry is ticker-agnostic. Sector-specific feature sets can be
    # added here later without changing the experiment runner.
    baseline = radar.get_feature_columns()

    feature_sets: dict[str, list[str]] = {
        "baseline": baseline,

        "candidate_momentum_v1": baseline + [
            "Abs_Momentum_14",
            "Log_Momentum_14",
            "EWM_Return_10",
            "Momentum_Dropoff_Z_14",
        ],

        "candidate_trend_quality_v1": baseline + [
            "ER_10",
            "EMA_13_Slope",
            "EMA_21_55_Gap",
            "BB_Position",
        ],

        "candidate_volume_v1": baseline + [
            "Volume_Ratio_20",
            "Log_Volume_Change",
            "Foreign_Net_5D_Ratio",
        ],

        "candidate_recovery_v1": baseline + [
            "Drawdown_60",
            "Distance_52W_High",
            "Distance_52W_Low",
        ],

        "candidate_light_combo_v1": baseline + [
            "Log_Momentum_14",
            "EWM_Return_10",
            "Momentum_Dropoff_Z_14",
            "ER_10",
            "Volume_Ratio_20",
            "Drawdown_60",
            "Distance_52W_High",
        ],
    }

    valuation_cols = get_valuation_feature_columns()

    feature_sets.update(
        {
            "candidate_valuation_v1": baseline + valuation_cols,

            "candidate_valuation_plus_trend_v1": baseline + valuation_cols + [
                "ER_10",
                "Volume_Ratio_20",
                "Drawdown_60",
                "Distance_52W_High",
            ],
        }
    )

    recovery_cols = feature_sets.get("candidate_recovery_v1", baseline)
    trend_quality_cols = feature_sets.get("candidate_trend_quality_v1", baseline)

    feature_sets.update(
        {
            "candidate_recovery_valuation_v1": unique_preserve_order(
                recovery_cols + valuation_cols
            ),

            "candidate_trend_recovery_v1": unique_preserve_order(
                trend_quality_cols
                + [col for col in recovery_cols if col not in baseline]
            ),
        }
    )

    # -------------------------------------------------------------------------
    # Ablation feature sets
    # -------------------------------------------------------------------------
    trend_cols = feature_sets.get("candidate_trend_quality_v1", baseline)
    recovery_cols = feature_sets.get("candidate_recovery_v1", baseline)

    trend_extra_cols = [col for col in trend_cols if col not in baseline]
    recovery_extra_cols = [col for col in recovery_cols if col not in baseline]

    trend_recovery_cols = feature_sets.get(
        "candidate_trend_recovery_v1",
        unique_preserve_order(
            trend_cols + [col for col in recovery_cols if col not in baseline]
        ),
    )

    recovery_valuation_cols = feature_sets.get(
        "candidate_recovery_valuation_v1",
        unique_preserve_order(recovery_cols + valuation_cols),
    )

    rsi_cols = [
        col for col in trend_recovery_cols
        if "RSI" in col.upper()
    ]

    drawdown_recovery_name_cols = [
        col for col in trend_recovery_cols
        if any(
            token in col.upper()
            for token in [
                "DRAWDOWN",
                "RECOVERY",
                "52W",
                "LOW",
                "HIGH",
                "DISTANCE",
            ]
        )
    ]

    market_regime_name_cols = [
        col for col in trend_recovery_cols
        if any(
            token in col.upper()
            for token in [
                "VN_",
                "VNINDEX",
                "MARKET",
                "INDEX",
                "RELATIVE",
                "RS_VN",
            ]
        )
    ]

    feature_sets.update(
        {
            "ablation_trend_recovery_full_v1": trend_recovery_cols,

            "ablation_trend_recovery_minus_trend_v1": unique_preserve_order(
                subtract_columns(trend_recovery_cols, trend_extra_cols)
            ),

            "ablation_trend_recovery_minus_recovery_v1": unique_preserve_order(
                subtract_columns(trend_recovery_cols, recovery_extra_cols)
            ),

            "ablation_trend_recovery_minus_rsi_v1": unique_preserve_order(
                subtract_columns(trend_recovery_cols, rsi_cols)
            ),

            "ablation_trend_recovery_minus_drawdown_recovery_v1": unique_preserve_order(
                subtract_columns(trend_recovery_cols, drawdown_recovery_name_cols)
            ),

            "ablation_trend_recovery_minus_market_regime_v1": unique_preserve_order(
                subtract_columns(trend_recovery_cols, market_regime_name_cols)
            ),

            "ablation_recovery_valuation_full_v1": recovery_valuation_cols,

            "ablation_recovery_valuation_minus_recovery_v1": unique_preserve_order(
                subtract_columns(recovery_valuation_cols, recovery_extra_cols)
            ),

            "ablation_recovery_valuation_minus_valuation_v1": unique_preserve_order(
                subtract_columns(recovery_valuation_cols, valuation_cols)
            ),
        }
    )


    # -------------------------------------------------------------------------
    # Cost-resilient recovery feature sets
    # -------------------------------------------------------------------------
    # These are designed to test whether the CTD recovery setup remains robust
    # after round-trip cost/slippage assumptions are added.
    base_cost_resilient_cols = feature_sets.get(
        "ablation_trend_recovery_minus_market_regime_v1",
        feature_sets.get("candidate_trend_recovery_v1", baseline),
    )

    cost_resilient_cols = unique_preserve_order(
        base_cost_resilient_cols
        + LIQUIDITY_COST_COLUMNS
        + RECOVERY_QUALITY_COLUMNS
        + NOISE_FILTER_COLUMNS
    )

    feature_sets.update(
        {
            "candidate_cost_resilient_recovery_v1": cost_resilient_cols,

            "ablation_cost_resilient_recovery_full_v1": cost_resilient_cols,

            "candidate_cost_resilient_recovery_no_noise_v1": unique_preserve_order(
                subtract_columns(cost_resilient_cols, NOISE_FILTER_COLUMNS)
            ),

            "ablation_cost_resilient_recovery_minus_liquidity_v1": unique_preserve_order(
                subtract_columns(cost_resilient_cols, LIQUIDITY_COST_COLUMNS)
            ),

            "ablation_cost_resilient_recovery_minus_recovery_quality_v1": unique_preserve_order(
                subtract_columns(cost_resilient_cols, RECOVERY_QUALITY_COLUMNS)
            ),

            "ablation_cost_resilient_recovery_minus_noise_filter_v1": unique_preserve_order(
                subtract_columns(cost_resilient_cols, NOISE_FILTER_COLUMNS)
            ),
        }
    )

    return {
        name: unique_preserve_order(cols)
        for name, cols in feature_sets.items()
    }


__all__ = [
    "unique_preserve_order",
    "subtract_columns",
    "get_feature_sets",
]
