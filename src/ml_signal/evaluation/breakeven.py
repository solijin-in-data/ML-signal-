from __future__ import annotations


def calculate_breakeven_precision(
    tp: float,
    sl: float,
    round_trip_cost: float = 0.0,
) -> float:
    """
    Calculate the minimum win probability needed for zero expected return.

    Assumption:
    - winning trade return = tp - round_trip_cost
    - losing trade return = sl - round_trip_cost

    With sl as a negative number, the breakeven precision is:

        (round_trip_cost - sl) / (tp - sl)

    Example:
        TP = 10%, SL = -5%, cost = 0.5%
        breakeven = (0.005 - (-0.05)) / (0.10 - (-0.05)) = 36.67%
    """
    if tp <= 0:
        raise ValueError("tp must be positive.")

    if sl >= 0:
        raise ValueError("sl must be negative.")

    denominator = tp - sl

    if denominator <= 0:
        raise ValueError("tp - sl must be positive.")

    return float((round_trip_cost - sl) / denominator)


__all__ = [
    "calculate_breakeven_precision",
]
