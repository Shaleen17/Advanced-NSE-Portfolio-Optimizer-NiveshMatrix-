"""Rule-based live market signal and alert helpers.

The functions in this module intentionally use transparent rules instead of
external paid APIs or opaque models. They convert normalized live quote rows
into explainable intraday metrics, breadth statistics, signals, and alerts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "Last Price",
    "Previous Close",
    "Open",
    "High",
    "Low",
    "Volume",
    "Change",
    "Change %",
]


def _coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with expected quote columns converted safely to numbers."""
    result = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "Ticker" not in result.columns:
        result["Ticker"] = ""
    return result


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Vectorized division that returns NaN for missing or zero denominators."""
    valid = denominator.notna() & (denominator != 0)
    output = pd.Series(np.nan, index=numerator.index, dtype="float64")
    output.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return output


def _safe_float(value: Any) -> float | None:
    """Convert a scalar to float or None for rule checks."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_intraday_metrics(live_df: pd.DataFrame) -> pd.DataFrame:
    """Add explainable intraday metrics used by signal rules.

    Added columns:
    - Intraday return %: current price versus today's open.
    - Gap from previous close %: current price versus previous close.
    - Distance from day high %: current price relative to high, usually <= 0.
    - Distance from day low %: current price relative to low, usually >= 0.
    - Day range %: high-low range relative to previous close, open, or price.
    - Volume rank: rank 1 is the highest volume row.
    - Volatility proxy: same-day range percent, used as a simple risk proxy.
    """
    frame = _coerce_numeric_columns(live_df)

    frame["Intraday return %"] = _safe_divide(
        frame["Last Price"] - frame["Open"],
        frame["Open"],
    )
    frame["Gap from previous close %"] = _safe_divide(
        frame["Last Price"] - frame["Previous Close"],
        frame["Previous Close"],
    )
    frame["Distance from day high %"] = _safe_divide(
        frame["Last Price"] - frame["High"],
        frame["High"],
    )
    frame["Distance from day low %"] = _safe_divide(
        frame["Last Price"] - frame["Low"],
        frame["Low"],
    )

    range_denominator = frame["Previous Close"].where(
        frame["Previous Close"].notna() & (frame["Previous Close"] != 0)
    )
    open_denominator = frame["Open"].where(frame["Open"].notna() & (frame["Open"] != 0))
    price_denominator = frame["Last Price"].where(
        frame["Last Price"].notna() & (frame["Last Price"] != 0)
    )
    range_denominator = range_denominator.fillna(open_denominator).fillna(price_denominator)
    frame["Day range %"] = _safe_divide(frame["High"] - frame["Low"], range_denominator)
    frame["Day Range %"] = frame["Day range %"]
    frame["Volatility proxy"] = frame["Day range %"]

    frame["Volume rank"] = frame["Volume"].rank(
        ascending=False,
        method="dense",
        na_option="bottom",
    )
    valid_volume = frame["Volume"].dropna()
    median_volume = valid_volume.median() if not valid_volume.empty else np.nan
    frame["Volume above median"] = np.where(
        frame["Volume"].notna() & pd.notna(median_volume),
        frame["Volume"] >= median_volume,
        False,
    )
    return frame


def generate_live_signal(row: pd.Series) -> str:
    """Generate one explainable signal label for a live quote row.

    Rule notes:
    - Strong buy/sell requires a large move, price near the day extreme, and
      above-median volume confirmation.
    - Buy/weakness is a softer momentum rule with either day-extreme proximity
      or volume confirmation.
    - Watchlist catches unusual volume with quiet price action.
    """
    last_price = _safe_float(row.get("Last Price"))
    change_pct = _safe_float(row.get("Change %"))
    if last_price is None or change_pct is None:
        return "Insufficient Data"

    distance_high = _safe_float(row.get("Distance from day high %"))
    distance_low = _safe_float(row.get("Distance from day low %"))
    near_day_high = distance_high is not None and distance_high >= -0.01
    near_day_low = distance_low is not None and distance_low <= 0.01
    high_volume = bool(row.get("Volume above median", False))
    quiet_move = abs(change_pct) <= 0.005

    if change_pct > 0.015 and near_day_high and high_volume:
        return "Strong Buy Momentum"
    if change_pct > 0.005 and (near_day_high or high_volume):
        return "Buy Momentum"
    if change_pct < -0.015 and near_day_low and high_volume:
        return "Strong Sell Pressure"
    if change_pct < -0.005 and (near_day_low or high_volume):
        return "Weakness"
    if high_volume and quiet_move:
        return "Watchlist"
    return "No Signal"


def generate_market_breadth(live_df: pd.DataFrame) -> dict[str, float | int]:
    """Summarize market breadth from current live quote returns."""
    frame = calculate_intraday_metrics(live_df)
    returns = pd.to_numeric(frame["Change %"], errors="coerce").dropna()
    advancers = int((returns > 0).sum())
    decliners = int((returns < 0).sum())
    unchanged = int((returns == 0).sum())
    total = int(len(returns))

    if decliners == 0:
        advance_decline_ratio = float("inf") if advancers > 0 else np.nan
    else:
        advance_decline_ratio = advancers / decliners

    return {
        "Advancers": advancers,
        "Decliners": decliners,
        "Unchanged": unchanged,
        "Advance-decline ratio": advance_decline_ratio,
        "Average return": float(returns.mean()) if total else np.nan,
        "Median return": float(returns.median()) if total else np.nan,
        "Positive breadth %": advancers / total if total else np.nan,
    }


def generate_alerts(live_df: pd.DataFrame) -> list[dict[str, str]]:
    """Create explainable alert feed messages from live quote rows."""
    frame = calculate_intraday_metrics(live_df)
    if "Live Signal" not in frame.columns:
        frame["Live Signal"] = frame.apply(generate_live_signal, axis=1)

    alerts: list[dict[str, str]] = []
    breadth = generate_market_breadth(frame)
    positive_breadth = _safe_float(breadth["Positive breadth %"])
    if positive_breadth is not None:
        declining_breadth = 1 - positive_breadth
        if declining_breadth >= 0.70:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Market breadth is weak: {declining_breadth:.0%} stocks declining",
                }
            )
        elif positive_breadth >= 0.70:
            alerts.append(
                {
                    "level": "success",
                    "message": f"Market breadth is strong: {positive_breadth:.0%} stocks advancing",
                }
            )

    for _, row in frame.iterrows():
        ticker = str(row.get("Ticker", "Unknown"))
        change_pct = _safe_float(row.get("Change %"))
        high_volume = bool(row.get("Volume above median", False))
        distance_low = _safe_float(row.get("Distance from day low %"))
        distance_high = _safe_float(row.get("Distance from day high %"))
        live_signal = str(row.get("Live Signal", ""))

        if live_signal == "Insufficient Data":
            alerts.append({"level": "warning", "message": f"{ticker} has missing live data"})
            continue
        if change_pct is not None and change_pct >= 0.015 and high_volume:
            alerts.append(
                {
                    "level": "success",
                    "message": f"{ticker} is up {change_pct:.1%} with high volume",
                }
            )
        if change_pct is not None and change_pct <= -0.015 and high_volume:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"{ticker} is down {abs(change_pct):.1%} with high volume",
                }
            )
        if distance_low is not None and distance_low <= 0.01:
            alerts.append({"level": "warning", "message": f"{ticker} is trading near day low"})
        elif distance_high is not None and distance_high >= -0.01:
            alerts.append({"level": "success", "message": f"{ticker} is trading near day high"})

    return alerts[:12]
