"""Live market orchestration, diagnostics, and dataframe preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.live_cache import (
    load_live_quote_cache,
    mark_stale_cache_frame,
    save_live_quote_cache,
)
from src.providers import LiveQuoteResult, fetch_live_quotes


@dataclass
class LiveMarketDiagnostics:
    """Operational diagnostics for the Live Market command center."""

    provider_used: str = "Unavailable"
    api_response_time: float | None = None
    cache_age_seconds: float | None = None
    symbols_requested: int = 0
    symbols_returned: int = 0
    missing_price_count: int = 0
    failed_symbols: list[str] = field(default_factory=list)
    last_successful_refresh: str = ""
    current_fallback_mode: str = "None"
    cache_timestamp: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class LiveMarketSnapshot:
    """Live quote payload plus metadata used by Streamlit UI."""

    dataframe: pd.DataFrame
    source: str
    status: str
    timestamp: str
    errors: list[str]
    failed_symbols: list[str]
    diagnostics: LiveMarketDiagnostics
    stale_cache_used: bool = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _empty_result(tickers: list[str], message: str) -> LiveMarketSnapshot:
    frame = pd.DataFrame({"Ticker": tickers})
    diagnostics = LiveMarketDiagnostics(
        symbols_requested=len(tickers),
        symbols_returned=0,
        missing_price_count=len(tickers),
        failed_symbols=tickers,
        errors=[message],
    )
    return LiveMarketSnapshot(
        dataframe=frame,
        source="Unavailable",
        status="failed",
        timestamp=_now_iso(),
        errors=[message],
        failed_symbols=tickers,
        diagnostics=diagnostics,
    )


def validate_live_quote_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize missing columns and numeric dtypes so downstream UI cannot crash."""
    required_columns = [
        "Ticker",
        "Last Price",
        "Previous Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "Change",
        "Change %",
        "Source",
        "Timestamp",
        "Status",
        "Error",
    ]
    frame = dataframe.copy()
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    for column in [
        "Last Price",
        "Previous Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "Change",
        "Change %",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Source"] = frame["Source"].fillna("Unavailable").replace("", "Unavailable")
    frame["Status"] = frame["Status"].fillna("failed").replace("", "failed")
    frame["Error"] = frame["Error"].fillna("")
    return frame[required_columns]


def prepare_live_market_frame(live_quotes: pd.DataFrame) -> pd.DataFrame:
    """Add terminal metrics, signals, and data-quality labels to live quotes."""
    frame = validate_live_quote_frame(live_quotes)
    range_denominator = frame["Previous Close"].where(
        frame["Previous Close"].notna() & (frame["Previous Close"] != 0)
    )
    open_denominator = frame["Open"].where(frame["Open"].notna() & (frame["Open"] != 0))
    last_price_denominator = frame["Last Price"].where(
        frame["Last Price"].notna() & (frame["Last Price"] != 0)
    )
    range_denominator = range_denominator.fillna(open_denominator).fillna(last_price_denominator)
    frame["Day Range %"] = np.where(
        frame["High"].notna()
        & frame["Low"].notna()
        & range_denominator.notna()
        & (range_denominator != 0),
        (frame["High"] - frame["Low"]) / range_denominator,
        np.nan,
    )

    def quote_signal(value) -> str:
        if pd.isna(value):
            return "— N/A"
        if value > 0:
            return "▲ Gainer"
        if value < 0:
            return "▼ Loser"
        return "— Flat"

    def data_quality(row: pd.Series) -> str:
        if pd.isna(row.get("Last Price")) or row.get("Status") == "failed":
            return "Unavailable"
        status = str(row.get("Status", "")).lower()
        source = str(row.get("Source", "")).lower()
        if status == "stale_cache":
            return "Stale Cache"
        if status == "cached" or "cache" in source:
            return "Cached"
        if status == "fallback" or "fallback" in source:
            return "Fallback"
        return "Live"

    frame["Signal"] = frame["Change %"].apply(quote_signal)
    frame["Data Quality"] = frame.apply(data_quality, axis=1)
    frame["Quote Available"] = frame["Last Price"].notna()
    return frame


def filter_live_market_frame(
    frame: pd.DataFrame,
    search_query: str,
    move_filter: str,
    minimum_volume: int,
    sort_option: str,
    holdings_only: bool,
    selected_tickers: list[str],
) -> pd.DataFrame:
    """Apply tab-level search, filters, and sort controls to quote rows."""
    filtered = frame.copy()
    if holdings_only:
        filtered = filtered[filtered["Ticker"].isin(selected_tickers)]
    search_query = search_query.strip().upper()
    if search_query:
        filtered = filtered[
            filtered["Ticker"].astype(str).str.upper().str.contains(search_query, regex=False, na=False)
        ]
    if move_filter == "Gainers":
        filtered = filtered[filtered["Change %"] > 0]
    elif move_filter == "Losers":
        filtered = filtered[filtered["Change %"] < 0]
    if minimum_volume > 0:
        filtered = filtered[filtered["Volume"].fillna(0) >= minimum_volume]

    sort_map = {
        "Change %": ("Change %", False),
        "Volume": ("Volume", False),
        "Volatility": ("Day Range %", False),
        "Ticker": ("Ticker", True),
    }
    sort_column, ascending = sort_map.get(sort_option, ("Change %", False))
    return filtered.sort_values(sort_column, ascending=ascending, na_position="last")


def diagnostics_from_result(
    result: LiveQuoteResult,
    tickers: list[str],
    response_time: float,
    cache_age_seconds: float | None,
    cache_timestamp: str,
    fallback_mode: str,
) -> LiveMarketDiagnostics:
    """Build diagnostics from a provider result and cache state."""
    frame = validate_live_quote_frame(result.dataframe)
    missing_price_count = int(frame["Last Price"].isna().sum())
    returned_count = int(frame["Last Price"].notna().sum())
    return LiveMarketDiagnostics(
        provider_used=result.source or "Unavailable",
        api_response_time=response_time,
        cache_age_seconds=cache_age_seconds,
        symbols_requested=len(tickers),
        symbols_returned=returned_count,
        missing_price_count=missing_price_count,
        failed_symbols=result.failed_symbols,
        last_successful_refresh=cache_timestamp,
        current_fallback_mode=fallback_mode,
        cache_timestamp=cache_timestamp,
        errors=result.errors,
    )


def fetch_live_market_snapshot(
    tickers: list[str] | tuple[str, ...],
    historical_last_prices: dict[str, Any] | None = None,
    historical_previous_closes: dict[str, Any] | None = None,
    provider_preference: str = "Auto",
    use_fallback_cache: bool = True,
) -> LiveMarketSnapshot:
    """Fetch live quotes, persist good snapshots, and fallback to stale cache on failure."""
    unique_tickers = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
    if not unique_tickers:
        return _empty_result([], "No symbols selected.")

    cache_payload = load_live_quote_cache()
    started = perf_counter()
    result = fetch_live_quotes(
        unique_tickers,
        historical_last_prices=historical_last_prices,
        historical_previous_closes=historical_previous_closes,
        provider_preference=provider_preference,
    )
    response_time = perf_counter() - started
    frame = validate_live_quote_frame(result.dataframe)

    returned_count = int(frame["Last Price"].notna().sum())
    fallback_mode = result.source if result.status in {"cached", "fallback"} else "None"
    cache_timestamp = cache_payload.timestamp
    cache_age = cache_payload.age_seconds

    has_fresh_provider_rows = bool(frame["Status"].astype(str).eq("success").any())
    if returned_count > 0 and has_fresh_provider_rows:
        cache_timestamp = save_live_quote_cache(frame)
        cache_age = 0.0
    elif use_fallback_cache and not cache_payload.dataframe.empty:
        stale_frame = mark_stale_cache_frame(cache_payload.dataframe, cache_payload.timestamp)
        stale_frame = validate_live_quote_frame(stale_frame)
        stale_frame = stale_frame[stale_frame["Ticker"].isin(unique_tickers)]
        if not stale_frame.empty:
            fallback_mode = "Stale persistent cache"
            diagnostics = diagnostics_from_result(
                result,
                unique_tickers,
                response_time,
                cache_payload.age_seconds,
                cache_payload.timestamp,
                fallback_mode,
            )
            diagnostics.symbols_returned = int(stale_frame["Last Price"].notna().sum())
            diagnostics.missing_price_count = int(stale_frame["Last Price"].isna().sum())
            diagnostics.failed_symbols = result.failed_symbols
            diagnostics.errors = result.errors + ["Loaded stale local cache after provider failure."]
            return LiveMarketSnapshot(
                dataframe=stale_frame,
                source="Stale local cache",
                status="stale_cache",
                timestamp=cache_payload.timestamp,
                errors=diagnostics.errors,
                failed_symbols=result.failed_symbols,
                diagnostics=diagnostics,
                stale_cache_used=True,
            )
    diagnostics = diagnostics_from_result(
        result,
        unique_tickers,
        response_time,
        cache_age,
        cache_timestamp,
        fallback_mode,
    )
    return LiveMarketSnapshot(
        dataframe=frame,
        source=result.source,
        status=result.status,
        timestamp=result.timestamp,
        errors=result.errors,
        failed_symbols=result.failed_symbols,
        diagnostics=diagnostics,
    )
