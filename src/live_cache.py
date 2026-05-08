"""Persistent local cache for live quote snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from config import LIVE_QUOTES_CACHE_FILE


@dataclass
class LiveQuoteCache:
    """Loaded cache payload plus freshness metadata."""

    dataframe: pd.DataFrame
    timestamp: str
    age_seconds: float | None
    error: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _cache_age_seconds(timestamp: str) -> float | None:
    try:
        saved_at = pd.to_datetime(timestamp, utc=True)
        if pd.isna(saved_at):
            return None
        return max((datetime.now(timezone.utc) - saved_at.to_pydatetime()).total_seconds(), 0.0)
    except (TypeError, ValueError):
        return None


def save_live_quote_cache(dataframe: pd.DataFrame) -> str:
    """Persist the latest successful quote snapshot to disk as CSV."""
    LIVE_QUOTES_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    saved_at = _now_iso()
    cache_frame = dataframe.copy()
    cache_frame["Cache Timestamp"] = saved_at
    cache_frame.to_csv(LIVE_QUOTES_CACHE_FILE, index=False)
    return saved_at


def load_live_quote_cache() -> LiveQuoteCache:
    """Load the last persisted quote snapshot, returning an empty frame on failure."""
    if not LIVE_QUOTES_CACHE_FILE.exists():
        return LiveQuoteCache(pd.DataFrame(), "", None, "No live quote cache exists yet.")
    try:
        frame = pd.read_csv(LIVE_QUOTES_CACHE_FILE)
    except Exception as error:
        return LiveQuoteCache(pd.DataFrame(), "", None, f"Could not read live quote cache: {error}")

    timestamp = ""
    if "Cache Timestamp" in frame.columns:
        timestamps = frame["Cache Timestamp"].dropna().astype(str)
        timestamp = timestamps.iloc[0] if not timestamps.empty else ""
        frame = frame.drop(columns=["Cache Timestamp"])
    if not timestamp:
        timestamp = _now_iso()
    return LiveQuoteCache(frame, timestamp, _cache_age_seconds(timestamp))


def mark_stale_cache_frame(dataframe: pd.DataFrame, cache_timestamp: str) -> pd.DataFrame:
    """Mark a cached quote frame clearly as stale fallback data."""
    frame = dataframe.copy()
    if "Source" not in frame.columns:
        frame["Source"] = "Stale local cache"
    else:
        frame["Source"] = frame["Source"].fillna("Stale local cache")
        frame["Source"] = "Stale local cache (" + frame["Source"].astype(str) + ")"
    frame["Status"] = "stale_cache"
    frame["Timestamp"] = cache_timestamp
    frame["Error"] = "Loaded from persistent local cache after provider failure."
    return frame
