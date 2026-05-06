"""Safe API key loading for external market-data providers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class ApiKeys:
    """API keys configured for optional market-data providers."""

    twelve_data: str | None = None
    fmp: str | None = None
    alpha_vantage: str | None = None
    marketstack: str | None = None
    finnhub: str | None = None


def _secret_value(*keys: str) -> str | None:
    """Read nested Streamlit secrets without raising when missing."""
    current = st.secrets
    try:
        for key in keys:
            current = current[key]
        value = str(current).strip()
        return value or None
    except Exception:
        return None


def _env_value(name: str) -> str | None:
    """Read an environment value as a clean optional string."""
    value = os.getenv(name, "").strip()
    return value or None


def get_api_keys() -> ApiKeys:
    """Return optional API keys from Streamlit secrets or environment variables."""
    return ApiKeys(
        twelve_data=_secret_value("api_keys", "twelve_data") or _env_value("TWELVE_DATA_API_KEY"),
        fmp=_secret_value("api_keys", "fmp") or _env_value("FMP_API_KEY"),
        alpha_vantage=_secret_value("api_keys", "alpha_vantage") or _env_value("ALPHA_VANTAGE_API_KEY"),
        marketstack=_secret_value("api_keys", "marketstack") or _env_value("MARKETSTACK_API_KEY"),
        finnhub=_secret_value("api_keys", "finnhub") or _env_value("FINNHUB_API_KEY"),
    )


def get_configured_provider_names() -> list[str]:
    """Return provider names that have keys configured."""
    keys = get_api_keys()
    names = []
    if keys.twelve_data:
        names.append("Twelve Data")
    if keys.fmp:
        names.append("Financial Modeling Prep")
    if keys.marketstack:
        names.append("Marketstack")
    if keys.finnhub:
        names.append("Finnhub")
    if keys.alpha_vantage:
        names.append("Alpha Vantage")
    return names
