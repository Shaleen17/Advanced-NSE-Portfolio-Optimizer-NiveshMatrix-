"""Multi-provider quote fetching with safe fallback behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from src.providers.settings import get_api_keys, get_configured_provider_names


REQUEST_TIMEOUT_SECONDS = 4


def _safe_float(value: Any) -> float | None:
    """Convert provider values to float when possible."""
    try:
        if value in (None, "", "None", "nan", "-"):
            return None
        return float(str(value).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def _base_symbol(ticker: str) -> str:
    """Return a provider-friendly base ticker symbol."""
    return ticker.upper().replace(".NS", "").strip()


def _now_iso() -> str:
    """Return current UTC timestamp for quote fetch metadata."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _row(
    ticker: str,
    provider: str,
    price: float | None,
    previous_close: float | None = None,
    change: float | None = None,
    change_percent: float | None = None,
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float | None = None,
    timestamp: str | None = None,
    data_type: str = "Near real-time / delayed",
) -> dict[str, Any]:
    """Build a normalized quote row."""
    if change is None and price is not None and previous_close not in (None, 0):
        change = price - previous_close
    if change_percent is None and change is not None and previous_close not in (None, 0):
        change_percent = change / previous_close
    elif change_percent is not None and abs(change_percent) > 1:
        change_percent = change_percent / 100

    return {
        "Ticker": ticker,
        "Provider": provider,
        "Price": price,
        "Previous Close": previous_close,
        "Change": change,
        "Change %": change_percent,
        "Open": open_price,
        "High": high,
        "Low": low,
        "Volume": volume,
        "Timestamp": timestamp or _now_iso(),
        "Data Type": data_type,
    }


def _http_get_json(
    url: str,
    params: dict[str, Any],
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
) -> Any:
    """GET JSON with a short timeout and friendly errors."""
    response = requests.get(url, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def _fetch_twelve_data(tickers: list[str], api_key: str) -> list[dict[str, Any]]:
    """Fetch quotes from Twelve Data."""
    symbol_map = {f"{_base_symbol(ticker)}:NSE": ticker for ticker in tickers}
    data = _http_get_json(
        "https://api.twelvedata.com/quote",
        {"symbol": ",".join(symbol_map), "apikey": api_key},
    )
    quotes = data if isinstance(data, dict) and "symbol" not in data else {"single": data}
    rows = []
    for provider_symbol, payload in quotes.items():
        if not isinstance(payload, dict) or payload.get("status") == "error":
            continue
        ticker = symbol_map.get(provider_symbol) or symbol_map.get(payload.get("symbol", ""))
        if not ticker and isinstance(payload.get("symbol"), str):
            ticker = symbol_map.get(payload["symbol"].replace("/", ":"))
        if not ticker:
            base = str(payload.get("symbol", provider_symbol)).split(":")[0].upper()
            ticker = next((item for item in tickers if _base_symbol(item) == base), None)
        price = _safe_float(payload.get("close"))
        if ticker and price is not None:
            rows.append(
                _row(
                    ticker=ticker,
                    provider="Twelve Data",
                    price=price,
                    previous_close=_safe_float(payload.get("previous_close")),
                    change=_safe_float(payload.get("change")),
                    change_percent=_safe_float(payload.get("percent_change")),
                    open_price=_safe_float(payload.get("open")),
                    high=_safe_float(payload.get("high")),
                    low=_safe_float(payload.get("low")),
                    volume=_safe_float(payload.get("volume")),
                    timestamp=payload.get("datetime") or _now_iso(),
                )
            )
    return rows


def _fetch_fmp(tickers: list[str], api_key: str) -> list[dict[str, Any]]:
    """Fetch quotes from Financial Modeling Prep."""
    data = _http_get_json(
        "https://financialmodelingprep.com/api/v3/quote/" + ",".join(tickers),
        {"apikey": api_key},
    )
    rows = []
    for payload in data if isinstance(data, list) else []:
        ticker = str(payload.get("symbol", "")).upper()
        if ticker not in tickers:
            ticker = next((item for item in tickers if _base_symbol(item) == _base_symbol(ticker)), "")
        price = _safe_float(payload.get("price"))
        if ticker and price is not None:
            ts_value = payload.get("timestamp")
            timestamp = None
            if isinstance(ts_value, (int, float)):
                timestamp = datetime.fromtimestamp(ts_value, timezone.utc).replace(microsecond=0).isoformat()
            rows.append(
                _row(
                    ticker=ticker,
                    provider="Financial Modeling Prep",
                    price=price,
                    previous_close=_safe_float(payload.get("previousClose")),
                    change=_safe_float(payload.get("change")),
                    change_percent=_safe_float(payload.get("changesPercentage")),
                    open_price=_safe_float(payload.get("open")),
                    high=_safe_float(payload.get("dayHigh")),
                    low=_safe_float(payload.get("dayLow")),
                    volume=_safe_float(payload.get("volume")),
                    timestamp=timestamp,
                )
            )
    return rows


def _fetch_finnhub(tickers: list[str], api_key: str) -> list[dict[str, Any]]:
    """Fetch quotes from Finnhub one symbol at a time."""
    rows = []
    for ticker in tickers[:10]:
        payload = _http_get_json(
            "https://finnhub.io/api/v1/quote",
            {"symbol": f"NSE:{_base_symbol(ticker)}", "token": api_key},
        )
        if not isinstance(payload, dict):
            continue
        price = _safe_float(payload.get("c"))
        if price is not None and price > 0:
            ts_value = payload.get("t")
            timestamp = None
            if isinstance(ts_value, (int, float)) and ts_value > 0:
                timestamp = datetime.fromtimestamp(ts_value, timezone.utc).replace(microsecond=0).isoformat()
            rows.append(
                _row(
                    ticker=ticker,
                    provider="Finnhub",
                    price=price,
                    previous_close=_safe_float(payload.get("pc")),
                    open_price=_safe_float(payload.get("o")),
                    high=_safe_float(payload.get("h")),
                    low=_safe_float(payload.get("l")),
                    timestamp=timestamp,
                )
            )
    return rows


def _fetch_marketstack(tickers: list[str], api_key: str) -> list[dict[str, Any]]:
    """Fetch latest EOD quotes from Marketstack."""
    provider_symbols = [f"{_base_symbol(ticker)}.XNSE" for ticker in tickers]
    symbol_map = dict(zip(provider_symbols, tickers))
    data = _http_get_json(
        "http://api.marketstack.com/v1/eod/latest",
        {"access_key": api_key, "symbols": ",".join(provider_symbols)},
        timeout_seconds=10,
    )
    rows = []
    for payload in data.get("data", []) if isinstance(data, dict) else []:
        ticker = symbol_map.get(payload.get("symbol", ""))
        price = _safe_float(payload.get("close"))
        if ticker and price is not None:
            rows.append(
                _row(
                    ticker=ticker,
                    provider="Marketstack",
                    price=price,
                    open_price=_safe_float(payload.get("open")),
                    high=_safe_float(payload.get("high")),
                    low=_safe_float(payload.get("low")),
                    volume=_safe_float(payload.get("volume")),
                    timestamp=payload.get("date"),
                    data_type="Latest EOD",
                )
            )
    return rows


def _fetch_alpha_vantage(tickers: list[str], api_key: str) -> list[dict[str, Any]]:
    """Fetch a small number of Global Quote rows from Alpha Vantage."""
    rows = []
    # Alpha Vantage free quota is tiny, so keep it as a cautious final fallback.
    for ticker in tickers[:3]:
        payload = _http_get_json(
            "https://www.alphavantage.co/query",
            {"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": api_key},
        )
        quote = payload.get("Global Quote", {}) if isinstance(payload, dict) else {}
        price = _safe_float(quote.get("05. price"))
        if price is not None:
            rows.append(
                _row(
                    ticker=ticker,
                    provider="Alpha Vantage",
                    price=price,
                    previous_close=_safe_float(quote.get("08. previous close")),
                    change=_safe_float(quote.get("09. change")),
                    change_percent=_safe_float(quote.get("10. change percent")),
                    open_price=_safe_float(quote.get("02. open")),
                    high=_safe_float(quote.get("03. high")),
                    low=_safe_float(quote.get("04. low")),
                    volume=_safe_float(quote.get("06. volume")),
                    timestamp=quote.get("07. latest trading day"),
                )
            )
    return rows


def fetch_live_quotes(tickers: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Fetch quotes using configured providers with fallback for missing symbols."""
    unique_tickers = []
    for ticker in tickers:
        clean = str(ticker).strip().upper()
        if clean and clean not in unique_tickers:
            unique_tickers.append(clean)

    if not unique_tickers:
        return pd.DataFrame()

    keys = get_api_keys()
    provider_plan = [
        ("Twelve Data", keys.twelve_data, _fetch_twelve_data),
        ("Financial Modeling Prep", keys.fmp, _fetch_fmp),
        ("Marketstack", keys.marketstack, _fetch_marketstack),
        ("Finnhub", keys.finnhub, _fetch_finnhub),
        ("Alpha Vantage", keys.alpha_vantage, _fetch_alpha_vantage),
    ]

    rows: list[dict[str, Any]] = []
    covered: set[str] = set()
    errors: list[str] = []

    for provider_name, api_key, fetcher in provider_plan:
        missing = [ticker for ticker in unique_tickers if ticker not in covered]
        if not missing:
            break
        if not api_key:
            continue
        try:
            provider_rows = fetcher(missing, api_key)
        except Exception as error:
            errors.append(f"{provider_name}: {error.__class__.__name__}")
            continue
        for row in provider_rows:
            ticker = row.get("Ticker")
            if ticker and ticker not in covered:
                rows.append(row)
                covered.add(ticker)

    for ticker in unique_tickers:
        if ticker not in covered:
            rows.append(
                {
                    "Ticker": ticker,
                    "Provider": "Unavailable",
                    "Price": None,
                    "Previous Close": None,
                    "Change": None,
                    "Change %": None,
                    "Open": None,
                    "High": None,
                    "Low": None,
                    "Volume": None,
                    "Timestamp": _now_iso(),
                    "Data Type": "No provider match",
                    "Provider Notes": "; ".join(errors[-3:]),
                }
            )

    frame = pd.DataFrame(rows)
    order = {ticker: index for index, ticker in enumerate(unique_tickers)}
    frame["_order"] = frame["Ticker"].map(order)
    frame = frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    if "Provider Notes" not in frame.columns:
        frame["Provider Notes"] = ""
    return frame
