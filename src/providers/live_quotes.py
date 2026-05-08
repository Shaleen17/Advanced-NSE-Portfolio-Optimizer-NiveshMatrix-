"""Production-grade multi-provider quote fetching with safe fallbacks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd
import requests
import yfinance as yf

from config import CLEANED_PRICE_FILE
from src.providers.settings import (
    get_api_keys,
    get_configured_provider_names as get_configured_api_provider_names,
)


REQUEST_TIMEOUT_SECONDS = 4
MAX_RETRIES = 2
SECRET_QUERY_RE = re.compile(r"(?i)([?&](?:apikey|api_key|access_key|token)=)[^&\s)]+")
STANDARD_COLUMNS = [
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


@dataclass
class LiveQuoteResult:
    """Structured result returned by the live market provider layer."""

    dataframe: pd.DataFrame
    source: str
    status: str
    timestamp: str
    errors: list[str] = field(default_factory=list)
    failed_symbols: list[str] = field(default_factory=list)


def _safe_float(value: Any) -> float | None:
    """Convert provider values to float when possible."""
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() in {"", "None", "nan", "-", "N/A"}:
            return None
        if pd.isna(value):
            return None
        converted = float(str(value).replace("%", "").replace(",", ""))
        if pd.isna(converted):
            return None
        return converted
    except (TypeError, ValueError):
        return None


def _base_symbol(ticker: str) -> str:
    """Return a provider-friendly base ticker symbol."""
    return ticker.upper().replace(".NS", "").strip()


def _now_iso() -> str:
    """Return current UTC timestamp for quote fetch metadata."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sanitize_error_message(message: str) -> str:
    """Redact API keys from provider error messages before they reach the UI."""
    return SECRET_QUERY_RE.sub(r"\1***", message)


def _normalize_tickers(tickers: list[str] | tuple[str, ...]) -> list[str]:
    """Normalize and deduplicate incoming tickers."""
    unique_tickers = []
    for ticker in tickers:
        clean = str(ticker).strip().upper()
        if clean and clean not in unique_tickers:
            unique_tickers.append(clean)
    return unique_tickers


def _row(
    ticker: str,
    source: str,
    last_price: float | None,
    previous_close: float | None = None,
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float | None = None,
    change: float | None = None,
    change_percent: float | None = None,
    timestamp: str | None = None,
    status: str = "success",
    error: str = "",
) -> dict[str, Any]:
    """Build one normalized quote row."""
    last_price = _safe_float(last_price)
    previous_close = _safe_float(previous_close)
    open_price = _safe_float(open_price)
    high = _safe_float(high)
    low = _safe_float(low)
    volume = _safe_float(volume)
    change = _safe_float(change)
    change_percent = _safe_float(change_percent)

    if change is None and last_price is not None and previous_close not in (None, 0):
        change = last_price - previous_close
    if change_percent is None and change is not None and previous_close not in (None, 0):
        change_percent = change / previous_close
    elif change_percent is not None and abs(change_percent) > 1:
        change_percent = change_percent / 100

    return {
        "Ticker": ticker,
        "Last Price": last_price,
        "Previous Close": previous_close,
        "Open": open_price,
        "High": high,
        "Low": low,
        "Volume": volume,
        "Change": change,
        "Change %": change_percent,
        "Source": source or "Unavailable",
        "Timestamp": timestamp or _now_iso(),
        "Status": status,
        "Error": error,
    }


def _failed_row(ticker: str, error: str) -> dict[str, Any]:
    """Build a final unavailable row for a symbol with no usable data."""
    return _row(
        ticker=ticker,
        source="Unavailable",
        last_price=None,
        status="failed",
        error=error or "No provider returned usable data.",
    )


def _http_get_json(
    url: str,
    params: dict[str, Any],
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
) -> Any:
    """GET JSON with short timeout and retry support."""
    last_error: Exception | None = None
    for _ in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
    if last_error:
        raise last_error
    raise RuntimeError("Provider request failed.")


def _run_provider(
    provider_name: str,
    fetcher: Callable[[list[str]], list[dict[str, Any]]],
    tickers: list[str],
) -> tuple[list[dict[str, Any]], str | None]:
    """Run one provider with retries without breaking the whole quote request."""
    last_error: Exception | None = None
    provider_attempts = MAX_RETRIES if provider_name == "Yahoo Finance" else 1
    for _ in range(provider_attempts):
        try:
            return fetcher(tickers), None
        except Exception as error:
            last_error = error
    if last_error is None:
        return [], f"{provider_name}: Unknown provider failure"
    message = _sanitize_error_message(str(last_error))
    return [], f"{provider_name}: {last_error.__class__.__name__}: {message}"


def _extract_yfinance_frame(history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract one ticker frame from yfinance batch output."""
    if history.empty:
        return pd.DataFrame()
    if isinstance(history.columns, pd.MultiIndex):
        if ticker in history.columns.get_level_values(0):
            return history[ticker].copy()
        return pd.DataFrame()
    return history.copy()


def _fetch_yahoo_finance(tickers: list[str]) -> list[dict[str, Any]]:
    """Fetch delayed quotes from Yahoo Finance through yfinance."""
    if not tickers:
        return []
    history = yf.download(
        tickers=tickers if len(tickers) > 1 else tickers[0],
        period="5d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    rows = []
    for ticker in tickers:
        ticker_frame = _extract_yfinance_frame(history, ticker)
        if ticker_frame.empty:
            continue
        needed_columns = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in ticker_frame]
        if not needed_columns:
            continue
        ticker_frame = ticker_frame.dropna(how="all", subset=needed_columns)
        if ticker_frame.empty or "Close" not in ticker_frame:
            continue
        latest = ticker_frame.iloc[-1]
        previous = ticker_frame.iloc[-2] if len(ticker_frame) > 1 else None
        last_price = _safe_float(latest.get("Close"))
        if last_price is None:
            continue
        previous_close = _safe_float(previous.get("Close")) if previous is not None else None
        timestamp = None
        if ticker_frame.index.size:
            timestamp = str(ticker_frame.index[-1])
        rows.append(
            _row(
                ticker=ticker,
                source="Yahoo Finance",
                last_price=last_price,
                previous_close=previous_close,
                open_price=latest.get("Open"),
                high=latest.get("High"),
                low=latest.get("Low"),
                volume=latest.get("Volume"),
                timestamp=timestamp,
                status="success",
            )
        )
    return rows


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
                    source="Twelve Data",
                    last_price=price,
                    previous_close=payload.get("previous_close"),
                    change=payload.get("change"),
                    change_percent=payload.get("percent_change"),
                    open_price=payload.get("open"),
                    high=payload.get("high"),
                    low=payload.get("low"),
                    volume=payload.get("volume"),
                    timestamp=payload.get("datetime") or _now_iso(),
                    status="success",
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
            timestamp = None
            ts_value = payload.get("timestamp")
            if isinstance(ts_value, (int, float)):
                timestamp = datetime.fromtimestamp(ts_value, timezone.utc).replace(microsecond=0).isoformat()
            rows.append(
                _row(
                    ticker=ticker,
                    source="Financial Modeling Prep",
                    last_price=price,
                    previous_close=payload.get("previousClose"),
                    change=payload.get("change"),
                    change_percent=payload.get("changesPercentage"),
                    open_price=payload.get("open"),
                    high=payload.get("dayHigh"),
                    low=payload.get("dayLow"),
                    volume=payload.get("volume"),
                    timestamp=timestamp,
                    status="success",
                )
            )
    return rows


def _fetch_finnhub(tickers: list[str], api_key: str) -> list[dict[str, Any]]:
    """Fetch quotes from Finnhub one symbol at a time."""
    rows = []
    for ticker in tickers[:10]:
        try:
            payload = _http_get_json(
                "https://finnhub.io/api/v1/quote",
                {"symbol": f"NSE:{_base_symbol(ticker)}", "token": api_key},
            )
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        price = _safe_float(payload.get("c"))
        if price is not None and price > 0:
            timestamp = None
            ts_value = payload.get("t")
            if isinstance(ts_value, (int, float)) and ts_value > 0:
                timestamp = datetime.fromtimestamp(ts_value, timezone.utc).replace(microsecond=0).isoformat()
            rows.append(
                _row(
                    ticker=ticker,
                    source="Finnhub",
                    last_price=price,
                    previous_close=payload.get("pc"),
                    open_price=payload.get("o"),
                    high=payload.get("h"),
                    low=payload.get("l"),
                    timestamp=timestamp,
                    status="success",
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
                    source="Marketstack",
                    last_price=price,
                    open_price=payload.get("open"),
                    high=payload.get("high"),
                    low=payload.get("low"),
                    volume=payload.get("volume"),
                    timestamp=payload.get("date"),
                    status="success",
                )
            )
    return rows


def _fetch_alpha_vantage(tickers: list[str], api_key: str) -> list[dict[str, Any]]:
    """Fetch a small number of Global Quote rows from Alpha Vantage."""
    rows = []
    for ticker in tickers[:3]:
        try:
            payload = _http_get_json(
                "https://www.alphavantage.co/query",
                {"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": api_key},
            )
        except Exception:
            continue
        quote = payload.get("Global Quote", {}) if isinstance(payload, dict) else {}
        price = _safe_float(quote.get("05. price"))
        if price is not None:
            rows.append(
                _row(
                    ticker=ticker,
                    source="Alpha Vantage",
                    last_price=price,
                    previous_close=quote.get("08. previous close"),
                    change=quote.get("09. change"),
                    change_percent=quote.get("10. change percent"),
                    open_price=quote.get("02. open"),
                    high=quote.get("03. high"),
                    low=quote.get("04. low"),
                    volume=quote.get("06. volume"),
                    timestamp=quote.get("07. latest trading day"),
                    status="success",
                )
            )
    return rows


def _fetch_local_cache(tickers: list[str]) -> list[dict[str, Any]]:
    """Use the cleaned local price dataset as a cached fallback."""
    if not CLEANED_PRICE_FILE.exists():
        return []
    try:
        prices = pd.read_csv(CLEANED_PRICE_FILE, parse_dates=["Date"])
    except Exception:
        return []
    rows = []
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        series = pd.to_numeric(prices[ticker], errors="coerce").dropna()
        if series.empty:
            continue
        last_index = series.index[-1]
        previous_value = series.iloc[-2] if len(series) > 1 else None
        timestamp = None
        if "Date" in prices.columns:
            timestamp = str(prices.loc[last_index, "Date"])
        rows.append(
            _row(
                ticker=ticker,
                source="Local cache",
                last_price=series.iloc[-1],
                previous_close=previous_value,
                timestamp=timestamp,
                status="cached",
            )
        )
    return rows


def _fetch_historical_fallback(
    tickers: list[str],
    historical_last_prices: dict[str, Any] | None,
    historical_previous_closes: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Use app-provided historical closes as the final fallback."""
    historical_last_prices = historical_last_prices or {}
    historical_previous_closes = historical_previous_closes or {}
    rows = []
    for ticker in tickers:
        last_price = _safe_float(historical_last_prices.get(ticker))
        if last_price is None:
            continue
        rows.append(
            _row(
                ticker=ticker,
                source="Historical fallback",
                last_price=last_price,
                previous_close=historical_previous_closes.get(ticker),
                timestamp=_now_iso(),
                status="fallback",
            )
        )
    return rows


def _normalize_frame(rows: list[dict[str, Any]], tickers: list[str]) -> pd.DataFrame:
    """Return a frame with stable columns, ordering, and numeric dtypes."""
    if not rows:
        rows = [_failed_row(ticker, "No provider returned usable data.") for ticker in tickers]
    frame = pd.DataFrame(rows)
    for column in STANDARD_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[STANDARD_COLUMNS]
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Source"] = frame["Source"].fillna("Unavailable").replace("", "Unavailable")
    frame["Status"] = frame["Status"].fillna("failed").replace("", "failed")
    frame["Timestamp"] = frame["Timestamp"].fillna(_now_iso()).replace("", _now_iso())
    frame["Error"] = frame["Error"].fillna("")
    order = {ticker: index for index, ticker in enumerate(tickers)}
    frame["_order"] = frame["Ticker"].map(order)
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _result_status(frame: pd.DataFrame) -> str:
    """Summarize row status values into the public result status."""
    if frame.empty:
        return "failed"
    statuses = set(frame["Status"].astype(str))
    if "failed" in statuses:
        return "partial" if len(statuses) > 1 else "failed"
    if statuses == {"success"}:
        return "success"
    if statuses == {"cached"}:
        return "cached"
    if "cached" in statuses or "fallback" in statuses:
        return "fallback"
    return "failed"


def _result_source(frame: pd.DataFrame) -> str:
    """Return a non-empty source summary."""
    if frame.empty:
        return "Unavailable"
    sources = [
        source
        for source in frame["Source"].dropna().astype(str).unique().tolist()
        if source and source != "Unavailable"
    ]
    return ", ".join(sources) if sources else "Unavailable"


def _build_result(
    rows: list[dict[str, Any]],
    tickers: list[str],
    errors: list[str],
) -> LiveQuoteResult:
    """Create a structured live quote result."""
    frame = _normalize_frame(rows, tickers)
    failed_symbols = frame.loc[
        frame["Last Price"].isna() | frame["Status"].eq("failed"),
        "Ticker",
    ].astype(str).tolist()
    return LiveQuoteResult(
        dataframe=frame,
        source=_result_source(frame),
        status=_result_status(frame),
        timestamp=_now_iso(),
        errors=errors,
        failed_symbols=failed_symbols,
    )


def get_configured_provider_names() -> list[str]:
    """Return providers in the order the engine will attempt them."""
    names = ["Yahoo Finance"]
    names.extend(get_configured_api_provider_names())
    names.extend(["Local cache", "Historical fallback"])
    return names


def _apply_provider_preference(
    provider_plan: list[tuple[str, Callable[[list[str]], list[dict[str, Any]]]]],
    provider_preference: str,
) -> list[tuple[str, Callable[[list[str]], list[dict[str, Any]]]]]:
    """Move a selected provider to the front while preserving fallback order."""
    if not provider_preference or provider_preference == "Auto":
        return provider_plan
    preferred = [item for item in provider_plan if item[0] == provider_preference]
    others = [item for item in provider_plan if item[0] != provider_preference]
    return preferred + others if preferred else provider_plan


def fetch_live_quotes(
    tickers: list[str] | tuple[str, ...],
    historical_last_prices: dict[str, Any] | None = None,
    historical_previous_closes: dict[str, Any] | None = None,
    provider_preference: str = "Auto",
) -> LiveQuoteResult:
    """Fetch quotes with per-symbol fallback and structured metadata."""
    unique_tickers = _normalize_tickers(tickers)
    if not unique_tickers:
        return LiveQuoteResult(
            dataframe=_normalize_frame([], []),
            source="Unavailable",
            status="failed",
            timestamp=_now_iso(),
            errors=["No symbols selected."],
            failed_symbols=[],
        )

    keys = get_api_keys()
    provider_plan: list[tuple[str, Callable[[list[str]], list[dict[str, Any]]]]] = [
        ("Yahoo Finance", _fetch_yahoo_finance),
    ]
    if keys.twelve_data:
        provider_plan.append(("Twelve Data", lambda symbols: _fetch_twelve_data(symbols, keys.twelve_data or "")))
    if keys.fmp:
        provider_plan.append(("Financial Modeling Prep", lambda symbols: _fetch_fmp(symbols, keys.fmp or "")))
    if keys.marketstack:
        provider_plan.append(("Marketstack", lambda symbols: _fetch_marketstack(symbols, keys.marketstack or "")))
    if keys.finnhub:
        provider_plan.append(("Finnhub", lambda symbols: _fetch_finnhub(symbols, keys.finnhub or "")))
    if keys.alpha_vantage:
        provider_plan.append(("Alpha Vantage", lambda symbols: _fetch_alpha_vantage(symbols, keys.alpha_vantage or "")))
    provider_plan.extend(
        [
            ("Local cache", _fetch_local_cache),
            (
                "Historical fallback",
                lambda symbols: _fetch_historical_fallback(
                    symbols,
                    historical_last_prices,
                    historical_previous_closes,
                ),
            ),
        ]
    )
    provider_plan = _apply_provider_preference(provider_plan, provider_preference)

    rows: list[dict[str, Any]] = []
    covered: set[str] = set()
    errors: list[str] = []

    for provider_name, fetcher in provider_plan:
        missing = [ticker for ticker in unique_tickers if ticker not in covered]
        if not missing:
            break
        provider_rows, provider_error = _run_provider(provider_name, fetcher, missing)
        if provider_error:
            errors.append(provider_error)
            continue
        for row in provider_rows:
            ticker = str(row.get("Ticker", "")).upper()
            price = _safe_float(row.get("Last Price"))
            if ticker in missing and ticker not in covered and price is not None:
                rows.append(row)
                covered.add(ticker)

    missing = [ticker for ticker in unique_tickers if ticker not in covered]
    if missing:
        error_text = "; ".join(errors[-3:]) if errors else "No provider returned usable data."
        rows.extend(_failed_row(ticker, error_text) for ticker in missing)

    return _build_result(rows, unique_tickers, errors)
