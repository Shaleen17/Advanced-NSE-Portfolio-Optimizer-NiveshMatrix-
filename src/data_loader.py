"""Data collection, validation, caching, and cleaning functions."""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", message=".*urllib3.*", category=Warning)

import yfinance as yf

from config import (
    BENCHMARK_RETURNS_FILE,
    BENCHMARK_TICKER,
    CLEANED_PRICE_FILE,
    NSE_TICKERS,
    ensure_project_folders,
)


class DataLoadError(RuntimeError):
    """Raised when price data cannot be loaded in a reliable way."""


def normalize_tickers(tickers: list[str] | tuple[str, ...]) -> list[str]:
    """Return uppercase unique tickers while preserving order."""
    cleaned: list[str] = []
    for ticker in tickers:
        ticker_text = str(ticker).strip().upper()
        if ticker_text and ticker_text not in cleaned:
            cleaned.append(ticker_text)
    return cleaned


def validate_tickers(tickers: list[str]) -> None:
    """Validate selected NSE tickers against the configured universe."""
    if len(tickers) < 2:
        raise DataLoadError("Select at least two stocks for portfolio optimization.")

    supported = set(NSE_TICKERS)
    invalid = [ticker for ticker in tickers if ticker not in supported]
    if invalid:
        raise DataLoadError(f"Unsupported ticker(s): {', '.join(invalid)}")


def validate_date_range(start_date: date, end_date: date) -> None:
    """Validate the selected date range."""
    if start_date >= end_date:
        raise DataLoadError("Start date must be before end date.")


def extract_adjusted_close(raw_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract Adjusted Close prices from yfinance data."""
    if raw_data.empty:
        raise DataLoadError("No data was returned by Yahoo Finance.")

    if isinstance(raw_data.columns, pd.MultiIndex):
        level_zero = raw_data.columns.get_level_values(0)
        level_one = raw_data.columns.get_level_values(1)
        if "Adj Close" in level_zero:
            prices = raw_data["Adj Close"]
        elif "Adj Close" in level_one:
            prices = raw_data.xs("Adj Close", axis=1, level=1)
        elif "Close" in level_zero:
            prices = raw_data["Close"]
        elif "Close" in level_one:
            prices = raw_data.xs("Close", axis=1, level=1)
        else:
            raise DataLoadError("Downloaded data does not contain close prices.")
    else:
        close_column = "Adj Close" if "Adj Close" in raw_data.columns else "Close"
        if close_column not in raw_data.columns:
            raise DataLoadError("Downloaded data does not contain close prices.")
        prices = raw_data[[close_column]].rename(columns={close_column: tickers[0]})

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.reindex(columns=tickers)
    return prices


def clean_price_data(prices: pd.DataFrame, max_missing_ratio: float = 0.40) -> pd.DataFrame:
    """Clean price data by removing sparse columns and filling small gaps."""
    if prices.empty:
        raise DataLoadError("Price dataset is empty.")

    missing_ratio = prices.isna().mean()
    usable_columns = missing_ratio[missing_ratio <= max_missing_ratio].index.tolist()
    cleaned = prices.loc[:, usable_columns].sort_index()
    cleaned = cleaned.ffill().bfill().dropna(how="any")

    if cleaned.shape[1] < 2:
        raise DataLoadError("Not enough stocks have usable price history.")
    if len(cleaned) < 60:
        raise DataLoadError("At least 60 trading days are required for analysis.")

    return cleaned


def download_price_data(tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    """Download Adjusted Close data for selected NSE tickers."""
    validate_tickers(tickers)
    validate_date_range(start_date, end_date)

    try:
        raw_data = yf.download(
            tickers=tickers,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=False,
            group_by="column",
            progress=False,
            threads=True,
        )
    except Exception as error:
        raise DataLoadError(f"Could not download stock data: {error}") from error

    return clean_price_data(extract_adjusted_close(raw_data, tickers))


def load_cached_prices(
    tickers: list[str],
    start_date: date,
    end_date: date,
    file_path: Path = CLEANED_PRICE_FILE,
) -> pd.DataFrame:
    """Load cleaned prices from the local CSV cache."""
    if not file_path.exists():
        raise DataLoadError(f"Cached price file not found: {file_path}")

    try:
        prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except Exception as error:
        raise DataLoadError(f"Could not read cached prices: {error}") from error

    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise DataLoadError("Cached data is missing: " + ", ".join(missing))

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    filtered = prices.loc[(prices.index >= start) & (prices.index <= end), tickers]
    return clean_price_data(filtered)


def save_cleaned_prices(prices: pd.DataFrame, file_path: Path = CLEANED_PRICE_FILE) -> None:
    """Save cleaned stock prices to disk."""
    ensure_project_folders()
    try:
        prices.to_csv(file_path)
    except Exception as error:
        raise DataLoadError(f"Could not save cleaned prices: {error}") from error


def get_price_data(
    tickers: list[str] | tuple[str, ...],
    start_date: date,
    end_date: date,
    use_cache_first: bool = True,
) -> pd.DataFrame:
    """Load cached stock prices first, then download fresh data if required."""
    selected_tickers = normalize_tickers(tickers)
    validate_tickers(selected_tickers)
    validate_date_range(start_date, end_date)

    if use_cache_first:
        try:
            return load_cached_prices(selected_tickers, start_date, end_date)
        except DataLoadError:
            pass

    prices = download_price_data(selected_tickers, start_date, end_date)
    save_cleaned_prices(prices)
    return prices


def load_cached_benchmark_returns(file_path: Path = BENCHMARK_RETURNS_FILE) -> pd.Series:
    """Load locally cached NIFTY 50 benchmark returns when available."""
    if not file_path.exists():
        raise DataLoadError("Cached benchmark returns file was not found.")

    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    if isinstance(data, pd.DataFrame):
        series = data.iloc[:, 0]
    else:
        series = data
    return pd.to_numeric(series, errors="coerce").dropna()


def download_benchmark_returns(start_date: date, end_date: date) -> pd.Series:
    """Download NIFTY 50 benchmark daily returns from Yahoo Finance."""
    try:
        raw_data = yf.download(
            tickers=BENCHMARK_TICKER,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=False,
            progress=False,
        )
    except Exception as error:
        raise DataLoadError(f"Could not download benchmark data: {error}") from error

    prices = extract_adjusted_close(raw_data, [BENCHMARK_TICKER])
    benchmark_returns = prices.iloc[:, 0].pct_change().dropna()
    ensure_project_folders()
    benchmark_returns.to_csv(BENCHMARK_RETURNS_FILE)
    return benchmark_returns


def get_benchmark_returns(
    start_date: date,
    end_date: date,
    use_cache_first: bool = True,
) -> pd.Series | None:
    """Return NIFTY 50 benchmark returns, or None when unavailable."""
    if use_cache_first:
        try:
            series = load_cached_benchmark_returns()
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            return series.loc[(series.index >= start) & (series.index <= end)]
        except Exception:
            pass

    try:
        return download_benchmark_returns(start_date, end_date)
    except Exception:
        return None
