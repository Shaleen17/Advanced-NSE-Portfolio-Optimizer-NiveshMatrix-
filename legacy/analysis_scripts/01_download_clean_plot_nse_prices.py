"""
First working script for an NSE portfolio optimization project.

This script downloads 5 years of Indian NSE stock price data, keeps the
Adjusted Close prices, handles missing values, saves the cleaned data, and
creates beginner-friendly charts with a black fintech-style theme.

Before running this file, install the required packages:

    python -m pip install -r requirements.txt

Run this script:

    python scripts/01_download_clean_plot_nse_prices.py
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# Matplotlib sometimes tries to write cache files inside the user home folder.
# This project-level cache folder avoids permission problems on college lab PCs.
MPL_CONFIG_DIR = Path(".matplotlib")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# 1. Project settings
# -----------------------------

# These are the NSE stocks used in the project.
# The ".NS" suffix tells Yahoo Finance that these are Indian NSE-listed stocks.
TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "BHARTIARTL.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "TITAN.NS",
    "SUNPHARMA.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TECHM.NS",
    "ULTRACEMCO.NS",
    "NESTLEIND.NS",
    "POWERGRID.NS",
    "NTPC.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "HINDALCO.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "GRASIM.NS",
    "M&M.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS",
    "BRITANNIA.NS",
    "DIVISLAB.NS",
    "APOLLOHOSP.NS",
    "BPCL.NS",
    "IOC.NS",
    "TATAMOTORS.NS",
    "TATACONSUM.NS",
    "UPL.NS",
    "SHREECEM.NS",
    "INDUSINDBK.NS",
    "SBILIFE.NS",
]

# Plot only selected major companies so the chart remains readable.
SELECTED_PLOT_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
    "HINDUNILVR.NS",
]

# Use the last 5 years of daily data.
PERIOD = "5y"

# Daily data is enough for portfolio optimization.
INTERVAL = "1d"

# If more than 10% of values are still missing after filling, remove that stock.
MISSING_LIMIT_PERCENT = 10

# Folder paths for saving cleaned data and charts.
DATA_OUTPUT_DIR = Path("data/processed")
FIGURE_OUTPUT_DIR = Path("reports/figures")

# Output file locations.
CLEANED_DATA_FILE = DATA_OUTPUT_DIR / "nse_adjusted_close_cleaned.csv"
PRICE_TREND_FIGURE = FIGURE_OUTPUT_DIR / "selected_nse_price_trends.png"
TOTAL_RETURN_FIGURE = FIGURE_OUTPUT_DIR / "selected_nse_total_returns.png"

# Dashboard/chart colors requested for the project.
PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"
PURE_GREEN = "#00FF00"
PURE_RED = "#FF0000"


# -----------------------------
# 2. Helper functions
# -----------------------------

def create_output_folders() -> None:
    """Create output folders if they do not already exist."""
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_stock_data(tickers: list[str]) -> pd.DataFrame:
    """Download historical NSE stock data from Yahoo Finance."""
    print("Downloading 5 years of NSE stock data from Yahoo Finance...")

    # auto_adjust=False is important because it keeps the "Adj Close" column.
    # Without this setting, some yfinance versions may return only adjusted prices
    # and may not show a separate "Adj Close" column.
    data = yf.download(
        tickers=tickers,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    return data


def extract_adjusted_close(data: pd.DataFrame) -> pd.DataFrame:
    """Extract Adjusted Close prices from downloaded yfinance data."""
    if data.empty:
        raise ValueError("No data was downloaded. Please check your internet connection or ticker symbols.")

    if "Adj Close" not in data.columns.get_level_values(0):
        raise KeyError(
            "Adjusted Close prices were not found. Try upgrading yfinance or use auto_adjust=False."
        )

    # For many tickers, yfinance returns columns grouped by price type.
    # data["Adj Close"] keeps only the adjusted close prices for all stocks.
    adjusted_close = data["Adj Close"].copy()

    # Sort dates from oldest to newest.
    adjusted_close = adjusted_close.sort_index()

    return adjusted_close


def clean_missing_values(prices: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values and remove stocks with too much missing data."""
    print("\nMissing values before cleaning:")
    print(prices.isna().sum().sort_values(ascending=False))

    # Forward fill uses the previous available price.
    # This is useful for small gaps caused by holidays or temporary data issues.
    filled_prices = prices.ffill()

    # Backward fill handles missing values at the beginning of the dataset.
    filled_prices = filled_prices.bfill()

    print("\nMissing values after forward fill and backward fill:")
    missing_after_fill = filled_prices.isna().sum().sort_values(ascending=False)
    print(missing_after_fill)

    # Convert missing counts into percentages.
    missing_percent = (filled_prices.isna().sum() / len(filled_prices)) * 100

    # Keep only stocks where remaining missing values are within the allowed limit.
    stocks_to_keep = missing_percent[missing_percent <= MISSING_LIMIT_PERCENT].index
    stocks_removed = missing_percent[missing_percent > MISSING_LIMIT_PERCENT].index

    if len(stocks_removed) > 0:
        print("\nStocks removed because too much data is still missing:")
        print(list(stocks_removed))
    else:
        print("\nNo stocks were removed after missing value cleaning.")

    cleaned_prices = filled_prices[stocks_to_keep].copy()

    # Drop rows that still contain any missing values after removing bad stocks.
    cleaned_prices = cleaned_prices.dropna(axis=0, how="any")

    return cleaned_prices


def apply_black_chart_theme(ax: plt.Axes) -> None:
    """Apply pure black background and pure white text to a chart."""
    ax.set_facecolor(PURE_BLACK)
    ax.figure.set_facecolor(PURE_BLACK)

    ax.title.set_color(PURE_WHITE)
    ax.xaxis.label.set_color(PURE_WHITE)
    ax.yaxis.label.set_color(PURE_WHITE)

    ax.tick_params(axis="x", colors=PURE_WHITE)
    ax.tick_params(axis="y", colors=PURE_WHITE)

    for spine in ax.spines.values():
        spine.set_color(PURE_WHITE)

    ax.grid(color=PURE_WHITE, alpha=0.15)


def plot_selected_price_trends(prices: pd.DataFrame) -> None:
    """Plot adjusted close price trends for selected major NSE companies."""
    available_tickers = [ticker for ticker in SELECTED_PLOT_TICKERS if ticker in prices.columns]

    if not available_tickers:
        print("No selected tickers are available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    for ticker in available_tickers:
        ax.plot(prices.index, prices[ticker], linewidth=1.8, label=ticker)

    ax.set_title("Adjusted Close Price Trends of Selected NSE Stocks", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price")

    apply_black_chart_theme(ax)

    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE, fontsize=9)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)

    plt.tight_layout()
    plt.savefig(PRICE_TREND_FIGURE, dpi=300, facecolor=PURE_BLACK)
    plt.show()

    print(f"\nPrice trend chart saved to: {PRICE_TREND_FIGURE}")


def plot_selected_total_returns(prices: pd.DataFrame) -> None:
    """Plot total returns for selected stocks using green for profit and red for loss."""
    available_tickers = [ticker for ticker in SELECTED_PLOT_TICKERS if ticker in prices.columns]

    if not available_tickers:
        print("No selected tickers are available for total return plotting.")
        return

    selected_prices = prices[available_tickers]

    # Total return measures how much each stock gained or lost over the full period.
    total_returns = (selected_prices.iloc[-1] / selected_prices.iloc[0]) - 1
    total_returns_percent = total_returns * 100

    # Positive returns are pure green. Negative returns are pure red.
    bar_colors = np.where(total_returns_percent >= 0, PURE_GREEN, PURE_RED)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(total_returns_percent.index, total_returns_percent.values, color=bar_colors)

    ax.set_title("Total Return of Selected NSE Stocks Over 5 Years", fontsize=16)
    ax.set_xlabel("Stock")
    ax.set_ylabel("Total Return (%)")

    apply_black_chart_theme(ax)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(TOTAL_RETURN_FIGURE, dpi=300, facecolor=PURE_BLACK)
    plt.show()

    print(f"Total return chart saved to: {TOTAL_RETURN_FIGURE}")


# -----------------------------
# 3. Main project workflow
# -----------------------------

def main() -> None:
    """Run the first step of the portfolio optimization project."""
    create_output_folders()

    raw_data = download_stock_data(TICKERS)

    adjusted_close = extract_adjusted_close(raw_data)

    print("\nFirst 5 rows of Adjusted Close price data:")
    print(adjusted_close.head())

    print("\nDataset shape before cleaning:")
    print(adjusted_close.shape)

    cleaned_prices = clean_missing_values(adjusted_close)

    print("\nFirst 5 rows of cleaned Adjusted Close price data:")
    print(cleaned_prices.head())

    print("\nDataset shape after cleaning:")
    print(cleaned_prices.shape)

    cleaned_prices.to_csv(CLEANED_DATA_FILE)
    print(f"\nCleaned data saved to: {CLEANED_DATA_FILE}")

    plot_selected_price_trends(cleaned_prices)
    plot_selected_total_returns(cleaned_prices)

    print("\nFirst data preparation step completed successfully.")


if __name__ == "__main__":
    main()
