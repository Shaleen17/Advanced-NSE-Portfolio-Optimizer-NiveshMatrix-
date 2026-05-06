"""
Intermediate portfolio analysis for the NSE portfolio optimization project.

This script continues from the cleaned Adjusted Close price dataset created
in the first step. It calculates daily returns, expected annual returns,
annual volatility, covariance, correlation, and creates dark fintech-style
charts for college project reporting.

Before running this file, install the required packages:

    python -m pip install -r requirements.txt

Run this script:

    python scripts/02_intermediate_portfolio_analysis.py
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# Matplotlib may need a writable cache folder on some Windows college lab PCs.
MPL_CONFIG_DIR = Path(".matplotlib")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# -----------------------------
# 1. Project settings
# -----------------------------

# Number of approximate trading days in one year.
TRADING_DAYS = 252

# Input file from the first project step.
CLEANED_PRICE_FILE = Path("data/processed/nse_adjusted_close_cleaned.csv")

# Output folders.
OUTPUT_DATA_DIR = Path("data/outputs")
FIGURE_OUTPUT_DIR = Path("reports/figures")

# Output files for tables.
DAILY_RETURNS_FILE = OUTPUT_DATA_DIR / "daily_returns.csv"
SUMMARY_TABLE_FILE = OUTPUT_DATA_DIR / "stock_risk_return_summary.csv"
COVARIANCE_MATRIX_FILE = OUTPUT_DATA_DIR / "covariance_matrix.csv"
CORRELATION_MATRIX_FILE = OUTPUT_DATA_DIR / "correlation_matrix.csv"

# Output files for charts.
CORRELATION_HEATMAP_FILE = FIGURE_OUTPUT_DIR / "correlation_heatmap.png"
EXPECTED_RETURN_CHART_FILE = FIGURE_OUTPUT_DIR / "annual_expected_returns.png"
VOLATILITY_CHART_FILE = FIGURE_OUTPUT_DIR / "annual_volatility_lowest_20.png"

# Pure colors requested for the final project theme.
PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"
PURE_GREEN = "#00FF00"
PURE_RED = "#FF0000"


# -----------------------------
# 2. Helper functions
# -----------------------------

def create_output_folders() -> None:
    """Create folders for output tables and charts."""
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cleaned_prices() -> pd.DataFrame:
    """Load cleaned Adjusted Close prices from CSV."""
    if not CLEANED_PRICE_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned price file not found: {CLEANED_PRICE_FILE}\n"
            "Please run 01_download_clean_plot_nse_prices.py first."
        )

    # index_col=0 means the first column, Date, becomes the DataFrame index.
    prices = pd.read_csv(CLEANED_PRICE_FILE, index_col=0, parse_dates=True)

    # Make sure rows are sorted from oldest date to newest date.
    prices = prices.sort_index()

    return prices


def apply_black_chart_theme(ax: plt.Axes) -> None:
    """Apply pure black background and pure white labels to a chart."""
    ax.set_facecolor(PURE_BLACK)
    ax.figure.set_facecolor(PURE_BLACK)

    ax.title.set_color(PURE_WHITE)
    ax.xaxis.label.set_color(PURE_WHITE)
    ax.yaxis.label.set_color(PURE_WHITE)

    ax.tick_params(axis="x", colors=PURE_WHITE)
    ax.tick_params(axis="y", colors=PURE_WHITE)

    for spine in ax.spines.values():
        spine.set_color(PURE_WHITE)

    ax.grid(color=PURE_WHITE, alpha=0.12)


def classify_risk_return(row: pd.Series) -> str:
    """
    Assign a simple beginner-friendly risk-return category.

    This is not investment advice. It is only a project-friendly label based
    on whether a stock is above or below the median return and median risk.
    """
    high_return = row["Annual Expected Return"] >= row["Median Annual Expected Return"]
    high_risk = row["Annual Volatility"] >= row["Median Annual Volatility"]

    if high_return and not high_risk:
        return "High Return / Lower Risk"
    if high_return and high_risk:
        return "High Return / Higher Risk"
    if not high_return and not high_risk:
        return "Lower Return / Lower Risk"
    return "Lower Return / Higher Risk"


def create_summary_table(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Create a stock-level risk and return summary table."""
    # Formula: Annual Return = mean daily return x 252
    average_daily_return = daily_returns.mean()
    annual_expected_return = average_daily_return * TRADING_DAYS

    # Formula: Annual Volatility = standard deviation of daily returns x sqrt(252)
    daily_volatility = daily_returns.std()
    annual_volatility = daily_volatility * np.sqrt(TRADING_DAYS)

    summary = pd.DataFrame(
        {
            "Stock Ticker": annual_expected_return.index,
            "Average Daily Return": average_daily_return.values,
            "Annual Expected Return": annual_expected_return.values,
            "Daily Volatility": daily_volatility.values,
            "Annual Volatility": annual_volatility.values,
        }
    )

    # Store medians temporarily so each stock can be classified.
    summary["Median Annual Expected Return"] = summary["Annual Expected Return"].median()
    summary["Median Annual Volatility"] = summary["Annual Volatility"].median()

    summary["Risk-Return Category"] = summary.apply(classify_risk_return, axis=1)

    # Remove helper columns after classification.
    summary = summary.drop(
        columns=["Median Annual Expected Return", "Median Annual Volatility"]
    )

    # Sort by highest annual expected return for the main summary.
    summary = summary.sort_values("Annual Expected Return", ascending=False)

    return summary


def print_percentage_table(table: pd.DataFrame, rows: int = 10) -> None:
    """Print a readable table with returns and volatility shown as percentages."""
    display_table = table.head(rows).copy()

    percent_columns = [
        "Average Daily Return",
        "Annual Expected Return",
        "Daily Volatility",
        "Annual Volatility",
    ]

    for column in percent_columns:
        display_table[column] = display_table[column].map(lambda value: f"{value * 100:.2f}%")

    print(display_table.to_string(index=False))


def plot_annual_expected_returns(summary: pd.DataFrame) -> None:
    """Plot annual expected returns with green for positive and red for negative."""
    plot_data = summary.sort_values("Annual Expected Return", ascending=True)
    colors = np.where(plot_data["Annual Expected Return"] >= 0, PURE_GREEN, PURE_RED)

    fig, ax = plt.subplots(figsize=(12, 14))
    ax.barh(plot_data["Stock Ticker"], plot_data["Annual Expected Return"] * 100, color=colors)

    ax.set_title("Annual Expected Return by Stock", fontsize=16)
    ax.set_xlabel("Annual Expected Return (%)")
    ax.set_ylabel("Stock Ticker")

    apply_black_chart_theme(ax)

    plt.tight_layout()
    plt.savefig(EXPECTED_RETURN_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    plt.show()

    print(f"\nAnnual expected return chart saved to: {EXPECTED_RETURN_CHART_FILE}")


def plot_lowest_volatility_stocks(summary: pd.DataFrame, count: int = 20) -> None:
    """Plot the lowest volatility stocks to show comparatively lower-risk stocks."""
    plot_data = summary.sort_values("Annual Volatility", ascending=True).head(count)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(plot_data["Stock Ticker"], plot_data["Annual Volatility"] * 100, color=PURE_WHITE)

    ax.set_title(f"Lowest {count} Stocks by Annual Volatility", fontsize=16)
    ax.set_xlabel("Annual Volatility (%)")
    ax.set_ylabel("Stock Ticker")

    apply_black_chart_theme(ax)

    plt.tight_layout()
    plt.savefig(VOLATILITY_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    plt.show()

    print(f"Annual volatility chart saved to: {VOLATILITY_CHART_FILE}")


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> None:
    """Plot a readable correlation heatmap on a pure black background."""
    plt.figure(figsize=(18, 14), facecolor=PURE_BLACK)

    # A diverging palette makes negative, neutral, and positive relationships easier to see.
    ax = sns.heatmap(
        correlation_matrix,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.2,
        linecolor="#222222",
        cbar_kws={"label": "Correlation"},
    )

    ax.set_title("Correlation Heatmap of NSE Stock Daily Returns", fontsize=16, color=PURE_WHITE)
    ax.set_xlabel("Stock Ticker", color=PURE_WHITE)
    ax.set_ylabel("Stock Ticker", color=PURE_WHITE)

    ax.tick_params(axis="x", colors=PURE_WHITE, labelsize=8, rotation=90)
    ax.tick_params(axis="y", colors=PURE_WHITE, labelsize=8)

    # Make colorbar readable on black background.
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.label.set_color(PURE_WHITE)
    colorbar.ax.tick_params(colors=PURE_WHITE)

    ax.figure.set_facecolor(PURE_BLACK)
    ax.set_facecolor(PURE_BLACK)

    plt.tight_layout()
    plt.savefig(CORRELATION_HEATMAP_FILE, dpi=300, facecolor=PURE_BLACK)
    plt.show()

    print(f"Correlation heatmap saved to: {CORRELATION_HEATMAP_FILE}")


# -----------------------------
# 3. Main workflow
# -----------------------------

def main() -> None:
    """Run intermediate portfolio analysis."""
    create_output_folders()

    prices = load_cleaned_prices()

    print("Cleaned Adjusted Close price data loaded successfully.")
    print(f"Price dataset shape: {prices.shape}")

    # Daily return formula:
    # Daily Return = Current Price / Previous Price - 1
    daily_returns = prices.pct_change()

    # The first row becomes missing because there is no previous day for comparison.
    daily_returns = daily_returns.dropna()

    print("\nFirst 5 rows of daily returns:")
    print(daily_returns.head())

    print("\nDaily returns dataset shape:")
    print(daily_returns.shape)

    daily_returns.to_csv(DAILY_RETURNS_FILE)
    print(f"\nDaily returns saved to: {DAILY_RETURNS_FILE}")

    summary = create_summary_table(daily_returns)
    summary.to_csv(SUMMARY_TABLE_FILE, index=False)

    print("\nTop 10 stocks sorted by highest annual expected return:")
    print_percentage_table(summary, rows=10)

    lowest_volatility = summary.sort_values("Annual Volatility", ascending=True)

    print("\nTop 10 stocks sorted by lowest annual volatility:")
    print_percentage_table(lowest_volatility, rows=10)

    print(f"\nRisk-return summary table saved to: {SUMMARY_TABLE_FILE}")

    # Covariance measures how two stocks move together in raw return units.
    covariance_matrix = daily_returns.cov()
    covariance_matrix.to_csv(COVARIANCE_MATRIX_FILE)

    print("\nCovariance matrix created.")
    print(f"Covariance matrix shape: {covariance_matrix.shape}")
    print(f"Covariance matrix saved to: {COVARIANCE_MATRIX_FILE}")

    # Correlation is easier to interpret because values stay between -1 and +1.
    correlation_matrix = daily_returns.corr()
    correlation_matrix.to_csv(CORRELATION_MATRIX_FILE)

    print("\nCorrelation matrix created.")
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    print(f"Correlation matrix saved to: {CORRELATION_MATRIX_FILE}")

    plot_annual_expected_returns(summary)
    plot_lowest_volatility_stocks(summary)
    plot_correlation_heatmap(correlation_matrix)

    print("\nIntermediate portfolio analysis completed successfully.")


if __name__ == "__main__":
    main()
