"""
Random portfolio calculation module for the NSE portfolio optimization project.

This script uses the daily returns, annual expected returns, and covariance
matrix created in step 2. It builds one random portfolio, calculates its
expected annual return, annual risk, and Sharpe ratio, then saves the results
and charts for college project reporting.

Before running this file:

    python scripts/01_download_clean_plot_nse_prices.py
    python scripts/02_intermediate_portfolio_analysis.py

Run this file:

    python scripts/03_random_portfolio_calculation.py
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# Matplotlib may need a writable cache folder on some Windows systems.
MPL_CONFIG_DIR = Path(".matplotlib")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# 1. Project settings
# -----------------------------

# India-friendly default risk-free rate assumption.
# For a college project, 6% is a reasonable editable assumption.
RISK_FREE_RATE = 0.06

# The Indian stock market has about 252 trading days in a year.
TRADING_DAYS = 252

# A fixed random seed makes the same random portfolio repeatable.
RANDOM_SEED = 42

# Input files created in step 2.
DAILY_RETURNS_FILE = Path("data/outputs/daily_returns.csv")
SUMMARY_TABLE_FILE = Path("data/outputs/stock_risk_return_summary.csv")
COVARIANCE_MATRIX_FILE = Path("data/outputs/covariance_matrix.csv")

# Output folders.
OUTPUT_DATA_DIR = Path("data/outputs")
FIGURE_OUTPUT_DIR = Path("reports/figures")

# Output files for this step.
RANDOM_WEIGHTS_FILE = OUTPUT_DATA_DIR / "random_portfolio_weights.csv"
RANDOM_RESULT_FILE = OUTPUT_DATA_DIR / "random_portfolio_result.csv"
RANDOM_COMPARISON_FILE = OUTPUT_DATA_DIR / "random_portfolio_weight_comparison.csv"
ALLOCATION_CHART_FILE = FIGURE_OUTPUT_DIR / "random_portfolio_allocation.png"
RISK_RETURN_COMPARISON_CHART_FILE = FIGURE_OUTPUT_DIR / "random_portfolio_risk_return_comparison.png"

# Pure colors requested for the project.
PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"
PURE_GREEN = "#00FF00"
PURE_RED = "#FF0000"


# -----------------------------
# 2. Helper functions
# -----------------------------

def create_output_folders() -> None:
    """Create output folders if they do not already exist."""
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_previous_analysis_outputs() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load daily returns, annual expected returns, and daily covariance matrix."""
    required_files = [DAILY_RETURNS_FILE, SUMMARY_TABLE_FILE, COVARIANCE_MATRIX_FILE]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing required file: {file_path}\n"
                "Please run 02_intermediate_portfolio_analysis.py first."
            )

    # Daily returns are loaded mainly to select all available stock tickers.
    daily_returns = pd.read_csv(DAILY_RETURNS_FILE, index_col=0, parse_dates=True)

    # The summary table already contains annual expected return for each stock.
    summary = pd.read_csv(SUMMARY_TABLE_FILE)

    # Convert the summary table into a Series indexed by stock ticker.
    annual_expected_returns = summary.set_index("Stock Ticker")["Annual Expected Return"]

    # The covariance matrix from step 2 is based on daily returns.
    daily_covariance_matrix = pd.read_csv(COVARIANCE_MATRIX_FILE, index_col=0)

    # Select only tickers that exist in all required inputs.
    available_tickers = [
        ticker
        for ticker in daily_returns.columns
        if ticker in annual_expected_returns.index and ticker in daily_covariance_matrix.columns
    ]

    # Align all data in the same ticker order so matrix multiplication is correct.
    daily_returns = daily_returns[available_tickers]
    annual_expected_returns = annual_expected_returns.loc[available_tickers]
    daily_covariance_matrix = daily_covariance_matrix.loc[available_tickers, available_tickers]

    return daily_returns, annual_expected_returns, daily_covariance_matrix


def create_random_weights(tickers: list[str], seed: int) -> pd.Series:
    """Create random portfolio weights that add up to exactly 1."""
    rng = np.random.default_rng(seed)

    # Generate one random number for each stock.
    raw_weights = rng.random(len(tickers))

    # Divide by the total so all weights add up to 1.
    normalized_weights = raw_weights / raw_weights.sum()

    return pd.Series(normalized_weights, index=tickers, name="Weight")


def calculate_portfolio_metrics(
    weights: pd.Series,
    annual_expected_returns: pd.Series,
    daily_covariance_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> dict[str, float]:
    """Calculate portfolio annual return, annual risk, and Sharpe ratio."""
    # Formula: Portfolio Return = sum(weights x expected returns)
    portfolio_return = float(np.sum(weights * annual_expected_returns))

    # Convert daily covariance into annual covariance.
    annual_covariance_matrix = daily_covariance_matrix * TRADING_DAYS

    # Formula: Portfolio Risk = sqrt(weights.T x covariance matrix x weights)
    portfolio_variance = float(weights.T @ annual_covariance_matrix @ weights)
    portfolio_risk = float(np.sqrt(portfolio_variance))

    # Formula: Sharpe Ratio = (Portfolio Return - Risk Free Rate) / Portfolio Risk
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk

    return {
        "Portfolio Return": portfolio_return,
        "Portfolio Risk": portfolio_risk,
        "Sharpe Ratio": sharpe_ratio,
        "Risk-Free Rate": risk_free_rate,
    }


def format_percent(value: float) -> str:
    """Convert a decimal value into percentage text."""
    return f"{value * 100:.2f}%"


def color_text(value: float, text: str) -> str:
    """Return terminal text using pure green for positive and pure red for negative."""
    if value >= 0:
        return f"\033[38;2;0;255;0m{text}\033[0m"
    return f"\033[38;2;255;0;0m{text}\033[0m"


def create_weight_table(weights: pd.Series) -> pd.DataFrame:
    """Create a beginner-friendly allocation table."""
    allocation_table = pd.DataFrame(
        {
            "Stock Ticker": weights.index,
            "Weight": weights.values,
            "Allocation (%)": weights.values * 100,
        }
    )

    # Show highest allocation first because it is easier to read.
    allocation_table = allocation_table.sort_values("Weight", ascending=False)

    return allocation_table


def print_allocation_table(allocation_table: pd.DataFrame) -> None:
    """Print the random portfolio weights in percentage format."""
    display_table = allocation_table.copy()
    display_table["Weight"] = display_table["Weight"].map(lambda value: f"{value:.4f}")
    display_table["Allocation (%)"] = display_table["Allocation (%)"].map(lambda value: f"{value:.2f}%")

    print("\nRandom portfolio allocation table:")
    print(display_table.to_string(index=False))


def apply_black_chart_theme(ax: plt.Axes) -> None:
    """Apply pure black background and pure white chart text."""
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


def save_or_show_chart(fig: plt.Figure) -> None:
    """Show charts only when the backend supports it, then close the figure."""
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def plot_allocation_chart(allocation_table: pd.DataFrame, metrics: dict[str, float]) -> None:
    """Create a horizontal bar chart of portfolio allocation."""
    fig, ax = plt.subplots(figsize=(12, 14))

    plot_data = allocation_table.sort_values("Allocation (%)", ascending=True)

    ax.barh(
        plot_data["Stock Ticker"],
        plot_data["Allocation (%)"],
        color=PURE_GREEN,
        edgecolor=PURE_WHITE,
        linewidth=0.4,
    )

    ax.set_title("Random Portfolio Stock Allocation", fontsize=16)
    ax.set_xlabel("Allocation (%)")
    ax.set_ylabel("Stock Ticker")

    apply_black_chart_theme(ax)

    # Show the important portfolio results inside the chart.
    return_color = PURE_GREEN if metrics["Portfolio Return"] >= 0 else PURE_RED
    sharpe_color = PURE_GREEN if metrics["Sharpe Ratio"] >= 0 else PURE_RED

    ax.text(
        0.97,
        0.04,
        f"Return: {format_percent(metrics['Portfolio Return'])}",
        color=return_color,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        0.97,
        0.01,
        f"Risk: {format_percent(metrics['Portfolio Risk'])} | Sharpe: {metrics['Sharpe Ratio']:.2f}",
        color=sharpe_color,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(ALLOCATION_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)

    print(f"\nAllocation chart saved to: {ALLOCATION_CHART_FILE}")


def create_random_portfolio_comparison(
    tickers: list[str],
    annual_expected_returns: pd.Series,
    daily_covariance_matrix: pd.DataFrame,
    risk_free_rate: float,
    number_of_portfolios: int = 8,
) -> pd.DataFrame:
    """Create multiple random portfolios to show that weights change results."""
    comparison_rows = []

    for portfolio_number in range(1, number_of_portfolios + 1):
        weights = create_random_weights(tickers, seed=100 + portfolio_number)
        metrics = calculate_portfolio_metrics(
            weights,
            annual_expected_returns,
            daily_covariance_matrix,
            risk_free_rate,
        )

        comparison_rows.append(
            {
                "Portfolio": f"Random Portfolio {portfolio_number}",
                "Portfolio Return": metrics["Portfolio Return"],
                "Portfolio Risk": metrics["Portfolio Risk"],
                "Sharpe Ratio": metrics["Sharpe Ratio"],
                "Largest Stock Allocation": weights.max(),
                "Largest Stock": weights.idxmax(),
            }
        )

    return pd.DataFrame(comparison_rows)


def plot_risk_return_comparison(comparison_table: pd.DataFrame) -> None:
    """Plot different random portfolios to show changing risk and return."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = np.where(comparison_table["Portfolio Return"] >= 0, PURE_GREEN, PURE_RED)

    ax.scatter(
        comparison_table["Portfolio Risk"] * 100,
        comparison_table["Portfolio Return"] * 100,
        s=140,
        c=colors,
        edgecolors=PURE_WHITE,
        linewidths=0.8,
    )

    for _, row in comparison_table.iterrows():
        label = row["Portfolio"].replace("Random Portfolio ", "P")
        ax.annotate(
            label,
            (row["Portfolio Risk"] * 100, row["Portfolio Return"] * 100),
            textcoords="offset points",
            xytext=(6, 6),
            color=PURE_WHITE,
            fontsize=9,
        )

    ax.set_title("Risk and Return Change When Portfolio Weights Change", fontsize=16)
    ax.set_xlabel("Portfolio Risk / Annual Volatility (%)")
    ax.set_ylabel("Portfolio Expected Annual Return (%)")

    apply_black_chart_theme(ax)

    plt.tight_layout()
    plt.savefig(RISK_RETURN_COMPARISON_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)

    print(f"Risk-return comparison chart saved to: {RISK_RETURN_COMPARISON_CHART_FILE}")


def print_portfolio_result(metrics: dict[str, float]) -> None:
    """Display portfolio return, risk, and Sharpe ratio clearly."""
    portfolio_return_text = format_percent(metrics["Portfolio Return"])
    portfolio_risk_text = format_percent(metrics["Portfolio Risk"])
    risk_free_rate_text = format_percent(metrics["Risk-Free Rate"])
    sharpe_text = f"{metrics['Sharpe Ratio']:.4f}"

    print("\nRandom portfolio result:")
    print(f"Portfolio Return: {color_text(metrics['Portfolio Return'], portfolio_return_text)}")
    print(f"Portfolio Risk:   {portfolio_risk_text}")
    print(f"Risk-Free Rate:   {risk_free_rate_text}")
    print(f"Sharpe Ratio:     {color_text(metrics['Sharpe Ratio'], sharpe_text)}")


# -----------------------------
# 3. Main workflow
# -----------------------------

def main() -> None:
    """Run the random portfolio calculation module."""
    create_output_folders()

    daily_returns, annual_expected_returns, daily_covariance_matrix = load_previous_analysis_outputs()

    tickers = list(daily_returns.columns)

    print("All available NSE stocks selected from the cleaned dataset.")
    print(f"Number of stocks used: {len(tickers)}")

    weights = create_random_weights(tickers, RANDOM_SEED)

    print(f"\nWeight check. Sum of all weights = {weights.sum():.6f}")

    allocation_table = create_weight_table(weights)
    allocation_table.to_csv(RANDOM_WEIGHTS_FILE, index=False)

    print_allocation_table(allocation_table)
    print(f"\nRandom portfolio weights saved to: {RANDOM_WEIGHTS_FILE}")

    metrics = calculate_portfolio_metrics(
        weights,
        annual_expected_returns,
        daily_covariance_matrix,
        RISK_FREE_RATE,
    )

    result_table = pd.DataFrame([metrics])
    result_table.to_csv(RANDOM_RESULT_FILE, index=False)

    print_portfolio_result(metrics)
    print(f"\nRandom portfolio result saved to: {RANDOM_RESULT_FILE}")

    plot_allocation_chart(allocation_table, metrics)

    comparison_table = create_random_portfolio_comparison(
        tickers,
        annual_expected_returns,
        daily_covariance_matrix,
        RISK_FREE_RATE,
    )
    comparison_table.to_csv(RANDOM_COMPARISON_FILE, index=False)

    print("\nDifferent random weights create different portfolio results:")
    display_comparison = comparison_table.copy()
    for column in ["Portfolio Return", "Portfolio Risk", "Largest Stock Allocation"]:
        display_comparison[column] = display_comparison[column].map(format_percent)
    display_comparison["Sharpe Ratio"] = display_comparison["Sharpe Ratio"].map(lambda value: f"{value:.4f}")
    print(display_comparison.to_string(index=False))
    print(f"\nRandom portfolio comparison saved to: {RANDOM_COMPARISON_FILE}")

    plot_risk_return_comparison(comparison_table)

    print("\nRandom portfolio calculation completed successfully.")


if __name__ == "__main__":
    main()
