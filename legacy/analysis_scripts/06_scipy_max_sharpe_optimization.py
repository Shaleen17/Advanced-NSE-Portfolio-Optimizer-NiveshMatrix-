"""
Maximum Sharpe Ratio optimization using scipy.optimize.

This script uses daily returns from the previous analysis step, calculates
annual expected returns and the annual covariance matrix, then uses SciPy's
SLSQP optimizer to find the long-only portfolio with the maximum Sharpe ratio.

Run this file after step 2:

    python scripts/02_intermediate_portfolio_analysis.py
    python scripts/06_scipy_max_sharpe_optimization.py
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
from scipy.optimize import minimize


# -----------------------------
# 1. Project settings
# -----------------------------

# India-friendly editable risk-free rate assumption.
RISK_FREE_RATE = 0.06

# Approximate number of trading days in one year.
TRADING_DAYS = 252

# Very tiny optimized weights below 0.10% are displayed as 0 for readability.
TINY_WEIGHT_THRESHOLD = 0.001

# Input file from step 2.
DAILY_RETURNS_FILE = Path("data/outputs/daily_returns.csv")

# Output folders.
OUTPUT_DATA_DIR = Path("data/outputs")
FIGURE_OUTPUT_DIR = Path("reports/figures")

# Output files.
OPTIMIZED_ALLOCATION_FILE = OUTPUT_DATA_DIR / "scipy_max_sharpe_allocation.csv"
OPTIMIZED_RESULT_FILE = OUTPUT_DATA_DIR / "scipy_max_sharpe_result.csv"
OPTIMIZED_CHART_FILE = FIGURE_OUTPUT_DIR / "scipy_max_sharpe_allocation.png"

# Pure colors requested for the project.
PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"
PURE_GREEN = "#00FF00"
PURE_RED = "#FF0000"


# -----------------------------
# 2. Data loading and inputs
# -----------------------------

def create_output_folders() -> None:
    """Create folders for output tables and charts."""
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_daily_returns() -> pd.DataFrame:
    """Load daily returns created in the intermediate analysis step."""
    if not DAILY_RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {DAILY_RETURNS_FILE}\n"
            "Please run 02_intermediate_portfolio_analysis.py first."
        )

    # The Date column becomes the index.
    daily_returns = pd.read_csv(DAILY_RETURNS_FILE, index_col=0, parse_dates=True)

    # Remove any accidental missing rows.
    daily_returns = daily_returns.dropna()

    return daily_returns


def calculate_annual_inputs(daily_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate annual expected returns and annual covariance matrix."""
    # Formula: Annual Return = mean daily return x 252
    annual_expected_returns = daily_returns.mean() * TRADING_DAYS

    # Formula: Annual Covariance Matrix = daily covariance matrix x 252
    annual_covariance_matrix = daily_returns.cov() * TRADING_DAYS

    return annual_expected_returns, annual_covariance_matrix


# -----------------------------
# 3. Portfolio math functions
# -----------------------------

def calculate_portfolio_return(weights: np.ndarray, annual_expected_returns: pd.Series) -> float:
    """Calculate portfolio expected annual return."""
    # Formula: Portfolio Return = sum(weights x annual expected returns)
    return float(np.sum(weights * annual_expected_returns.to_numpy()))


def calculate_portfolio_risk(weights: np.ndarray, annual_covariance_matrix: pd.DataFrame) -> float:
    """Calculate portfolio annual risk/volatility."""
    covariance_values = annual_covariance_matrix.to_numpy()

    # Formula: Portfolio Risk = sqrt(weights.T x annual covariance matrix x weights)
    portfolio_variance = float(weights.T @ covariance_values @ weights)

    # max protects against tiny negative floating-point noise.
    return float(np.sqrt(max(portfolio_variance, 0)))


def calculate_sharpe_ratio(
    weights: np.ndarray,
    annual_expected_returns: pd.Series,
    annual_covariance_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """Calculate portfolio Sharpe ratio."""
    portfolio_return = calculate_portfolio_return(weights, annual_expected_returns)
    portfolio_risk = calculate_portfolio_risk(weights, annual_covariance_matrix)

    # Avoid division by zero if a portfolio somehow has no volatility.
    if portfolio_risk == 0:
        return -np.inf

    # Formula: Sharpe Ratio = (Portfolio Return - Risk Free Rate) / Portfolio Risk
    return (portfolio_return - risk_free_rate) / portfolio_risk


def negative_sharpe_ratio(
    weights: np.ndarray,
    annual_expected_returns: pd.Series,
    annual_covariance_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """
    Return negative Sharpe ratio because scipy.optimize.minimize minimizes.

    We want the highest Sharpe ratio. SciPy's minimize function searches for
    the lowest value, so we multiply Sharpe ratio by -1. The lowest negative
    Sharpe ratio is the same as the highest positive Sharpe ratio.
    """
    return -calculate_sharpe_ratio(
        weights,
        annual_expected_returns,
        annual_covariance_matrix,
        risk_free_rate,
    )


# -----------------------------
# 4. Optimization function
# -----------------------------

def optimize_max_sharpe(
    annual_expected_returns: pd.Series,
    annual_covariance_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> np.ndarray:
    """Use SciPy SLSQP to find the maximum Sharpe ratio portfolio."""
    number_of_stocks = len(annual_expected_returns)

    # Equal weights are a simple beginner-friendly starting point.
    initial_weights = np.repeat(1 / number_of_stocks, number_of_stocks)

    # No short selling: every stock weight must be between 0 and 1.
    bounds = tuple((0, 1) for _ in range(number_of_stocks))

    # Fully invested portfolio: all weights must add up to 1.
    constraints = (
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1,
        },
    )

    result = minimize(
        fun=negative_sharpe_ratio,
        x0=initial_weights,
        args=(annual_expected_returns, annual_covariance_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"SciPy optimization failed: {result.message}")

    return result.x


# -----------------------------
# 5. Output and visualization
# -----------------------------

def create_allocation_table(tickers: list[str], optimized_weights: np.ndarray) -> pd.DataFrame:
    """Create a clean allocation table from optimized weights."""
    allocation_table = pd.DataFrame(
        {
            "Stock Ticker": tickers,
            "Optimized Weight": optimized_weights,
        }
    )

    # Values below the tiny threshold are displayed as 0 for a cleaner report table.
    allocation_table["Display Weight"] = allocation_table["Optimized Weight"].where(
        allocation_table["Optimized Weight"] >= TINY_WEIGHT_THRESHOLD,
        0,
    )

    allocation_table["Allocation (%)"] = allocation_table["Display Weight"] * 100

    # Highest allocation first.
    allocation_table = allocation_table.sort_values("Display Weight", ascending=False)

    return allocation_table


def format_percent(value: float) -> str:
    """Convert decimal number to percentage format."""
    return f"{value * 100:.2f}%"


def color_text(value: float, text: str) -> str:
    """Return green terminal text for positive values and red for negative values."""
    if value >= 0:
        return f"\033[38;2;0;255;0m{text}\033[0m"
    return f"\033[38;2;255;0;0m{text}\033[0m"


def print_optimized_results(
    portfolio_return: float,
    portfolio_risk: float,
    sharpe_ratio: float,
    allocation_table: pd.DataFrame,
) -> None:
    """Print optimized portfolio metrics and allocation clearly."""
    print("\nOptimized Maximum Sharpe Ratio Portfolio")
    print("----------------------------------------")
    print(f"Optimized Portfolio Return: {color_text(portfolio_return, format_percent(portfolio_return))}")
    print(f"Optimized Portfolio Risk:   {format_percent(portfolio_risk)}")
    print(f"Optimized Sharpe Ratio:     {color_text(sharpe_ratio, f'{sharpe_ratio:.4f}')}")

    display_table = allocation_table.copy()
    display_table["Optimized Weight"] = display_table["Optimized Weight"].map(lambda value: f"{value:.6f}")
    display_table["Display Weight"] = display_table["Display Weight"].map(lambda value: f"{value:.6f}")
    display_table["Allocation (%)"] = display_table["Allocation (%)"].map(lambda value: f"{value:.2f}%")

    print("\nOptimized allocation table:")
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
    """Show chart only if the current backend supports interactive display."""
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def plot_optimized_allocation(
    allocation_table: pd.DataFrame,
    portfolio_return: float,
    portfolio_risk: float,
    sharpe_ratio: float,
) -> None:
    """Create a readable bar chart of optimized portfolio allocation."""
    plot_data = allocation_table[allocation_table["Display Weight"] > 0].copy()

    if plot_data.empty:
        print("No visible allocations to plot after applying tiny-weight threshold.")
        return

    plot_data = plot_data.sort_values("Allocation (%)", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(PURE_BLACK)

    ax.barh(
        plot_data["Stock Ticker"],
        plot_data["Allocation (%)"],
        color=PURE_GREEN,
        edgecolor=PURE_WHITE,
        linewidth=0.6,
    )

    ax.set_title("SciPy Optimized Maximum Sharpe Portfolio Allocation", fontsize=16)
    ax.set_xlabel("Allocation (%)")
    ax.set_ylabel("Stock Ticker")

    apply_black_chart_theme(ax)

    # Add important results directly on the chart.
    return_color = PURE_GREEN if portfolio_return >= 0 else PURE_RED
    sharpe_color = PURE_GREEN if sharpe_ratio >= 0 else PURE_RED

    ax.text(
        0.98,
        0.06,
        f"Return: {format_percent(portfolio_return)}",
        transform=ax.transAxes,
        color=return_color,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        0.98,
        0.02,
        f"Risk: {format_percent(portfolio_risk)} | Sharpe: {sharpe_ratio:.2f}",
        transform=ax.transAxes,
        color=sharpe_color,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(OPTIMIZED_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)

    print(f"\nOptimized allocation chart saved to: {OPTIMIZED_CHART_FILE}")


# -----------------------------
# 6. Main workflow
# -----------------------------

def main() -> None:
    """Run maximum Sharpe ratio optimization."""
    create_output_folders()

    daily_returns = load_daily_returns()

    annual_expected_returns, annual_covariance_matrix = calculate_annual_inputs(daily_returns)

    tickers = list(annual_expected_returns.index)

    print("SciPy maximum Sharpe ratio optimization started.")
    print(f"Number of NSE stocks used: {len(tickers)}")
    print(f"Risk-free rate assumption: {format_percent(RISK_FREE_RATE)}")

    optimized_weights = optimize_max_sharpe(
        annual_expected_returns,
        annual_covariance_matrix,
        RISK_FREE_RATE,
    )

    optimized_return = calculate_portfolio_return(optimized_weights, annual_expected_returns)
    optimized_risk = calculate_portfolio_risk(optimized_weights, annual_covariance_matrix)
    optimized_sharpe = calculate_sharpe_ratio(
        optimized_weights,
        annual_expected_returns,
        annual_covariance_matrix,
        RISK_FREE_RATE,
    )

    allocation_table = create_allocation_table(tickers, optimized_weights)
    allocation_table.to_csv(OPTIMIZED_ALLOCATION_FILE, index=False)

    result_table = pd.DataFrame(
        [
            {
                "Optimized Portfolio Return": optimized_return,
                "Optimized Portfolio Risk": optimized_risk,
                "Optimized Sharpe Ratio": optimized_sharpe,
                "Risk-Free Rate": RISK_FREE_RATE,
                "Exact Weight Sum": optimized_weights.sum(),
                "Displayed Weight Sum": allocation_table["Display Weight"].sum(),
            }
        ]
    )
    result_table.to_csv(OPTIMIZED_RESULT_FILE, index=False)

    print_optimized_results(
        optimized_return,
        optimized_risk,
        optimized_sharpe,
        allocation_table,
    )

    print(f"\nOptimized allocation saved to: {OPTIMIZED_ALLOCATION_FILE}")
    print(f"Optimized result saved to: {OPTIMIZED_RESULT_FILE}")
    print(f"Exact optimized weight sum: {optimized_weights.sum():.6f}")
    print(f"Displayed weight sum after tiny-weight cleanup: {allocation_table['Display Weight'].sum():.6f}")

    plot_optimized_allocation(
        allocation_table,
        optimized_return,
        optimized_risk,
        optimized_sharpe,
    )

    print("\nSciPy maximum Sharpe ratio optimization completed successfully.")


if __name__ == "__main__":
    main()
