"""
Minimum Volatility Portfolio optimization using scipy.optimize.

This script finds the long-only portfolio with the lowest possible annual
volatility under realistic beginner-friendly constraints:

1. All weights must add up to 1.
2. No short selling is allowed.
3. Each stock weight must be between 0 and 1.
4. The portfolio must be fully invested.

Run this file after step 2. To compare with the maximum Sharpe portfolio,
run step 6 first:

    python scripts/02_intermediate_portfolio_analysis.py
    python scripts/06_scipy_max_sharpe_optimization.py
    python scripts/07_scipy_min_volatility_optimization.py
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

# Risk-free rate used only for Sharpe ratio comparison.
RISK_FREE_RATE = 0.06

# Approximate number of trading days in one year.
TRADING_DAYS = 252

# Very tiny weights below 0.10% are displayed as 0 for readability.
TINY_WEIGHT_THRESHOLD = 0.001

# Input file from step 2.
DAILY_RETURNS_FILE = Path("data/outputs/daily_returns.csv")

# Optional comparison input from step 6.
MAX_SHARPE_RESULT_FILE = Path("data/outputs/scipy_max_sharpe_result.csv")

# Output folders.
OUTPUT_DATA_DIR = Path("data/outputs")
FIGURE_OUTPUT_DIR = Path("reports/figures")

# Output files.
MIN_VOL_ALLOCATION_FILE = OUTPUT_DATA_DIR / "scipy_min_volatility_allocation.csv"
MIN_VOL_RESULT_FILE = OUTPUT_DATA_DIR / "scipy_min_volatility_result.csv"
OPTIMIZED_COMPARISON_FILE = OUTPUT_DATA_DIR / "scipy_optimized_portfolio_comparison.csv"
MIN_VOL_CHART_FILE = FIGURE_OUTPUT_DIR / "scipy_min_volatility_allocation.png"
OPTIMIZED_COMPARISON_CHART_FILE = FIGURE_OUTPUT_DIR / "scipy_optimized_portfolio_comparison.png"

# Pure colors requested for the project.
PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"
PURE_GREEN = "#00FF00"
PURE_RED = "#FF0000"


# -----------------------------
# 2. Data loading and inputs
# -----------------------------

def create_output_folders() -> None:
    """Create output folders if they do not already exist."""
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_daily_returns() -> pd.DataFrame:
    """Load daily returns created in the intermediate analysis step."""
    if not DAILY_RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {DAILY_RETURNS_FILE}\n"
            "Please run 02_intermediate_portfolio_analysis.py first."
        )

    daily_returns = pd.read_csv(DAILY_RETURNS_FILE, index_col=0, parse_dates=True)
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
    """Calculate portfolio annual expected return."""
    return float(np.sum(weights * annual_expected_returns.to_numpy()))


def calculate_portfolio_risk(weights: np.ndarray, annual_covariance_matrix: pd.DataFrame) -> float:
    """Calculate portfolio annual volatility."""
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

    if portfolio_risk == 0:
        return -np.inf

    return (portfolio_return - risk_free_rate) / portfolio_risk


# -----------------------------
# 4. Optimization function
# -----------------------------

def optimize_minimum_volatility(annual_covariance_matrix: pd.DataFrame) -> np.ndarray:
    """Use SciPy SLSQP to find the minimum volatility portfolio."""
    number_of_stocks = len(annual_covariance_matrix)

    # Equal weights are a simple starting point.
    initial_weights = np.repeat(1 / number_of_stocks, number_of_stocks)

    # No short selling: each stock weight must stay between 0 and 1.
    bounds = tuple((0, 1) for _ in range(number_of_stocks))

    # Fully invested portfolio: total weight must equal 1.
    constraints = (
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1,
        },
    )

    result = minimize(
        fun=calculate_portfolio_risk,
        x0=initial_weights,
        args=(annual_covariance_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"SciPy minimum volatility optimization failed: {result.message}")

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

    # Show very tiny weights as 0 to keep the report clean.
    allocation_table["Display Weight"] = allocation_table["Optimized Weight"].where(
        allocation_table["Optimized Weight"] >= TINY_WEIGHT_THRESHOLD,
        0,
    )

    allocation_table["Allocation (%)"] = allocation_table["Display Weight"] * 100
    allocation_table = allocation_table.sort_values("Display Weight", ascending=False)

    return allocation_table


def format_percent(value: float) -> str:
    """Convert decimal value into percentage format."""
    return f"{value * 100:.2f}%"


def color_text(value: float, text: str) -> str:
    """Return green terminal text for positive values and red for negative values."""
    if value >= 0:
        return f"\033[38;2;0;255;0m{text}\033[0m"
    return f"\033[38;2;255;0;0m{text}\033[0m"


def print_min_vol_results(
    portfolio_return: float,
    portfolio_risk: float,
    sharpe_ratio: float,
    allocation_table: pd.DataFrame,
) -> None:
    """Print minimum volatility portfolio results clearly."""
    print("\nOptimized Minimum Volatility Portfolio")
    print("--------------------------------------")
    print(f"Minimum Volatility Portfolio Return: {color_text(portfolio_return, format_percent(portfolio_return))}")
    print(f"Minimum Volatility Portfolio Risk:   {format_percent(portfolio_risk)}")
    print(f"Minimum Volatility Sharpe Ratio:     {color_text(sharpe_ratio, f'{sharpe_ratio:.4f}')}")

    display_table = allocation_table.copy()
    display_table["Optimized Weight"] = display_table["Optimized Weight"].map(lambda value: f"{value:.6f}")
    display_table["Display Weight"] = display_table["Display Weight"].map(lambda value: f"{value:.6f}")
    display_table["Allocation (%)"] = display_table["Allocation (%)"].map(lambda value: f"{value:.2f}%")

    print("\nMinimum volatility allocation table:")
    print(display_table.to_string(index=False))


def create_comparison_table(
    min_vol_return: float,
    min_vol_risk: float,
    min_vol_sharpe: float,
) -> pd.DataFrame:
    """Create comparison table between min volatility and max Sharpe portfolios."""
    rows = [
        {
            "Portfolio": "Minimum Volatility Portfolio",
            "Annual Return": min_vol_return,
            "Annual Risk": min_vol_risk,
            "Sharpe Ratio": min_vol_sharpe,
        }
    ]

    if MAX_SHARPE_RESULT_FILE.exists():
        max_sharpe_result = pd.read_csv(MAX_SHARPE_RESULT_FILE).iloc[0]
        rows.append(
            {
                "Portfolio": "Maximum Sharpe Portfolio",
                "Annual Return": max_sharpe_result["Optimized Portfolio Return"],
                "Annual Risk": max_sharpe_result["Optimized Portfolio Risk"],
                "Sharpe Ratio": max_sharpe_result["Optimized Sharpe Ratio"],
            }
        )
    else:
        print(
            "\nMaximum Sharpe result file was not found, so comparison includes only "
            "the minimum volatility portfolio. Run 06_scipy_max_sharpe_optimization.py "
            "to create the comparison."
        )

    return pd.DataFrame(rows)


def print_comparison_table(comparison_table: pd.DataFrame) -> None:
    """Print the optimized portfolio comparison in readable format."""
    display_table = comparison_table.copy()
    display_table["Annual Return"] = display_table["Annual Return"].map(format_percent)
    display_table["Annual Risk"] = display_table["Annual Risk"].map(format_percent)
    display_table["Sharpe Ratio"] = display_table["Sharpe Ratio"].map(lambda value: f"{value:.4f}")

    print("\nComparison with maximum Sharpe portfolio:")
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


def plot_min_volatility_allocation(
    allocation_table: pd.DataFrame,
    portfolio_return: float,
    portfolio_risk: float,
    sharpe_ratio: float,
) -> None:
    """Create a readable bar chart of minimum volatility allocation."""
    plot_data = allocation_table[allocation_table["Display Weight"] > 0].copy()

    if plot_data.empty:
        print("No visible allocations to plot after applying tiny-weight threshold.")
        return

    plot_data = plot_data.sort_values("Allocation (%)", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor(PURE_BLACK)

    ax.barh(
        plot_data["Stock Ticker"],
        plot_data["Allocation (%)"],
        color=PURE_GREEN,
        edgecolor=PURE_WHITE,
        linewidth=0.6,
    )

    ax.set_title("SciPy Optimized Minimum Volatility Portfolio Allocation", fontsize=16)
    ax.set_xlabel("Allocation (%)")
    ax.set_ylabel("Stock Ticker")

    apply_black_chart_theme(ax)

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
    plt.savefig(MIN_VOL_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)

    print(f"\nMinimum volatility allocation chart saved to: {MIN_VOL_CHART_FILE}")


def plot_optimized_comparison(comparison_table: pd.DataFrame) -> None:
    """Create a simple risk-return comparison chart."""
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(PURE_BLACK)

    colors = [PURE_GREEN if value >= 0 else PURE_RED for value in comparison_table["Annual Return"]]

    ax.scatter(
        comparison_table["Annual Risk"] * 100,
        comparison_table["Annual Return"] * 100,
        s=220,
        color=colors,
        edgecolors=PURE_WHITE,
        linewidths=1.2,
    )

    for _, row in comparison_table.iterrows():
        ax.annotate(
            row["Portfolio"],
            (row["Annual Risk"] * 100, row["Annual Return"] * 100),
            textcoords="offset points",
            xytext=(8, 8),
            color=PURE_WHITE,
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Minimum Volatility vs Maximum Sharpe Portfolio", fontsize=15)
    ax.set_xlabel("Annual Risk / Volatility (%)")
    ax.set_ylabel("Annual Expected Return (%)")

    apply_black_chart_theme(ax)

    plt.tight_layout()
    plt.savefig(OPTIMIZED_COMPARISON_CHART_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)

    print(f"Optimized portfolio comparison chart saved to: {OPTIMIZED_COMPARISON_CHART_FILE}")


# -----------------------------
# 6. Main workflow
# -----------------------------

def main() -> None:
    """Run minimum volatility portfolio optimization."""
    create_output_folders()

    daily_returns = load_daily_returns()
    annual_expected_returns, annual_covariance_matrix = calculate_annual_inputs(daily_returns)

    tickers = list(annual_expected_returns.index)

    print("SciPy minimum volatility optimization started.")
    print(f"Number of NSE stocks used: {len(tickers)}")
    print(f"Risk-free rate assumption for Sharpe comparison: {format_percent(RISK_FREE_RATE)}")

    min_vol_weights = optimize_minimum_volatility(annual_covariance_matrix)

    min_vol_return = calculate_portfolio_return(min_vol_weights, annual_expected_returns)
    min_vol_risk = calculate_portfolio_risk(min_vol_weights, annual_covariance_matrix)
    min_vol_sharpe = calculate_sharpe_ratio(
        min_vol_weights,
        annual_expected_returns,
        annual_covariance_matrix,
        RISK_FREE_RATE,
    )

    allocation_table = create_allocation_table(tickers, min_vol_weights)
    allocation_table.to_csv(MIN_VOL_ALLOCATION_FILE, index=False)

    result_table = pd.DataFrame(
        [
            {
                "Minimum Volatility Portfolio Return": min_vol_return,
                "Minimum Volatility Portfolio Risk": min_vol_risk,
                "Minimum Volatility Sharpe Ratio": min_vol_sharpe,
                "Risk-Free Rate": RISK_FREE_RATE,
                "Exact Weight Sum": min_vol_weights.sum(),
                "Displayed Weight Sum": allocation_table["Display Weight"].sum(),
            }
        ]
    )
    result_table.to_csv(MIN_VOL_RESULT_FILE, index=False)

    print_min_vol_results(
        min_vol_return,
        min_vol_risk,
        min_vol_sharpe,
        allocation_table,
    )

    comparison_table = create_comparison_table(
        min_vol_return,
        min_vol_risk,
        min_vol_sharpe,
    )
    comparison_table.to_csv(OPTIMIZED_COMPARISON_FILE, index=False)

    print(f"\nMinimum volatility allocation saved to: {MIN_VOL_ALLOCATION_FILE}")
    print(f"Minimum volatility result saved to: {MIN_VOL_RESULT_FILE}")
    print(f"Optimized comparison saved to: {OPTIMIZED_COMPARISON_FILE}")
    print(f"Exact optimized weight sum: {min_vol_weights.sum():.6f}")
    print(f"Displayed weight sum after tiny-weight cleanup: {allocation_table['Display Weight'].sum():.6f}")

    print_comparison_table(comparison_table)

    plot_min_volatility_allocation(
        allocation_table,
        min_vol_return,
        min_vol_risk,
        min_vol_sharpe,
    )
    plot_optimized_comparison(comparison_table)

    print("\nSciPy minimum volatility optimization completed successfully.")


if __name__ == "__main__":
    main()
