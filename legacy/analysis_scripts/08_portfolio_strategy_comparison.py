"""
Portfolio Strategy Comparison for the NSE portfolio optimization project.

This script compares four strategies:

1. Equal Weight Portfolio
2. One Random Portfolio
3. Maximum Sharpe Ratio Portfolio
4. Minimum Volatility Portfolio

All strategies are evaluated using the same annual expected returns,
annual covariance matrix, and risk-free rate.

Run this file after steps 2, 3, 6, and 7:

    python scripts/02_intermediate_portfolio_analysis.py
    python scripts/03_random_portfolio_calculation.py
    python scripts/06_scipy_max_sharpe_optimization.py
    python scripts/07_scipy_min_volatility_optimization.py
    python scripts/08_portfolio_strategy_comparison.py

Educational warning:
This project is for educational purposes only and should not be treated as
financial advice.
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

# India-friendly risk-free rate assumption.
RISK_FREE_RATE = 0.06

# Approximate number of trading days in one year.
TRADING_DAYS = 252

# Input files.
DAILY_RETURNS_FILE = Path("data/outputs/daily_returns.csv")
RANDOM_WEIGHTS_FILE = Path("data/outputs/random_portfolio_weights.csv")
MAX_SHARPE_ALLOCATION_FILE = Path("data/outputs/scipy_max_sharpe_allocation.csv")
MIN_VOL_ALLOCATION_FILE = Path("data/outputs/scipy_min_volatility_allocation.csv")

# Output folders.
OUTPUT_DATA_DIR = Path("data/outputs")
FIGURE_OUTPUT_DIR = Path("reports/figures")

# Output files.
STRATEGY_COMPARISON_FILE = OUTPUT_DATA_DIR / "portfolio_strategy_comparison.csv"
STRATEGY_COMPARISON_FORMATTED_FILE = OUTPUT_DATA_DIR / "portfolio_strategy_comparison_formatted.csv"
RETURN_COMPARISON_CHART_FILE = FIGURE_OUTPUT_DIR / "strategy_return_comparison.png"
RISK_COMPARISON_CHART_FILE = FIGURE_OUTPUT_DIR / "strategy_risk_comparison.png"
SHARPE_COMPARISON_CHART_FILE = FIGURE_OUTPUT_DIR / "strategy_sharpe_comparison.png"

# Pure colors requested for the project.
PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"
PURE_GREEN = "#00FF00"
PURE_RED = "#FF0000"


# -----------------------------
# 2. Data loading
# -----------------------------

def create_output_folders() -> None:
    """Create output folders if they do not already exist."""
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def require_file(file_path: Path, setup_hint: str) -> None:
    """Raise a clear error if a required input file is missing."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}\n{setup_hint}")


def load_daily_returns() -> pd.DataFrame:
    """Load daily returns from the intermediate analysis step."""
    require_file(
        DAILY_RETURNS_FILE,
        "Please run 02_intermediate_portfolio_analysis.py first.",
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


def load_random_weights(tickers: list[str]) -> pd.Series:
    """Load the random portfolio weights created in step 3."""
    require_file(
        RANDOM_WEIGHTS_FILE,
        "Please run 03_random_portfolio_calculation.py first.",
    )

    random_weights_table = pd.read_csv(RANDOM_WEIGHTS_FILE)
    weights = random_weights_table.set_index("Stock Ticker")["Weight"]

    return align_and_normalize_weights(weights, tickers)


def load_optimized_weights(file_path: Path, weight_column: str, setup_hint: str, tickers: list[str]) -> pd.Series:
    """Load optimized weights from a SciPy allocation file."""
    require_file(file_path, setup_hint)

    allocation_table = pd.read_csv(file_path)
    weights = allocation_table.set_index("Stock Ticker")[weight_column]

    return align_and_normalize_weights(weights, tickers)


def align_and_normalize_weights(weights: pd.Series, tickers: list[str]) -> pd.Series:
    """Align weight series to ticker order and normalize the total to 1."""
    aligned_weights = weights.reindex(tickers).fillna(0).astype(float)

    total_weight = aligned_weights.sum()
    if total_weight <= 0:
        raise ValueError("Weight total is zero or negative, so portfolio metrics cannot be calculated.")

    # Normalize again to avoid tiny rounding differences from CSV files.
    aligned_weights = aligned_weights / total_weight

    return aligned_weights


# -----------------------------
# 3. Portfolio math
# -----------------------------

def calculate_portfolio_return(weights: pd.Series, annual_expected_returns: pd.Series) -> float:
    """Calculate portfolio annual expected return."""
    return float(np.sum(weights * annual_expected_returns))


def calculate_portfolio_risk(weights: pd.Series, annual_covariance_matrix: pd.DataFrame) -> float:
    """Calculate portfolio annual risk/volatility."""
    weight_values = weights.to_numpy()
    covariance_values = annual_covariance_matrix.loc[weights.index, weights.index].to_numpy()

    portfolio_variance = float(weight_values.T @ covariance_values @ weight_values)
    return float(np.sqrt(max(portfolio_variance, 0)))


def calculate_sharpe_ratio(portfolio_return: float, portfolio_risk: float) -> float:
    """Calculate Sharpe ratio using the project risk-free rate."""
    if portfolio_risk == 0:
        return -np.inf

    return (portfolio_return - RISK_FREE_RATE) / portfolio_risk


def calculate_strategy_metrics(
    strategy_name: str,
    weights: pd.Series,
    annual_expected_returns: pd.Series,
    annual_covariance_matrix: pd.DataFrame,
) -> dict[str, float | str]:
    """Calculate return, risk, and Sharpe ratio for one strategy."""
    portfolio_return = calculate_portfolio_return(weights, annual_expected_returns)
    portfolio_risk = calculate_portfolio_risk(weights, annual_covariance_matrix)
    sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_risk)

    return {
        "Strategy": strategy_name,
        "Annual Return": portfolio_return,
        "Annual Risk": portfolio_risk,
        "Sharpe Ratio": sharpe_ratio,
    }


def build_strategy_comparison(
    tickers: list[str],
    annual_expected_returns: pd.Series,
    annual_covariance_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Create a comparison DataFrame for all four portfolio strategies."""
    number_of_stocks = len(tickers)

    # Strategy 1: Equal weight portfolio.
    equal_weights = pd.Series(
        np.repeat(1 / number_of_stocks, number_of_stocks),
        index=tickers,
        name="Equal Weight",
    )

    # Strategy 2: One random portfolio from step 3.
    random_weights = load_random_weights(tickers)

    # Strategy 3: SciPy optimized maximum Sharpe portfolio from step 6.
    max_sharpe_weights = load_optimized_weights(
        MAX_SHARPE_ALLOCATION_FILE,
        "Optimized Weight",
        "Please run 06_scipy_max_sharpe_optimization.py first.",
        tickers,
    )

    # Strategy 4: SciPy optimized minimum volatility portfolio from step 7.
    min_volatility_weights = load_optimized_weights(
        MIN_VOL_ALLOCATION_FILE,
        "Optimized Weight",
        "Please run 07_scipy_min_volatility_optimization.py first.",
        tickers,
    )

    rows = [
        calculate_strategy_metrics(
            "Equal Weight Portfolio",
            equal_weights,
            annual_expected_returns,
            annual_covariance_matrix,
        ),
        calculate_strategy_metrics(
            "One Random Portfolio",
            random_weights,
            annual_expected_returns,
            annual_covariance_matrix,
        ),
        calculate_strategy_metrics(
            "Maximum Sharpe Ratio Portfolio",
            max_sharpe_weights,
            annual_expected_returns,
            annual_covariance_matrix,
        ),
        calculate_strategy_metrics(
            "Minimum Volatility Portfolio",
            min_volatility_weights,
            annual_expected_returns,
            annual_covariance_matrix,
        ),
    ]

    return pd.DataFrame(rows)


# -----------------------------
# 4. Formatting and printing
# -----------------------------

def format_percent(value: float) -> str:
    """Convert decimal number to percentage format."""
    return f"{value * 100:.2f}%"


def color_text(value: float, text: str) -> str:
    """Return green terminal text for positive values and red for negative values."""
    if value >= 0:
        return f"\033[38;2;0;255;0m{text}\033[0m"
    return f"\033[38;2;255;0;0m{text}\033[0m"


def create_formatted_comparison_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Create a report-friendly formatted comparison table."""
    formatted = comparison.copy()

    formatted["Annual Return"] = formatted["Annual Return"].map(format_percent)
    formatted["Annual Risk"] = formatted["Annual Risk"].map(format_percent)
    formatted["Sharpe Ratio"] = formatted["Sharpe Ratio"].map(lambda value: f"{value:.4f}")

    return formatted


def print_comparison_with_highlights(comparison: pd.DataFrame) -> None:
    """Print comparison table and highlight best return, lowest risk, and highest Sharpe."""
    best_return_strategy = comparison.loc[comparison["Annual Return"].idxmax(), "Strategy"]
    lowest_risk_strategy = comparison.loc[comparison["Annual Risk"].idxmin(), "Strategy"]
    highest_sharpe_strategy = comparison.loc[comparison["Sharpe Ratio"].idxmax(), "Strategy"]

    print("\nPortfolio Strategy Comparison")
    print("-----------------------------")

    for _, row in comparison.iterrows():
        return_text = color_text(row["Annual Return"], format_percent(row["Annual Return"]))
        risk_text = format_percent(row["Annual Risk"])
        sharpe_text = color_text(row["Sharpe Ratio"], f"{row['Sharpe Ratio']:.4f}")

        print(f"\nStrategy: {row['Strategy']}")
        print(f"Annual Return: {return_text}")
        print(f"Annual Risk:   {risk_text}")
        print(f"Sharpe Ratio:  {sharpe_text}")

    print("\nHighlights")
    print("----------")
    print(f"Best annual return:     {best_return_strategy}")
    print(f"Lowest annual risk:     {lowest_risk_strategy}")
    print(f"Highest Sharpe ratio:   {highest_sharpe_strategy}")


def explain_best_strategy(comparison: pd.DataFrame) -> None:
    """Print a beginner-friendly explanation of which strategy is best."""
    best_return_strategy = comparison.loc[comparison["Annual Return"].idxmax()]
    lowest_risk_strategy = comparison.loc[comparison["Annual Risk"].idxmin()]
    highest_sharpe_strategy = comparison.loc[comparison["Sharpe Ratio"].idxmax()]

    print("\nBeginner-friendly interpretation")
    print("--------------------------------")
    print(
        f"The highest return comes from {best_return_strategy['Strategy']} "
        f"with {format_percent(best_return_strategy['Annual Return'])} annual return."
    )
    print(
        f"The lowest risk comes from {lowest_risk_strategy['Strategy']} "
        f"with {format_percent(lowest_risk_strategy['Annual Risk'])} annual volatility."
    )
    print(
        f"The best risk-adjusted strategy is {highest_sharpe_strategy['Strategy']} "
        f"because it has the highest Sharpe ratio of {highest_sharpe_strategy['Sharpe Ratio']:.4f}."
    )
    print(
        "Highest return is not always the best choice if it requires taking too much risk. "
        "Sharpe ratio helps compare return and risk together."
    )
    print(
        "\nWarning: This project is for educational purposes only and should not be "
        "treated as financial advice."
    )


# -----------------------------
# 5. Charts
# -----------------------------

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

    ax.grid(axis="y", color=PURE_WHITE, alpha=0.12)


def save_or_show_chart(fig: plt.Figure) -> None:
    """Show chart only if the current backend supports interactive display."""
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def plot_metric_bar_chart(
    comparison: pd.DataFrame,
    metric_column: str,
    title: str,
    ylabel: str,
    output_file: Path,
    value_is_percent: bool,
    lower_is_better: bool = False,
) -> None:
    """Create one modern dark bar chart for a selected comparison metric."""
    plot_data = comparison.copy()

    values = plot_data[metric_column]
    display_values = values * 100 if value_is_percent else values

    # Use green for positive/profit values and red for negative values.
    colors = [PURE_GREEN if value >= 0 else PURE_RED for value in values]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(PURE_BLACK)

    bars = ax.bar(
        plot_data["Strategy"],
        display_values,
        color=colors,
        edgecolor=PURE_WHITE,
        linewidth=0.7,
    )

    ax.set_title(title, fontsize=16, pad=16)
    ax.set_xlabel("Portfolio Strategy")
    ax.set_ylabel(ylabel)

    apply_black_chart_theme(ax)

    # Rotate labels so strategy names do not overlap.
    plt.xticks(rotation=18, ha="right")

    # Highlight the best bar for this metric.
    if lower_is_better:
        best_index = values.idxmin()
    else:
        best_index = values.idxmax()

    bars[best_index].set_edgecolor(PURE_GREEN)
    bars[best_index].set_linewidth(2.5)

    # Add value labels above each bar.
    for bar, raw_value, display_value in zip(bars, values, display_values):
        if value_is_percent:
            label = f"{display_value:.2f}%"
        else:
            label = f"{display_value:.4f}"

        label_color = PURE_GREEN if raw_value >= 0 else PURE_RED
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            color=label_color,
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)

    print(f"{title} chart saved to: {output_file}")


# -----------------------------
# 6. Main workflow
# -----------------------------

def main() -> None:
    """Run portfolio strategy comparison."""
    create_output_folders()

    daily_returns = load_daily_returns()
    annual_expected_returns, annual_covariance_matrix = calculate_annual_inputs(daily_returns)

    tickers = list(annual_expected_returns.index)

    print("Portfolio strategy comparison started.")
    print(f"Number of NSE stocks used: {len(tickers)}")
    print(f"Risk-free rate assumption: {format_percent(RISK_FREE_RATE)}")

    comparison = build_strategy_comparison(
        tickers,
        annual_expected_returns,
        annual_covariance_matrix,
    )

    comparison.to_csv(STRATEGY_COMPARISON_FILE, index=False)

    formatted_comparison = create_formatted_comparison_table(comparison)
    formatted_comparison.to_csv(STRATEGY_COMPARISON_FORMATTED_FILE, index=False)

    print_comparison_with_highlights(comparison)
    explain_best_strategy(comparison)

    print(f"\nRaw strategy comparison saved to: {STRATEGY_COMPARISON_FILE}")
    print(f"Formatted strategy comparison saved to: {STRATEGY_COMPARISON_FORMATTED_FILE}")

    plot_metric_bar_chart(
        comparison,
        metric_column="Annual Return",
        title="Portfolio Strategy Annual Return Comparison",
        ylabel="Annual Return (%)",
        output_file=RETURN_COMPARISON_CHART_FILE,
        value_is_percent=True,
        lower_is_better=False,
    )

    plot_metric_bar_chart(
        comparison,
        metric_column="Annual Risk",
        title="Portfolio Strategy Annual Risk Comparison",
        ylabel="Annual Risk / Volatility (%)",
        output_file=RISK_COMPARISON_CHART_FILE,
        value_is_percent=True,
        lower_is_better=True,
    )

    plot_metric_bar_chart(
        comparison,
        metric_column="Sharpe Ratio",
        title="Portfolio Strategy Sharpe Ratio Comparison",
        ylabel="Sharpe Ratio",
        output_file=SHARPE_COMPARISON_CHART_FILE,
        value_is_percent=False,
        lower_is_better=False,
    )

    print("\nPortfolio strategy comparison completed successfully.")


if __name__ == "__main__":
    main()
