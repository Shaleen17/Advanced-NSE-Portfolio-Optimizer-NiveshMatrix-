"""
Monte Carlo portfolio simulation for the NSE portfolio optimization project.

This script generates many random portfolios using all available NSE stocks
from the cleaned dataset. For each portfolio, it calculates annual return,
annual risk, and Sharpe ratio. It then identifies the best random portfolio
by Sharpe ratio and the lowest-risk random portfolio.

Important:
Monte Carlo simulation gives a strong idea of possible portfolios, but it does
not guarantee the mathematically best portfolio. In the next step, SciPy
optimization can be used to find the best portfolio more directly.

Run this file after steps 1 and 2:

    python scripts/01_download_clean_plot_nse_prices.py
    python scripts/02_intermediate_portfolio_analysis.py
    python scripts/04_monte_carlo_portfolio_simulation.py
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

# Monte Carlo means trying many random portfolios.
NUMBER_OF_PORTFOLIOS = 10_000

# Risk-free rate assumption for India.
RISK_FREE_RATE = 0.06

# Approximate number of trading days in one year.
TRADING_DAYS = 252

# Fixed random seed makes the simulation reproducible.
RANDOM_SEED = 42

# Input files from previous project steps.
CLEANED_PRICE_FILE = Path("data/processed/nse_adjusted_close_cleaned.csv")
DAILY_RETURNS_FILE = Path("data/outputs/daily_returns.csv")

# Output folders.
OUTPUT_DATA_DIR = Path("data/outputs")
FIGURE_OUTPUT_DIR = Path("reports/figures")

# Output files.
SIMULATION_RESULTS_FILE = OUTPUT_DATA_DIR / "monte_carlo_portfolio_results.csv"
MAX_SHARPE_WEIGHTS_FILE = OUTPUT_DATA_DIR / "monte_carlo_max_sharpe_weights.csv"
MIN_VOLATILITY_WEIGHTS_FILE = OUTPUT_DATA_DIR / "monte_carlo_min_volatility_weights.csv"
MONTE_CARLO_SCATTER_FILE = FIGURE_OUTPUT_DIR / "monte_carlo_risk_return_scatter.png"
BEST_PORTFOLIO_ALLOCATION_FILE = FIGURE_OUTPUT_DIR / "monte_carlo_best_portfolio_allocations.png"

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


def load_project_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned prices and daily returns from previous project steps."""
    if not CLEANED_PRICE_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {CLEANED_PRICE_FILE}\n"
            "Please run 01_download_clean_plot_nse_prices.py first."
        )

    if not DAILY_RETURNS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {DAILY_RETURNS_FILE}\n"
            "Please run 02_intermediate_portfolio_analysis.py first."
        )

    # Load cleaned Adjusted Close prices.
    prices = pd.read_csv(CLEANED_PRICE_FILE, index_col=0, parse_dates=True)

    # Load daily returns created in the intermediate analysis step.
    daily_returns = pd.read_csv(DAILY_RETURNS_FILE, index_col=0, parse_dates=True)

    # Use only stock columns that are available in both datasets.
    available_tickers = [ticker for ticker in prices.columns if ticker in daily_returns.columns]

    prices = prices[available_tickers]
    daily_returns = daily_returns[available_tickers]

    return prices, daily_returns


def calculate_annual_inputs(daily_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate annual expected returns and annual covariance matrix."""
    # Formula: Annual Return = mean daily return x 252
    annual_expected_returns = daily_returns.mean() * TRADING_DAYS

    # Formula: Annual Covariance Matrix = daily covariance matrix x 252
    annual_covariance_matrix = daily_returns.cov() * TRADING_DAYS

    return annual_expected_returns, annual_covariance_matrix


def generate_random_weights(
    number_of_portfolios: int,
    number_of_stocks: int,
    seed: int,
) -> np.ndarray:
    """Generate random portfolio weights where every row adds up to 1."""
    rng = np.random.default_rng(seed)

    # Create a random matrix with one row per portfolio and one column per stock.
    raw_weights = rng.random((number_of_portfolios, number_of_stocks))

    # Divide each row by its row total so every portfolio adds up to 1.
    weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)

    return weights


def run_monte_carlo_simulation(
    tickers: list[str],
    annual_expected_returns: pd.Series,
    annual_covariance_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Generate random portfolios and calculate return, risk, and Sharpe ratio."""
    number_of_stocks = len(tickers)

    weights = generate_random_weights(
        NUMBER_OF_PORTFOLIOS,
        number_of_stocks,
        RANDOM_SEED,
    )

    # Convert annual expected returns into a NumPy array for fast calculation.
    expected_return_values = annual_expected_returns.loc[tickers].to_numpy()

    # Convert annual covariance matrix into a NumPy array for matrix math.
    covariance_values = annual_covariance_matrix.loc[tickers, tickers].to_numpy()

    # Formula: Portfolio Return = sum(weights x annual expected returns)
    portfolio_returns = weights @ expected_return_values

    # Formula: Portfolio Risk = sqrt(weights.T x annual covariance matrix x weights)
    # einsum calculates the portfolio variance for every random portfolio efficiently.
    portfolio_variances = np.einsum("ij,jk,ik->i", weights, covariance_values, weights)
    portfolio_risks = np.sqrt(portfolio_variances)

    # Formula: Sharpe Ratio = (Portfolio Return - risk-free rate) / Portfolio Risk
    sharpe_ratios = (portfolio_returns - RISK_FREE_RATE) / portfolio_risks

    # Store main portfolio metrics.
    results = pd.DataFrame(
        {
            "Portfolio Number": np.arange(1, NUMBER_OF_PORTFOLIOS + 1),
            "Annual Portfolio Return": portfolio_returns,
            "Annual Portfolio Risk": portfolio_risks,
            "Sharpe Ratio": sharpe_ratios,
        }
    )

    # Store every stock weight in the same DataFrame.
    weight_columns = [f"Weight_{ticker}" for ticker in tickers]
    weights_table = pd.DataFrame(weights, columns=weight_columns)

    results = pd.concat([results, weights_table], axis=1)

    return results


def get_weight_columns(results: pd.DataFrame) -> list[str]:
    """Return all weight columns from the simulation result table."""
    return [column for column in results.columns if column.startswith("Weight_")]


def extract_weight_table(portfolio_row: pd.Series, portfolio_name: str) -> pd.DataFrame:
    """Extract stock weights from one selected portfolio row."""
    weight_items = []

    for column, value in portfolio_row.items():
        if column.startswith("Weight_"):
            ticker = column.replace("Weight_", "")
            weight_items.append(
                {
                    "Portfolio": portfolio_name,
                    "Stock Ticker": ticker,
                    "Weight": value,
                    "Allocation (%)": value * 100,
                }
            )

    weight_table = pd.DataFrame(weight_items)
    weight_table = weight_table.sort_values("Weight", ascending=False)

    return weight_table


def format_percent(value: float) -> str:
    """Convert decimal number into percentage format."""
    return f"{value * 100:.2f}%"


def color_text(value: float, text: str) -> str:
    """Return green terminal text for positive values and red for negative values."""
    if value >= 0:
        return f"\033[38;2;0;255;0m{text}\033[0m"
    return f"\033[38;2;255;0;0m{text}\033[0m"


def print_portfolio_result(title: str, portfolio_row: pd.Series) -> None:
    """Print one portfolio result clearly."""
    portfolio_return = portfolio_row["Annual Portfolio Return"]
    portfolio_risk = portfolio_row["Annual Portfolio Risk"]
    sharpe_ratio = portfolio_row["Sharpe Ratio"]

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Portfolio Number: {int(portfolio_row['Portfolio Number'])}")
    print(f"Annual Return:    {color_text(portfolio_return, format_percent(portfolio_return))}")
    print(f"Annual Risk:      {format_percent(portfolio_risk)}")
    print(f"Sharpe Ratio:     {color_text(sharpe_ratio, f'{sharpe_ratio:.4f}')}")


def create_ranking_table(results: pd.DataFrame) -> pd.DataFrame:
    """Create a readable table with metrics and largest stock allocation."""
    weight_columns = get_weight_columns(results)

    ranking = results[
        [
            "Portfolio Number",
            "Annual Portfolio Return",
            "Annual Portfolio Risk",
            "Sharpe Ratio",
        ]
    ].copy()

    # Find the largest allocation in each portfolio.
    ranking["Largest Allocation"] = results[weight_columns].max(axis=1)

    # Find which stock has the largest allocation in each portfolio.
    largest_weight_columns = results[weight_columns].idxmax(axis=1)
    ranking["Largest Stock"] = largest_weight_columns.str.replace("Weight_", "", regex=False)

    return ranking


def print_top_10_tables(results: pd.DataFrame) -> None:
    """Print top 10 portfolios by Sharpe ratio and by lowest volatility."""
    ranking = create_ranking_table(results)

    top_sharpe = ranking.sort_values("Sharpe Ratio", ascending=False).head(10)
    top_low_risk = ranking.sort_values("Annual Portfolio Risk", ascending=True).head(10)

    print("\nTop 10 portfolios by highest Sharpe ratio:")
    print(format_ranking_for_display(top_sharpe).to_string(index=False))

    print("\nTop 10 portfolios by lowest volatility:")
    print(format_ranking_for_display(top_low_risk).to_string(index=False))


def format_ranking_for_display(ranking: pd.DataFrame) -> pd.DataFrame:
    """Format ranking table values as readable percentages."""
    display_table = ranking.copy()
    display_table["Annual Portfolio Return"] = display_table["Annual Portfolio Return"].map(format_percent)
    display_table["Annual Portfolio Risk"] = display_table["Annual Portfolio Risk"].map(format_percent)
    display_table["Largest Allocation"] = display_table["Largest Allocation"].map(format_percent)
    display_table["Sharpe Ratio"] = display_table["Sharpe Ratio"].map(lambda value: f"{value:.4f}")

    return display_table


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
    """Show charts only if the current backend supports interactive display."""
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def plot_monte_carlo_scatter(
    results: pd.DataFrame,
    max_sharpe_portfolio: pd.Series,
    min_volatility_portfolio: pd.Series,
) -> None:
    """Plot Monte Carlo risk-return scatter chart."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Positive return portfolios are pure green. Negative return portfolios are pure red.
    colors = np.where(results["Annual Portfolio Return"] >= 0, PURE_GREEN, PURE_RED)

    scatter = ax.scatter(
        results["Annual Portfolio Risk"] * 100,
        results["Annual Portfolio Return"] * 100,
        c=colors,
        alpha=0.25,
        s=18,
        edgecolors="none",
    )

    # Highlight the best Sharpe ratio random portfolio.
    ax.scatter(
        max_sharpe_portfolio["Annual Portfolio Risk"] * 100,
        max_sharpe_portfolio["Annual Portfolio Return"] * 100,
        marker="*",
        s=320,
        color=PURE_WHITE,
        edgecolors=PURE_GREEN,
        linewidths=1.5,
        label="Highest Sharpe Ratio",
    )

    # Highlight the lowest volatility random portfolio.
    ax.scatter(
        min_volatility_portfolio["Annual Portfolio Risk"] * 100,
        min_volatility_portfolio["Annual Portfolio Return"] * 100,
        marker="D",
        s=130,
        color=PURE_BLACK,
        edgecolors=PURE_WHITE,
        linewidths=1.5,
        label="Lowest Volatility",
    )

    ax.set_title("Monte Carlo Portfolio Simulation", fontsize=16)
    ax.set_xlabel("Annual Portfolio Risk / Volatility (%)")
    ax.set_ylabel("Annual Portfolio Return (%)")

    apply_black_chart_theme(ax)

    legend = ax.legend(facecolor=PURE_BLACK, edgecolor=PURE_WHITE)
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)

    # The variable is intentionally referenced so linters do not mark it unused.
    _ = scatter

    plt.tight_layout()
    plt.savefig(MONTE_CARLO_SCATTER_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)

    print(f"\nMonte Carlo scatter chart saved to: {MONTE_CARLO_SCATTER_FILE}")


def plot_best_portfolio_allocations(
    max_sharpe_weights: pd.DataFrame,
    min_volatility_weights: pd.DataFrame,
    top_n: int = 15,
) -> None:
    """Plot top allocations for max Sharpe and minimum volatility portfolios."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor(PURE_BLACK)

    chart_inputs = [
        (axes[0], max_sharpe_weights.head(top_n), "Highest Sharpe Portfolio Allocation"),
        (axes[1], min_volatility_weights.head(top_n), "Lowest Volatility Portfolio Allocation"),
    ]

    for ax, data, title in chart_inputs:
        plot_data = data.sort_values("Allocation (%)", ascending=True)

        ax.barh(
            plot_data["Stock Ticker"],
            plot_data["Allocation (%)"],
            color=PURE_GREEN,
            edgecolor=PURE_WHITE,
            linewidth=0.4,
        )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Allocation (%)")
        ax.set_ylabel("Stock Ticker")

        apply_black_chart_theme(ax)

    plt.tight_layout()
    plt.savefig(BEST_PORTFOLIO_ALLOCATION_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)

    print(f"Best portfolio allocation chart saved to: {BEST_PORTFOLIO_ALLOCATION_FILE}")


# -----------------------------
# 3. Main workflow
# -----------------------------

def main() -> None:
    """Run the Monte Carlo portfolio simulation."""
    create_output_folders()

    prices, daily_returns = load_project_data()

    tickers = list(daily_returns.columns)

    print("Monte Carlo portfolio simulation started.")
    print(f"Cleaned price dataset shape: {prices.shape}")
    print(f"Daily returns dataset shape: {daily_returns.shape}")
    print(f"Number of stocks used: {len(tickers)}")
    print(f"Number of random portfolios: {NUMBER_OF_PORTFOLIOS:,}")
    print(f"Risk-free rate assumption: {format_percent(RISK_FREE_RATE)}")

    annual_expected_returns, annual_covariance_matrix = calculate_annual_inputs(daily_returns)

    results = run_monte_carlo_simulation(
        tickers,
        annual_expected_returns,
        annual_covariance_matrix,
    )

    # Save all portfolio metrics and every portfolio weight.
    results.to_csv(SIMULATION_RESULTS_FILE, index=False)
    print(f"\nFull Monte Carlo results saved to: {SIMULATION_RESULTS_FILE}")

    # Find the portfolio with the highest Sharpe ratio.
    max_sharpe_index = results["Sharpe Ratio"].idxmax()
    max_sharpe_portfolio = results.loc[max_sharpe_index]

    # Find the portfolio with the lowest annual volatility.
    min_volatility_index = results["Annual Portfolio Risk"].idxmin()
    min_volatility_portfolio = results.loc[min_volatility_index]

    print_portfolio_result("Portfolio With Highest Sharpe Ratio", max_sharpe_portfolio)
    print_portfolio_result("Portfolio With Lowest Volatility", min_volatility_portfolio)

    print_top_10_tables(results)

    max_sharpe_weights = extract_weight_table(
        max_sharpe_portfolio,
        "Highest Sharpe Ratio Portfolio",
    )
    min_volatility_weights = extract_weight_table(
        min_volatility_portfolio,
        "Lowest Volatility Portfolio",
    )

    max_sharpe_weights.to_csv(MAX_SHARPE_WEIGHTS_FILE, index=False)
    min_volatility_weights.to_csv(MIN_VOLATILITY_WEIGHTS_FILE, index=False)

    print(f"\nHighest Sharpe portfolio weights saved to: {MAX_SHARPE_WEIGHTS_FILE}")
    print(f"Lowest volatility portfolio weights saved to: {MIN_VOLATILITY_WEIGHTS_FILE}")

    print("\nTop 10 allocations in the highest Sharpe portfolio:")
    print(format_allocation_for_display(max_sharpe_weights.head(10)).to_string(index=False))

    print("\nTop 10 allocations in the lowest volatility portfolio:")
    print(format_allocation_for_display(min_volatility_weights.head(10)).to_string(index=False))

    plot_monte_carlo_scatter(results, max_sharpe_portfolio, min_volatility_portfolio)
    plot_best_portfolio_allocations(max_sharpe_weights, min_volatility_weights)

    print("\nMonte Carlo simulation completed successfully.")


def format_allocation_for_display(weight_table: pd.DataFrame) -> pd.DataFrame:
    """Format allocation table for readable terminal output."""
    display_table = weight_table.copy()
    display_table["Weight"] = display_table["Weight"].map(lambda value: f"{value:.4f}")
    display_table["Allocation (%)"] = display_table["Allocation (%)"].map(lambda value: f"{value:.2f}%")

    return display_table


if __name__ == "__main__":
    main()
