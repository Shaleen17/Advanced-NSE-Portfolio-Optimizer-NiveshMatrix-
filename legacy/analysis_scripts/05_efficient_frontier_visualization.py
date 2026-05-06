"""
Efficient Frontier visualization for the NSE portfolio optimization project.

This script uses the Monte Carlo portfolio simulation results from step 4.
It plots all 10,000 random portfolios on a risk-return chart, colors each
portfolio by Sharpe ratio, and highlights:

1. The maximum Sharpe ratio portfolio
2. The minimum volatility portfolio

Run this file after step 4:

    python scripts/04_monte_carlo_portfolio_simulation.py
    python scripts/05_efficient_frontier_visualization.py
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
import pandas as pd


# -----------------------------
# 1. Project settings
# -----------------------------

# Input file from the Monte Carlo simulation step.
MONTE_CARLO_RESULTS_FILE = Path("data/outputs/monte_carlo_portfolio_results.csv")

# Output chart file requested by the user.
EFFICIENT_FRONTIER_FILE = Path("reports/figures/efficient_frontier.png")

# Pure colors requested for the dashboard/chart style.
PURE_BLACK = "#000000"
PURE_WHITE = "#FFFFFF"
PURE_GREEN = "#00FF00"
PURE_RED = "#FF0000"


# -----------------------------
# 2. Helper functions
# -----------------------------

def load_monte_carlo_results() -> pd.DataFrame:
    """Load all simulated portfolios from the Monte Carlo output CSV."""
    if not MONTE_CARLO_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {MONTE_CARLO_RESULTS_FILE}\n"
            "Please run 04_monte_carlo_portfolio_simulation.py first."
        )

    results = pd.read_csv(MONTE_CARLO_RESULTS_FILE)

    required_columns = [
        "Portfolio Number",
        "Annual Portfolio Return",
        "Annual Portfolio Risk",
        "Sharpe Ratio",
    ]

    missing_columns = [column for column in required_columns if column not in results.columns]
    if missing_columns:
        raise ValueError(f"Monte Carlo file is missing required columns: {missing_columns}")

    return results


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


def format_percent(value: float) -> str:
    """Convert a decimal value into percentage text."""
    return f"{value * 100:.2f}%"


def print_selected_portfolio(title: str, portfolio: pd.Series) -> None:
    """Print a selected portfolio result clearly."""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Portfolio Number: {int(portfolio['Portfolio Number'])}")
    print(f"Expected Return:  {format_percent(portfolio['Annual Portfolio Return'])}")
    print(f"Risk/Volatility:  {format_percent(portfolio['Annual Portfolio Risk'])}")
    print(f"Sharpe Ratio:     {portfolio['Sharpe Ratio']:.4f}")


def save_or_show_chart(fig: plt.Figure) -> None:
    """Show chart only when the backend supports interactive display."""
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def plot_efficient_frontier(results: pd.DataFrame) -> None:
    """Create the Efficient Frontier style Monte Carlo scatter chart."""
    EFFICIENT_FRONTIER_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Find the maximum Sharpe ratio portfolio.
    max_sharpe_portfolio = results.loc[results["Sharpe Ratio"].idxmax()]

    # Find the minimum volatility portfolio.
    min_volatility_portfolio = results.loc[results["Annual Portfolio Risk"].idxmin()]

    print_selected_portfolio("Maximum Sharpe Ratio Portfolio", max_sharpe_portfolio)
    print_selected_portfolio("Minimum Volatility Portfolio", min_volatility_portfolio)

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(PURE_BLACK)

    # Plot all random portfolios.
    # X-axis = risk, Y-axis = expected return, color = Sharpe ratio.
    scatter = ax.scatter(
        results["Annual Portfolio Risk"] * 100,
        results["Annual Portfolio Return"] * 100,
        c=results["Sharpe Ratio"],
        cmap="viridis",
        s=20,
        alpha=0.65,
        edgecolors="none",
    )

    # Highlight the maximum Sharpe portfolio with a large pure green star.
    ax.scatter(
        max_sharpe_portfolio["Annual Portfolio Risk"] * 100,
        max_sharpe_portfolio["Annual Portfolio Return"] * 100,
        marker="*",
        s=650,
        color=PURE_GREEN,
        edgecolors=PURE_WHITE,
        linewidths=1.4,
        label="Maximum Sharpe Ratio",
        zorder=5,
    )

    # Highlight the minimum volatility portfolio with a large pure white diamond.
    ax.scatter(
        min_volatility_portfolio["Annual Portfolio Risk"] * 100,
        min_volatility_portfolio["Annual Portfolio Return"] * 100,
        marker="D",
        s=260,
        color=PURE_WHITE,
        edgecolors=PURE_GREEN,
        linewidths=1.4,
        label="Minimum Volatility",
        zorder=5,
    )

    # Add readable annotation text near the highlighted points.
    ax.annotate(
        f"Max Sharpe\nReturn {format_percent(max_sharpe_portfolio['Annual Portfolio Return'])}\n"
        f"Risk {format_percent(max_sharpe_portfolio['Annual Portfolio Risk'])}\n"
        f"Sharpe {max_sharpe_portfolio['Sharpe Ratio']:.2f}",
        xy=(
            max_sharpe_portfolio["Annual Portfolio Risk"] * 100,
            max_sharpe_portfolio["Annual Portfolio Return"] * 100,
        ),
        xytext=(18, 18),
        textcoords="offset points",
        color=PURE_GREEN,
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": PURE_BLACK, "edgecolor": PURE_GREEN},
    )

    ax.annotate(
        f"Min Volatility\nReturn {format_percent(min_volatility_portfolio['Annual Portfolio Return'])}\n"
        f"Risk {format_percent(min_volatility_portfolio['Annual Portfolio Risk'])}\n"
        f"Sharpe {min_volatility_portfolio['Sharpe Ratio']:.2f}",
        xy=(
            min_volatility_portfolio["Annual Portfolio Risk"] * 100,
            min_volatility_portfolio["Annual Portfolio Return"] * 100,
        ),
        xytext=(18, -72),
        textcoords="offset points",
        color=PURE_WHITE,
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": PURE_BLACK, "edgecolor": PURE_WHITE},
    )

    ax.set_title("Efficient Frontier Using Monte Carlo Portfolio Simulation", fontsize=18, pad=18)
    ax.set_xlabel("Portfolio Risk / Annual Volatility (%)", fontsize=12)
    ax.set_ylabel("Portfolio Expected Annual Return (%)", fontsize=12)

    apply_black_chart_theme(ax)

    # Add a colorbar to show that portfolio color represents Sharpe ratio.
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Sharpe Ratio", color=PURE_WHITE)
    colorbar.ax.yaxis.label.set_color(PURE_WHITE)
    colorbar.ax.tick_params(colors=PURE_WHITE)
    colorbar.outline.set_edgecolor(PURE_WHITE)

    # Add a legend for the highlighted portfolios.
    legend = ax.legend(
        facecolor=PURE_BLACK,
        edgecolor=PURE_WHITE,
        loc="lower right",
        framealpha=1.0,
    )
    for text in legend.get_texts():
        text.set_color(PURE_WHITE)

    # Add a small explanatory note directly on the chart.
    ax.text(
        0.02,
        0.97,
        "Better portfolios usually sit higher and further left: higher return, lower risk.",
        color=PURE_WHITE,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
    )

    plt.tight_layout()
    plt.savefig(EFFICIENT_FRONTIER_FILE, dpi=300, facecolor=PURE_BLACK)
    save_or_show_chart(fig)

    print(f"\nEfficient Frontier chart saved to: {EFFICIENT_FRONTIER_FILE}")


# -----------------------------
# 3. Main workflow
# -----------------------------

def main() -> None:
    """Run Efficient Frontier visualization."""
    results = load_monte_carlo_results()

    print("Monte Carlo portfolio results loaded successfully.")
    print(f"Total portfolios plotted: {len(results):,}")

    plot_efficient_frontier(results)

    print("\nEfficient Frontier visualization completed successfully.")


if __name__ == "__main__":
    main()
