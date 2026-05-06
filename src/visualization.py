"""Charting utilities with the NiveshMatrix black fintech theme."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CHART_COLORS
from src.risk import cumulative_returns, drawdown_series


def apply_chart_theme(ax: plt.Axes, title: str = "") -> None:
    """Apply pure black background and white labels to a Matplotlib axis."""
    fig = ax.figure
    fig.patch.set_facecolor(CHART_COLORS["background"])
    ax.set_facecolor(CHART_COLORS["background"])
    ax.set_title(title, color=CHART_COLORS["text"], fontsize=13, pad=12)
    ax.tick_params(colors=CHART_COLORS["text"])
    ax.xaxis.label.set_color(CHART_COLORS["text"])
    ax.yaxis.label.set_color(CHART_COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(CHART_COLORS["border"])
    ax.grid(True, color=CHART_COLORS["border"], alpha=0.30)


def style_legend(ax: plt.Axes) -> None:
    """Style a chart legend for the dark dashboard."""
    legend = ax.legend(facecolor=CHART_COLORS["background"], edgecolor=CHART_COLORS["border"])
    if legend:
        for text in legend.get_texts():
            text.set_color(CHART_COLORS["text"])


def price_trend_chart(prices: pd.DataFrame) -> plt.Figure:
    """Plot normalized price trend for selected stocks."""
    normalized = prices / prices.iloc[0]
    fig, ax = plt.subplots(figsize=(11, 5))
    normalized.plot(ax=ax, linewidth=1.6)
    apply_chart_theme(ax, "Normalized Stock Price Trends")
    ax.set_ylabel("Growth of 1.00")
    style_legend(ax)
    fig.tight_layout()
    return fig


def returns_distribution_chart(daily_returns: pd.DataFrame) -> plt.Figure:
    """Plot distribution of daily returns."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    daily_returns.stack().hist(ax=ax, bins=60, color=CHART_COLORS["profit"], alpha=0.70)
    apply_chart_theme(ax, "Distribution of Daily Stock Returns")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def correlation_heatmap(correlation_matrix: pd.DataFrame) -> plt.Figure:
    """Plot a correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        correlation_matrix,
        ax=ax,
        cmap="Greens",
        center=0,
        square=True,
        cbar_kws={"label": "Correlation"},
    )
    apply_chart_theme(ax, "Correlation Matrix")
    fig.tight_layout()
    return fig


def allocation_chart(allocation: pd.DataFrame, weight_column: str, title: str) -> plt.Figure:
    """Plot top allocation weights."""
    plot_data = allocation[allocation[weight_column] > 0.001].head(20)
    colors = [
        CHART_COLORS["profit"] if weight >= 0.01 else CHART_COLORS["text"]
        for weight in plot_data[weight_column]
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(plot_data["Ticker"], plot_data[weight_column], color=colors)
    apply_chart_theme(ax, title)
    ax.set_ylabel("Weight")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def efficient_frontier_chart(
    random_results: pd.DataFrame,
    comparison: pd.DataFrame,
) -> plt.Figure:
    """Plot Monte Carlo portfolios and selected strategy markers."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    scatter = ax.scatter(
        random_results["Annual Risk"],
        random_results["Expected Annual Return"],
        c=random_results["Sharpe Ratio"],
        cmap="Greens",
        alpha=0.55,
        s=15,
    )
    for _, row in comparison.iterrows():
        color = CHART_COLORS["profit"] if row["Sharpe Ratio"] >= 0 else CHART_COLORS["loss"]
        ax.scatter(row["Annual Risk"], row["Expected Annual Return"], s=110, color=color)
        ax.annotate(
            row["Strategy"],
            (row["Annual Risk"], row["Expected Annual Return"]),
            color=CHART_COLORS["text"],
            fontsize=9,
            xytext=(6, 6),
            textcoords="offset points",
        )
    apply_chart_theme(ax, "Monte Carlo Simulation and Efficient Frontier")
    ax.set_xlabel("Annual Risk")
    ax.set_ylabel("Expected Annual Return")
    color_bar = fig.colorbar(scatter, ax=ax)
    color_bar.set_label("Sharpe Ratio", color=CHART_COLORS["text"])
    color_bar.ax.yaxis.set_tick_params(color=CHART_COLORS["text"])
    plt.setp(color_bar.ax.get_yticklabels(), color=CHART_COLORS["text"])
    fig.tight_layout()
    return fig


def cumulative_return_chart(strategy_returns: pd.DataFrame, title: str) -> plt.Figure:
    """Plot cumulative return curves."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for column in strategy_returns.columns:
        cumulative_returns(strategy_returns[column]).plot(ax=ax, linewidth=1.8, label=column)
    apply_chart_theme(ax, title)
    ax.set_ylabel("Cumulative Return")
    style_legend(ax)
    fig.tight_layout()
    return fig


def drawdown_chart(strategy_returns: pd.DataFrame, title: str) -> plt.Figure:
    """Plot drawdown curves."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for column in strategy_returns.columns:
        drawdown_series(strategy_returns[column]).plot(ax=ax, linewidth=1.8, label=column)
    apply_chart_theme(ax, title)
    ax.set_ylabel("Drawdown")
    style_legend(ax)
    fig.tight_layout()
    return fig


def turnover_chart(turnover_table: pd.DataFrame) -> plt.Figure:
    """Plot monthly portfolio turnover."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if not turnover_table.empty:
        ax.plot(turnover_table["Date"], turnover_table["Turnover"], color=CHART_COLORS["profit"])
    apply_chart_theme(ax, "Monthly Rebalancing Turnover")
    ax.set_ylabel("Turnover")
    fig.tight_layout()
    return fig


def factor_score_chart(factor_scores: pd.DataFrame) -> plt.Figure:
    """Plot top factor scores."""
    top = factor_scores.head(15)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(top["Ticker"], top["Overall Factor Score"], color=CHART_COLORS["profit"])
    apply_chart_theme(ax, "Top Factor Scores")
    ax.set_ylabel("Overall Factor Score")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def ml_prediction_chart(predictions: pd.DataFrame) -> plt.Figure:
    """Plot latest ML-predicted 21-day returns."""
    top = predictions.head(15)
    colors = [
        CHART_COLORS["profit"] if value >= 0 else CHART_COLORS["loss"]
        for value in top["Predicted 21D Return"]
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(top["Ticker"], top["Predicted 21D Return"], color=colors)
    apply_chart_theme(ax, "Machine Learning Predicted 21D Returns")
    ax.set_ylabel("Predicted Return")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig
