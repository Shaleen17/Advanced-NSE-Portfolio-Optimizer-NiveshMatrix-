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
    """Plot the primary latest ML prediction column."""
    value_column = _primary_ml_prediction_column(predictions)
    top = predictions.head(15)
    colors = [
        CHART_COLORS["profit"] if value >= 0 else CHART_COLORS["loss"]
        for value in top[value_column]
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(top["Ticker"], top[value_column], color=colors)
    apply_chart_theme(ax, f"Machine Learning {value_column}")
    ax.set_ylabel(value_column)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def ml_model_performance_chart(fold_metrics: pd.DataFrame) -> plt.Figure:
    """Plot walk-forward model performance across folds."""
    metric_column = _primary_ml_performance_metric(fold_metrics)
    fig, ax = plt.subplots(figsize=(11, 5))

    for model_name, group in fold_metrics.dropna(subset=[metric_column]).groupby("Model"):
        group = group.sort_values("Fold")
        ax.plot(
            group["Fold"],
            group[metric_column],
            marker="o",
            linewidth=1.8,
            label=model_name,
        )

    apply_chart_theme(ax, f"Walk-Forward Model Comparison: {metric_column}")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_column)
    style_legend(ax)
    fig.tight_layout()
    return fig


def ml_feature_importance_chart(feature_importance: pd.DataFrame, limit: int = 12) -> plt.Figure:
    """Plot top ML feature importances or absolute coefficients."""
    top = feature_importance.head(limit).sort_values("Absolute Importance")
    fig, ax = plt.subplots(figsize=(11, 5))
    if not top.empty:
        ax.barh(top["Feature"], top["Absolute Importance"], color=CHART_COLORS["profit"])
    title = "Top ML Predictive Features"
    if not top.empty and "Explanation Type" in top.columns:
        title = str(top["Explanation Type"].iloc[0])
    apply_chart_theme(ax, title)
    ax.set_xlabel("Absolute Importance")
    fig.tight_layout()
    return fig


def _primary_ml_performance_metric(fold_metrics: pd.DataFrame) -> str:
    """Choose the metric column to compare model validation performance."""
    for column in [
        "RMSE",
        "ROC-AUC",
        "F1",
        "Spearman Correlation",
        "Top-3 Hit Rate",
    ]:
        if column in fold_metrics.columns and fold_metrics[column].notna().any():
            return column
    raise ValueError("No numeric ML validation metric is available for charting.")


def _primary_ml_prediction_column(predictions: pd.DataFrame) -> str:
    """Pick the best numeric prediction column for the ML bar chart."""
    probability_columns = [
        column for column in predictions.columns if column.startswith("Probability ")
    ]
    if probability_columns:
        return probability_columns[0]

    predicted_columns = [
        column
        for column in predictions.columns
        if column.startswith("Predicted ") and column != "Predicted Direction"
    ]
    if predicted_columns:
        return predicted_columns[0]

    numeric_columns = [
        column
        for column in predictions.select_dtypes(include="number").columns
        if column != "Ticker"
    ]
    if numeric_columns:
        return numeric_columns[0]

    raise ValueError("No numeric ML prediction column is available for charting.")


def live_top_movers_chart(live_quotes: pd.DataFrame, gainers: bool = True, limit: int = 8) -> plt.Figure:
    """Plot top live gainers or losers by Change %."""
    frame = live_quotes.dropna(subset=["Change %"]).copy()
    if gainers:
        frame = frame[frame["Change %"] > 0].sort_values("Change %", ascending=False).head(limit)
        title = "Top Live Gainers"
        color = CHART_COLORS["profit"]
    else:
        frame = frame[frame["Change %"] < 0].sort_values("Change %", ascending=True).head(limit)
        title = "Top Live Losers"
        color = CHART_COLORS["loss"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not frame.empty:
        ax.bar(frame["Ticker"], frame["Change %"], color=color)
    apply_chart_theme(ax, title)
    ax.set_ylabel("Change %")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def live_volume_leaders_chart(live_quotes: pd.DataFrame, limit: int = 10) -> plt.Figure:
    """Plot highest-volume live symbols."""
    frame = live_quotes.dropna(subset=["Volume"]).sort_values("Volume", ascending=False).head(limit)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not frame.empty:
        ax.bar(frame["Ticker"], frame["Volume"], color=CHART_COLORS["profit"])
    apply_chart_theme(ax, "Live Volume Leaders")
    ax.set_ylabel("Volume")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def live_change_distribution_chart(live_quotes: pd.DataFrame) -> plt.Figure:
    """Plot distribution of live Change % values."""
    returns = pd.to_numeric(live_quotes.get("Change %", pd.Series(dtype=float)), errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not returns.empty:
        ax.hist(returns, bins=min(20, max(5, len(returns))), color=CHART_COLORS["profit"], alpha=0.72)
        ax.axvline(0, color=CHART_COLORS["loss"], linewidth=1.5)
    apply_chart_theme(ax, "Live Change % Distribution")
    ax.set_xlabel("Change %")
    ax.set_ylabel("Symbols")
    fig.tight_layout()
    return fig


def live_breadth_chart(advancers: int, decliners: int, unchanged: int) -> plt.Figure:
    """Plot market breadth as a simple dark-theme bar chart."""
    labels = ["Advancers", "Decliners", "Unchanged"]
    values = [advancers, decliners, unchanged]
    colors = [CHART_COLORS["profit"], CHART_COLORS["loss"], CHART_COLORS["muted"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(labels, values, color=colors)
    apply_chart_theme(ax, "Advancers vs Decliners")
    ax.set_ylabel("Symbols")
    fig.tight_layout()
    return fig
