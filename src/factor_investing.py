"""Price-based factor investing analysis for NSE stocks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import calculate_daily_returns


def zscore(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Return z-scores with optional inversion for lower-is-better factors."""
    clean = series.replace([np.inf, -np.inf], np.nan)
    std = clean.std()
    if std == 0 or np.isnan(std):
        score = pd.Series(0.0, index=series.index)
    else:
        score = (clean - clean.mean()) / std
    score = score.fillna(0.0)
    return score if higher_is_better else -score


def build_factor_score_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Create factor scores using momentum, volatility, and trend proxies."""
    returns = calculate_daily_returns(prices)
    lookback_6m = min(126, max(21, len(prices) // 3))
    lookback_12m = min(252, max(42, len(prices) // 2))

    momentum_6m = prices.iloc[-1] / prices.iloc[-lookback_6m] - 1
    momentum_12m = prices.iloc[-1] / prices.iloc[-lookback_12m] - 1
    volatility = returns.std() * np.sqrt(252)
    ma_window = min(50, max(10, len(prices) // 5))
    trend_ratio = prices.iloc[-1] / prices.rolling(ma_window).mean().iloc[-1] - 1

    table = pd.DataFrame(
        {
            "Ticker": prices.columns,
            "Momentum 6M": momentum_6m.values,
            "Momentum 12M": momentum_12m.values,
            "Annual Volatility": volatility.values,
            "Trend Ratio": trend_ratio.values,
        }
    ).set_index("Ticker")

    table["Momentum Score"] = zscore(table["Momentum 6M"]) + zscore(table["Momentum 12M"])
    table["Low Volatility Score"] = zscore(table["Annual Volatility"], higher_is_better=False)
    table["Trend Score"] = zscore(table["Trend Ratio"])
    table["Overall Factor Score"] = table[
        ["Momentum Score", "Low Volatility Score", "Trend Score"]
    ].mean(axis=1)
    return table.reset_index().sort_values("Overall Factor Score", ascending=False)


def build_factor_portfolio_weights(
    factor_scores: pd.DataFrame,
    portfolio_size: int = 10,
) -> pd.DataFrame:
    """Allocate to top factor-ranked stocks using positive shifted scores."""
    top = factor_scores.head(max(2, min(portfolio_size, len(factor_scores)))).copy()
    shifted_score = top["Overall Factor Score"] - top["Overall Factor Score"].min() + 0.01
    if shifted_score.sum() <= 0:
        top["Factor Weight"] = 1.0 / len(top)
    else:
        top["Factor Weight"] = shifted_score / shifted_score.sum()
    return top[["Ticker", "Overall Factor Score", "Factor Weight"]]
