"""Machine learning helpers for experimental return prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from config import TRADING_DAYS

FEATURE_COLUMNS = [
    "Return 5D",
    "Return 20D",
    "Return 60D",
    "Volatility 20D",
    "MA 50 Ratio",
]


def build_feature_dataset(prices: pd.DataFrame, horizon_days: int = 21) -> pd.DataFrame:
    """Build a supervised learning dataset from historical prices."""
    rows: list[pd.DataFrame] = []
    for ticker in prices.columns:
        price = prices[ticker].dropna()
        returns = price.pct_change()
        frame = pd.DataFrame(index=price.index)
        frame["Ticker"] = ticker
        frame["Return 5D"] = price.pct_change(5)
        frame["Return 20D"] = price.pct_change(20)
        frame["Return 60D"] = price.pct_change(60)
        frame["Volatility 20D"] = returns.rolling(20).std()
        frame["MA 50 Ratio"] = price / price.rolling(50).mean() - 1
        frame["Target 21D Return"] = price.shift(-horizon_days) / price - 1
        rows.append(frame.dropna())

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows).reset_index(names="Date")


def train_return_model(dataset: pd.DataFrame) -> tuple[RandomForestRegressor, pd.DataFrame]:
    """Train a Random Forest model and return fold-level validation metrics."""
    if dataset.empty or len(dataset) < 250:
        raise ValueError("At least 250 ML rows are required for return prediction.")

    x_data = dataset[FEATURE_COLUMNS]
    y_data = dataset["Target 21D Return"]
    split_count = min(5, max(2, len(dataset) // 250))
    splitter = TimeSeriesSplit(n_splits=split_count)
    rows = []

    for fold, (train_index, test_index) in enumerate(splitter.split(x_data), start=1):
        model = RandomForestRegressor(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_data.iloc[train_index], y_data.iloc[train_index])
        prediction = model.predict(x_data.iloc[test_index])
        actual = y_data.iloc[test_index]
        rows.append(
            {
                "Fold": fold,
                "MAE": mean_absolute_error(actual, prediction),
                "RMSE": float(np.sqrt(mean_squared_error(actual, prediction))),
                "Directional Accuracy": float(
                    (np.sign(actual.values) == np.sign(prediction)).mean()
                ),
            }
        )

    final_model = RandomForestRegressor(
        n_estimators=220,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(x_data, y_data)
    return final_model, pd.DataFrame(rows)


def predict_latest_returns(
    prices: pd.DataFrame,
    model: RandomForestRegressor,
    horizon_days: int = 21,
) -> pd.DataFrame:
    """Predict latest horizon returns for each selected ticker."""
    dataset = build_feature_dataset(prices, horizon_days)
    if dataset.empty:
        raise ValueError("Could not build ML features from the selected data.")

    latest_rows = dataset.sort_values("Date").groupby("Ticker").tail(1).copy()
    latest_rows["Predicted 21D Return"] = model.predict(latest_rows[FEATURE_COLUMNS])
    latest_rows["Annualized Predicted Return"] = (
        1 + latest_rows["Predicted 21D Return"]
    ) ** (TRADING_DAYS / horizon_days) - 1
    return latest_rows[
        ["Ticker", "Predicted 21D Return", "Annualized Predicted Return"]
    ].sort_values("Predicted 21D Return", ascending=False)
