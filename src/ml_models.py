"""Leak-safe machine learning pipeline for experimental return prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:  # pragma: no cover - depends on sklearn version.
    HistGradientBoostingRegressor = None
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import TRADING_DAYS

DATE_COLUMN = "Date"
TICKER_COLUMN = "Ticker"
DEFAULT_HORIZON_DAYS = 21
PREDICTION_HORIZONS = (5, 21, 63)
TARGET_COLUMN = "Target 21D Return"
PREDICTION_COLUMN = "Predicted 21D Return"
ANNUALIZED_PREDICTION_COLUMN = "Annualized Predicted Return"

TARGET_TYPE_REGRESSION = "regression"
TARGET_TYPE_CLASSIFICATION = "classification"
TARGET_TYPE_RISK_ADJUSTED = "risk_adjusted"
TARGET_TYPE_RANKING = "ranking"
TARGET_TYPE_LABELS = {
    TARGET_TYPE_REGRESSION: "Future return regression",
    TARGET_TYPE_CLASSIFICATION: "Direction classification",
    TARGET_TYPE_RISK_ADJUSTED: "Risk-adjusted return",
    TARGET_TYPE_RANKING: "Cross-sectional ranking",
}
TARGET_TYPE_OPTIONS = tuple(TARGET_TYPE_LABELS.keys())

MLModel = BaseEstimator

ML_WEIGHT_SOFTMAX = "softmax"
ML_WEIGHT_RANK_TOP_K = "rank_top_k"
ML_WEIGHT_POSITIVE_ONLY = "positive_signal_only"
ML_WEIGHT_VOLATILITY_ADJUSTED = "volatility_adjusted"
ML_WEIGHT_METHOD_LABELS = {
    ML_WEIGHT_SOFTMAX: "Softmax predicted return weights",
    ML_WEIGHT_RANK_TOP_K: "Rank-based top-k weights",
    ML_WEIGHT_POSITIVE_ONLY: "Positive-signal-only weights",
    ML_WEIGHT_VOLATILITY_ADJUSTED: "Volatility-adjusted predicted return weights",
}

RETURN_WINDOWS = (1, 5, 10, 20, 60, 120)
VOLATILITY_WINDOWS = (10, 20, 60)
SMA_WINDOWS = (20, 50, 200)
MOMENTUM_WINDOWS = (20, 60, 120)
BOLLINGER_WINDOW = 20
BOLLINGER_STD_MULTIPLIER = 2.0
RSI_WINDOW = 14
MACD_FAST_SPAN = 12
MACD_SLOW_SPAN = 26
ROLLING_DRAWDOWN_WINDOW = 60
DOWNSIDE_VOLATILITY_WINDOW = 20

RISK_FEATURE_COLUMNS = (
    "Rolling Max Drawdown 60D",
    "Downside Volatility 20D",
)

TECHNICAL_FEATURE_COLUMNS = (
    "RSI 14",
    "MACD",
    "Bollinger Band Position",
)

CROSS_SECTIONAL_FEATURE_COLUMNS = (
    "Ticker Return Rank",
    "Volatility Rank",
    "Momentum Rank",
)

LEGACY_FEATURE_COLUMNS = (
    "Return 5D",
    "Return 20D",
    "Return 60D",
    "Volatility 20D",
    "MA 50 Ratio",
)


def _return_feature_name(window: int) -> str:
    """Return the feature name for a trailing point-to-point return."""
    return f"Return {window}D"


def _volatility_feature_name(window: int) -> str:
    """Return the feature name for trailing rolling volatility."""
    return f"Volatility {window}D"


def _sma_ratio_feature_name(window: int) -> str:
    """Return the feature name for price relative to a simple moving average."""
    return f"Price / SMA{window} - 1"


def _momentum_feature_name(window: int) -> str:
    """Return the feature name for trailing average daily momentum."""
    return f"Momentum {window}D"


def _build_feature_columns() -> tuple[str, ...]:
    """Build the model feature list from grouped feature definitions."""
    return (
        *(_return_feature_name(window) for window in RETURN_WINDOWS),
        *(_volatility_feature_name(window) for window in VOLATILITY_WINDOWS),
        *(_sma_ratio_feature_name(window) for window in SMA_WINDOWS),
        "SMA20 / SMA50 - 1",
        *(_momentum_feature_name(window) for window in MOMENTUM_WINDOWS),
        *RISK_FEATURE_COLUMNS,
        *TECHNICAL_FEATURE_COLUMNS,
        *CROSS_SECTIONAL_FEATURE_COLUMNS,
    )


# Public constant remains available, but its content is now generated.
FEATURE_COLUMNS = list(_build_feature_columns())
DEFAULT_MODEL_FEATURE_COLUMNS = tuple(FEATURE_COLUMNS)

MODEL_RIDGE = "Ridge Regression"
MODEL_LASSO = "Lasso Regression"
MODEL_RANDOM_FOREST_REGRESSOR = "RandomForestRegressor"
MODEL_EXTRA_TREES_REGRESSOR = "ExtraTreesRegressor"
MODEL_GRADIENT_BOOSTING_REGRESSOR = "GradientBoostingRegressor"
MODEL_HIST_GRADIENT_BOOSTING_REGRESSOR = "HistGradientBoostingRegressor"
MODEL_LOGISTIC_REGRESSION = "LogisticRegression"
MODEL_RANDOM_FOREST_CLASSIFIER = "RandomForestClassifier"

REGRESSION_MODEL_NAMES = (
    MODEL_RIDGE,
    MODEL_LASSO,
    MODEL_RANDOM_FOREST_REGRESSOR,
    MODEL_EXTRA_TREES_REGRESSOR,
    MODEL_GRADIENT_BOOSTING_REGRESSOR,
    *(
        (MODEL_HIST_GRADIENT_BOOSTING_REGRESSOR,)
        if HistGradientBoostingRegressor is not None
        else tuple()
    ),
)
CLASSIFICATION_MODEL_NAMES = (
    MODEL_LOGISTIC_REGRESSION,
    MODEL_RANDOM_FOREST_CLASSIFIER,
)

MODEL_PARAMETERS: dict[str, dict[str, object]] = {
    MODEL_RIDGE: {"alpha": 1.0},
    MODEL_LASSO: {"alpha": 0.01, "max_iter": 4000, "tol": 0.01},
    MODEL_RANDOM_FOREST_REGRESSOR: {
        "n_estimators": "configurable",
        "max_depth": "configurable",
        "min_samples_leaf": "configurable",
        "random_state": "configurable",
        "n_jobs": "configurable",
    },
    MODEL_EXTRA_TREES_REGRESSOR: {
        "n_estimators": "configurable",
        "max_depth": "configurable",
        "min_samples_leaf": "configurable",
        "random_state": "configurable",
        "n_jobs": "configurable",
    },
    MODEL_GRADIENT_BOOSTING_REGRESSOR: {
        "n_estimators": "configurable",
        "learning_rate": 0.04,
        "max_depth": 3,
        "min_samples_leaf": "configurable",
        "random_state": "configurable",
    },
    MODEL_HIST_GRADIENT_BOOSTING_REGRESSOR: {
        "max_iter": "configurable",
        "learning_rate": 0.04,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.01,
        "random_state": "configurable",
    },
    MODEL_LOGISTIC_REGRESSION: {
        "C": 1.0,
        "max_iter": 700,
        "class_weight": "balanced",
        "random_state": "configurable",
    },
    MODEL_RANDOM_FOREST_CLASSIFIER: {
        "n_estimators": "configurable",
        "max_depth": "configurable",
        "min_samples_leaf": "configurable",
        "class_weight": "balanced",
        "random_state": "configurable",
        "n_jobs": "configurable",
    },
}


@dataclass(frozen=True)
class MLTrainingConfig:
    """Configuration shared by the ML model zoo."""

    validation_estimators: int = 70
    final_estimators: int = 110
    max_depth: int | None = 6
    min_samples_leaf: int = 5
    random_state: int = 42
    n_jobs: int = 1
    min_training_rows: int = 250
    min_splits: int = 2
    max_splits: int = 5


@dataclass(frozen=True)
class ValidationFold:
    """Date-based validation fold with a purged training window."""

    fold: int
    train_index: np.ndarray
    validation_index: np.ndarray
    train_start: object
    train_end: object
    validation_start: object
    validation_end: object


@dataclass(frozen=True)
class MLTrainingResult:
    """Container for model-zoo training outputs."""

    best_model: MLModel
    best_model_name: str
    fold_metrics: pd.DataFrame
    model_comparison: pd.DataFrame


@dataclass
class ReturnPredictionPipeline:
    """Feature, target, training, evaluation, and prediction pipeline.

    The pipeline uses only trailing price information for features, creates
    future-return targets separately, and validates with expanding date folds.
    A purge gap equal to the prediction horizon keeps training labels from
    overlapping each validation window.
    """

    horizon_days: int = DEFAULT_HORIZON_DAYS
    feature_columns: tuple[str, ...] = DEFAULT_MODEL_FEATURE_COLUMNS
    target_column_name: str | None = None
    target_type: str = TARGET_TYPE_REGRESSION
    config: MLTrainingConfig = field(default_factory=MLTrainingConfig)

    def __post_init__(self) -> None:
        """Validate basic pipeline settings early."""
        self.target_type = normalize_target_type(self.target_type)
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be a positive integer.")
        if not self.feature_columns:
            raise ValueError("At least one feature column is required.")

    @property
    def target_column(self) -> str:
        """Return the active target column name."""
        return self.target_column_name or target_column_for_horizon(
            self.horizon_days,
            self.target_type,
        )

    @property
    def prediction_column(self) -> str:
        """Return the active prediction column name."""
        return prediction_column_for_horizon(self.horizon_days, self.target_type)

    def generate_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Create trailing, point-in-time features for every ticker/date.

        The returned DataFrame does not contain future targets, so it is safe
        to use for latest predictions.
        """
        clean_prices = self._validate_prices(prices)
        frames: list[pd.DataFrame] = []

        for ticker in clean_prices.columns:
            price = clean_prices[ticker].dropna()
            if price.empty:
                continue

            returns = price.pct_change(fill_method=None)
            frame = pd.DataFrame(index=price.index)
            frame[TICKER_COLUMN] = str(ticker)
            for window in RETURN_WINDOWS:
                frame[_return_feature_name(window)] = price.pct_change(
                    window, fill_method=None
                )

            for window in VOLATILITY_WINDOWS:
                frame[_volatility_feature_name(window)] = returns.rolling(window).std()

            sma_values: dict[int, pd.Series] = {}
            for window in SMA_WINDOWS:
                sma_values[window] = price.rolling(window).mean()
                frame[_sma_ratio_feature_name(window)] = (
                    _safe_divide(price, sma_values[window]) - 1
                )

            frame["SMA20 / SMA50 - 1"] = (
                _safe_divide(sma_values[20], sma_values[50]) - 1
            )
            # Legacy aliases keep older fitted RandomForest instances usable.
            frame["MA 20 Ratio"] = frame["Price / SMA20 - 1"]
            frame["MA 50 Ratio"] = frame["Price / SMA50 - 1"]

            for window in MOMENTUM_WINDOWS:
                frame[_momentum_feature_name(window)] = returns.rolling(window).mean()

            frame["Rolling Max Drawdown 60D"] = _rolling_max_drawdown(
                price, ROLLING_DRAWDOWN_WINDOW
            )
            frame["Drawdown 60D"] = price / price.rolling(60).max() - 1
            frame["Downside Volatility 20D"] = (
                returns.where(returns < 0, 0.0)
                .rolling(DOWNSIDE_VOLATILITY_WINDOW)
                .std()
            )
            frame["RSI 14"] = _relative_strength_index(price, window=RSI_WINDOW)
            frame["MACD"] = _moving_average_convergence_divergence(price)
            frame["Bollinger Band Position"] = _bollinger_band_position(
                price,
                window=BOLLINGER_WINDOW,
                std_multiplier=BOLLINGER_STD_MULTIPLIER,
            )
            frames.append(frame)

        if not frames:
            return _empty_feature_frame(self.feature_columns)

        features = pd.concat(frames)
        features = features.reset_index(names=DATE_COLUMN)
        features = _add_cross_sectional_features(features)
        return self._clean_feature_frame(features)

    def generate_targets(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Create future-return targets without mixing them into features."""
        clean_prices = self._validate_prices(prices)
        frames: list[pd.DataFrame] = []

        for ticker in clean_prices.columns:
            price = clean_prices[ticker].dropna()
            if price.empty:
                continue

            future_return = price.shift(-self.horizon_days) / price - 1
            frame = pd.DataFrame(index=price.index)
            frame[TICKER_COLUMN] = str(ticker)
            if self.target_type == TARGET_TYPE_CLASSIFICATION:
                frame[self.target_column] = np.where(
                    future_return.notna(),
                    (future_return > 0).astype(int),
                    np.nan,
                )
            elif self.target_type == TARGET_TYPE_RISK_ADJUSTED:
                recent_volatility = price.pct_change(fill_method=None).rolling(20).std()
                frame[self.target_column] = _safe_divide(
                    future_return,
                    recent_volatility,
                )
            elif self.target_type == TARGET_TYPE_RANKING:
                frame["Future Return"] = future_return
            else:
                frame[self.target_column] = future_return
            frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=[DATE_COLUMN, TICKER_COLUMN, self.target_column])

        targets = pd.concat(frames).reset_index(names=DATE_COLUMN)
        if self.target_type == TARGET_TYPE_RANKING:
            targets[self.target_column] = targets.groupby(DATE_COLUMN)[
                "Future Return"
            ].rank(method="average", pct=True)
            targets = targets.drop(columns=["Future Return"])

        targets[self.target_column] = pd.to_numeric(
            targets[self.target_column], errors="coerce"
        )
        targets = targets.replace([np.inf, -np.inf], np.nan)
        targets = targets.dropna(subset=[self.target_column])
        return targets.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)

    def build_dataset(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build a supervised training dataset from prices."""
        features = self.generate_features(prices)
        targets = self.generate_targets(prices)

        if features.empty or targets.empty:
            return pd.DataFrame(
                columns=[
                    DATE_COLUMN,
                    TICKER_COLUMN,
                    *self.feature_columns,
                    self.target_column,
                ]
            )

        dataset = features.merge(
            targets,
            on=[DATE_COLUMN, TICKER_COLUMN],
            how="inner",
            validate="one_to_one",
        )
        return self._prepare_training_dataset(dataset, enforce_min_rows=False)

    def evaluate_model(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the model zoo with purged, date-based walk-forward folds."""
        prepared = self._prepare_training_dataset(dataset)
        return self._evaluate_prepared_dataset(prepared)

    def train_model_zoo(self, dataset: pd.DataFrame) -> MLTrainingResult:
        """Train all suitable models and refit the best model on all rows."""
        prepared = self._prepare_training_dataset(dataset)
        fold_metrics = self._evaluate_prepared_dataset(prepared)
        model_comparison = self._build_model_comparison(fold_metrics)
        best_model_name = str(model_comparison.iloc[0]["Model"])

        best_model = self._new_model(best_model_name, final_model=True)
        y_data = prepared[self.target_column]
        if self.target_type == TARGET_TYPE_CLASSIFICATION:
            y_data = y_data.astype(int)
        best_model.fit(prepared[list(self.feature_columns)], y_data)
        _attach_model_metadata(
            best_model,
            horizon_days=self.horizon_days,
            target_type=self.target_type,
            feature_columns=self.feature_columns,
            model_name=best_model_name,
        )
        return MLTrainingResult(
            best_model=best_model,
            best_model_name=best_model_name,
            fold_metrics=fold_metrics,
            model_comparison=model_comparison,
        )

    def train_model(
        self, dataset: pd.DataFrame
    ) -> tuple[MLModel, pd.DataFrame]:
        """Train the model zoo and return the selected model plus fold metrics."""
        result = self.train_model_zoo(dataset)
        return result.best_model, result.fold_metrics

    def predict_latest(
        self,
        prices: pd.DataFrame,
        model: MLModel,
    ) -> pd.DataFrame:
        """Predict the latest available horizon return for each ticker."""
        if not hasattr(model, "predict"):
            raise TypeError("model must provide a predict method.")

        features = self.generate_features(prices)
        if features.empty:
            raise ValueError("Could not build ML features from the selected data.")

        latest_rows = (
            features.sort_values([DATE_COLUMN, TICKER_COLUMN])
            .groupby(TICKER_COLUMN, group_keys=False)
            .tail(1)
            .copy()
        )
        model_features = _feature_columns_for_model(model, latest_rows, self.feature_columns)
        x_latest = latest_rows[model_features]

        if self.target_type == TARGET_TYPE_CLASSIFICATION:
            latest_rows[self.prediction_column] = model.predict(x_latest).astype(int)
            probability_column = probability_column_for_horizon(self.horizon_days)
            latest_rows[probability_column] = _positive_class_probability(model, x_latest)
            columns = [TICKER_COLUMN, self.prediction_column, probability_column]
            sort_column = probability_column
        else:
            latest_rows[self.prediction_column] = model.predict(x_latest)
            if self.target_type == TARGET_TYPE_REGRESSION:
                latest_rows[ANNUALIZED_PREDICTION_COLUMN] = (
                    (1 + latest_rows[self.prediction_column])
                    ** (TRADING_DAYS / self.horizon_days)
                    - 1
                )
                columns = [
                    TICKER_COLUMN,
                    self.prediction_column,
                    ANNUALIZED_PREDICTION_COLUMN,
                ]
            else:
                columns = [TICKER_COLUMN, self.prediction_column]
            sort_column = self.prediction_column

        if (
            self.target_type == TARGET_TYPE_REGRESSION
            and self.prediction_column != PREDICTION_COLUMN
        ):
            latest_rows[PREDICTION_COLUMN] = latest_rows[self.prediction_column]

        predictions = latest_rows[columns].sort_values(sort_column, ascending=False)
        return predictions.reset_index(drop=True)

    def _validate_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the raw price table."""
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame.")
        if prices.empty:
            raise ValueError("prices cannot be empty.")
        if isinstance(prices.columns, pd.MultiIndex):
            raise ValueError("prices must have one flat column per ticker.")

        clean_prices = prices.copy()
        clean_prices = clean_prices.apply(pd.to_numeric, errors="coerce")
        clean_prices = clean_prices.dropna(axis=1, how="all")
        if clean_prices.empty:
            raise ValueError("prices must contain at least one numeric ticker series.")

        clean_prices = clean_prices.sort_index()
        clean_prices = clean_prices[~clean_prices.index.duplicated(keep="last")]
        return clean_prices

    def _clean_feature_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Drop incomplete or non-finite feature rows."""
        missing = [column for column in self.feature_columns if column not in features.columns]
        if missing:
            raise ValueError(f"Missing generated feature columns: {', '.join(missing)}")

        clean_features = features.copy()
        for column in self.feature_columns:
            clean_features[column] = pd.to_numeric(clean_features[column], errors="coerce")
        clean_features = clean_features.replace([np.inf, -np.inf], np.nan)
        clean_features = clean_features.dropna(subset=list(self.feature_columns))
        return clean_features.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)

    def _prepare_training_dataset(
        self,
        dataset: pd.DataFrame,
        *,
        enforce_min_rows: bool = True,
    ) -> pd.DataFrame:
        """Validate, clean, and sort a supervised dataset."""
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame.")
        if dataset.empty:
            if enforce_min_rows:
                raise ValueError("At least 250 ML rows are required for return prediction.")
            return dataset.copy()

        required_columns = {
            DATE_COLUMN,
            TICKER_COLUMN,
            self.target_column,
            *self.feature_columns,
        }
        missing = sorted(required_columns.difference(dataset.columns))
        if missing:
            raise ValueError(f"ML dataset is missing required columns: {', '.join(missing)}")

        prepared = dataset.copy()
        parsed_dates = pd.to_datetime(prepared[DATE_COLUMN], errors="coerce")
        if parsed_dates.notna().all():
            prepared[DATE_COLUMN] = parsed_dates

        numeric_columns = [*self.feature_columns, self.target_column]
        for column in numeric_columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared = prepared.replace([np.inf, -np.inf], np.nan)
        prepared = prepared.dropna(subset=numeric_columns + [DATE_COLUMN, TICKER_COLUMN])
        if self.target_type == TARGET_TYPE_CLASSIFICATION:
            prepared[self.target_column] = prepared[self.target_column].astype(int)
        prepared = prepared.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)

        if enforce_min_rows and len(prepared) < self.config.min_training_rows:
            raise ValueError(
                f"At least {self.config.min_training_rows} clean ML rows are required "
                "for return prediction."
            )
        return prepared

    def _evaluate_prepared_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Run walk-forward validation for every suitable model."""
        folds = self._build_validation_folds(dataset)
        model_names = model_names_for_target(self.target_type)
        metric_rows: list[dict[str, object]] = []
        for model_name in model_names:
            metric_rows.extend(
                self._evaluate_model_on_folds(dataset, folds, model_name)
            )
        return pd.DataFrame(metric_rows)

    def _evaluate_model_on_folds(
        self,
        dataset: pd.DataFrame,
        folds: list[ValidationFold],
        model_name: str,
    ) -> list[dict[str, object]]:
        """Evaluate one model across the prepared walk-forward folds."""
        x_data = dataset[list(self.feature_columns)]
        y_data = dataset[self.target_column]
        rows: list[dict[str, object]] = []

        for fold in folds:
            model = self._new_model(model_name, final_model=False)
            x_train = x_data.iloc[fold.train_index]
            x_validation = x_data.iloc[fold.validation_index]
            y_train = y_data.iloc[fold.train_index]
            if self.target_type == TARGET_TYPE_CLASSIFICATION:
                y_train = y_train.astype(int)

            actual = y_data.iloc[fold.validation_index]
            row = self._base_metric_row(fold, model_name)
            try:
                model.fit(x_train, y_train)
                prediction = model.predict(x_validation)
            except Exception as error:
                row["Status"] = "Failed"
                row["Error"] = str(error)
                row.update(_empty_metric_values(self.target_type))
                rows.append(row)
                continue

            row["Status"] = "OK"
            row["Error"] = ""

            if self.target_type == TARGET_TYPE_CLASSIFICATION:
                actual_classes = actual.astype(int)
                predicted_classes = pd.Series(prediction, index=actual.index).astype(int)
                probability = _positive_class_probability(model, x_validation)
                row.update(
                    {
                        "Accuracy": accuracy_score(actual_classes, predicted_classes),
                        "Precision": precision_score(
                            actual_classes,
                            predicted_classes,
                            zero_division=0,
                        ),
                        "Recall": recall_score(
                            actual_classes,
                            predicted_classes,
                            zero_division=0,
                        ),
                        "F1": f1_score(
                            actual_classes,
                            predicted_classes,
                            zero_division=0,
                        ),
                        "ROC-AUC": _safe_roc_auc(actual_classes, probability),
                    }
                )
            elif self.target_type == TARGET_TYPE_RANKING:
                validation_frame = dataset.iloc[fold.validation_index][
                    [DATE_COLUMN, TICKER_COLUMN, self.target_column]
                ].copy()
                validation_frame["Prediction"] = prediction
                spearman, top_k_hit_rate = _ranking_validation_metrics(
                    validation_frame,
                    actual_column=self.target_column,
                    prediction_column="Prediction",
                )
                row.update(
                    {
                        "Spearman Correlation": spearman,
                        "Top-3 Hit Rate": top_k_hit_rate,
                    }
                )
            else:
                row.update(
                    {
                        "MAE": mean_absolute_error(actual, prediction),
                        "RMSE": float(np.sqrt(mean_squared_error(actual, prediction))),
                        "R2": r2_score(actual, prediction),
                    }
                )

            rows.append(row)

        return rows

    def _build_model_comparison(self, fold_metrics: pd.DataFrame) -> pd.DataFrame:
        """Aggregate fold metrics and rank models for final selection."""
        if fold_metrics.empty:
            raise ValueError("No model validation metrics were produced.")

        metric_columns = _metric_columns_for_target(self.target_type)
        rows: list[dict[str, object]] = []
        for model_name, group in fold_metrics.groupby("Model", sort=False):
            row: dict[str, object] = {
                "Model": model_name,
                "Validation Folds": int(group["Fold"].nunique()),
            }
            for metric in metric_columns:
                if metric in group.columns:
                    row[f"Avg {metric}"] = pd.to_numeric(
                        group[metric], errors="coerce"
                    ).mean()
            error_count = int((group.get("Status") == "Failed").sum())
            row["Failed Folds"] = error_count
            rows.append(row)

        comparison = pd.DataFrame(rows)
        score_column, higher_is_better = _selection_metric_for_target(
            self.target_type,
            comparison,
        )
        comparison["Selection Metric"] = score_column.replace("Avg ", "")
        comparison["Selection Score"] = pd.to_numeric(
            comparison[score_column], errors="coerce"
        )
        comparison = comparison.sort_values(
            "Selection Score",
            ascending=not higher_is_better,
            na_position="last",
        ).reset_index(drop=True)

        if comparison.empty or pd.isna(comparison.iloc[0]["Selection Score"]):
            raise ValueError("All candidate ML models failed validation.")

        comparison["Selected"] = np.where(comparison.index == 0, "Yes", "No")
        return comparison

    def _base_metric_row(
        self,
        fold: ValidationFold,
        model_name: str,
    ) -> dict[str, object]:
        """Return shared validation metric metadata for one fold."""
        return {
            "Fold": fold.fold,
            "Model": model_name,
            "Target Type": TARGET_TYPE_LABELS[self.target_type],
            "Horizon Days": self.horizon_days,
            "Train Start": fold.train_start,
            "Train End": fold.train_end,
            "Validation Start": fold.validation_start,
            "Validation End": fold.validation_end,
            "Train Rows": int(len(fold.train_index)),
            "Validation Rows": int(len(fold.validation_index)),
        }

    def _build_validation_folds(self, dataset: pd.DataFrame) -> list[ValidationFold]:
        """Build expanding validation folds split by date with a purge gap."""
        unique_dates = pd.Index(dataset[DATE_COLUMN].drop_duplicates()).sort_values()
        if len(unique_dates) <= self.horizon_days + self.config.min_splits:
            raise ValueError(
                "Not enough unique dates for purged time-series validation. "
                "Use a longer price history."
            )

        requested_splits = min(
            self.config.max_splits,
            max(self.config.min_splits, len(unique_dates) // 126),
        )

        for split_count in range(requested_splits, self.config.min_splits - 1, -1):
            folds = self._folds_for_split_count(dataset, unique_dates, split_count)
            if len(folds) >= self.config.min_splits:
                return folds

        raise ValueError(
            "Not enough clean rows for purged time-series validation. "
            "Use a longer price history or fewer missing prices."
        )

    def _folds_for_split_count(
        self,
        dataset: pd.DataFrame,
        unique_dates: pd.Index,
        split_count: int,
    ) -> list[ValidationFold]:
        """Create folds for one requested split count."""
        validation_window = max(1, len(unique_dates) // (split_count + 1))
        folds: list[ValidationFold] = []

        for fold_number in range(1, split_count + 1):
            validation_start_position = len(unique_dates) - validation_window * (
                split_count - fold_number + 1
            )
            validation_end_position = len(unique_dates) - validation_window * (
                split_count - fold_number
            )
            if fold_number == split_count:
                validation_end_position = len(unique_dates)

            train_end_position = validation_start_position - self.horizon_days
            if train_end_position <= 0:
                continue

            train_dates = unique_dates[:train_end_position]
            validation_dates = unique_dates[
                validation_start_position:validation_end_position
            ]
            train_mask = dataset[DATE_COLUMN].isin(train_dates).to_numpy()
            validation_mask = dataset[DATE_COLUMN].isin(validation_dates).to_numpy()
            train_index = np.flatnonzero(train_mask)
            validation_index = np.flatnonzero(validation_mask)

            if len(train_index) == 0 or len(validation_index) == 0:
                continue

            folds.append(
                ValidationFold(
                    fold=len(folds) + 1,
                    train_index=train_index,
                    validation_index=validation_index,
                    train_start=train_dates[0],
                    train_end=train_dates[-1],
                    validation_start=validation_dates[0],
                    validation_end=validation_dates[-1],
                )
            )

        return folds

    def _new_model(self, model_name: str, *, final_model: bool) -> MLModel:
        """Create one configured model from the model zoo."""
        return create_model(model_name, self.config, final_model=final_model)


def model_names_for_target(target_type: str) -> tuple[str, ...]:
    """Return model names that are suitable for the selected target type."""
    resolved_target_type = normalize_target_type(target_type)
    if resolved_target_type == TARGET_TYPE_CLASSIFICATION:
        return CLASSIFICATION_MODEL_NAMES
    return REGRESSION_MODEL_NAMES


def create_model(
    model_name: str,
    config: MLTrainingConfig | None = None,
    *,
    final_model: bool = False,
) -> MLModel:
    """Create a configured scikit-learn model by name."""
    config = config or MLTrainingConfig()
    params = resolved_model_parameters(model_name, config, final_model=final_model)

    if model_name == MODEL_RIDGE:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(**params)),
            ]
        )
    if model_name == MODEL_LASSO:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(**params)),
            ]
        )
    if model_name == MODEL_RANDOM_FOREST_REGRESSOR:
        return RandomForestRegressor(**params)
    if model_name == MODEL_EXTRA_TREES_REGRESSOR:
        return ExtraTreesRegressor(**params)
    if model_name == MODEL_GRADIENT_BOOSTING_REGRESSOR:
        return GradientBoostingRegressor(**params)
    if (
        model_name == MODEL_HIST_GRADIENT_BOOSTING_REGRESSOR
        and HistGradientBoostingRegressor is not None
    ):
        return HistGradientBoostingRegressor(**params)
    if model_name == MODEL_LOGISTIC_REGRESSION:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(**params)),
            ]
        )
    if model_name == MODEL_RANDOM_FOREST_CLASSIFIER:
        return RandomForestClassifier(**params)

    raise ValueError(f"Unsupported ML model '{model_name}'.")


def resolved_model_parameters(
    model_name: str,
    config: MLTrainingConfig,
    *,
    final_model: bool,
) -> dict[str, object]:
    """Resolve configurable model parameters into concrete sklearn kwargs."""
    if model_name not in MODEL_PARAMETERS:
        raise ValueError(f"Unsupported ML model '{model_name}'.")

    params = dict(MODEL_PARAMETERS[model_name])
    tree_estimators = (
        config.final_estimators if final_model else config.validation_estimators
    )

    for key, value in list(params.items()):
        if value != "configurable":
            continue
        if key == "n_estimators":
            params[key] = tree_estimators
        elif key == "max_iter":
            params[key] = tree_estimators
        elif key == "max_depth":
            params[key] = config.max_depth
        elif key == "min_samples_leaf":
            params[key] = config.min_samples_leaf
        elif key == "random_state":
            params[key] = config.random_state
        elif key == "n_jobs":
            params[key] = config.n_jobs
        else:
            raise ValueError(f"Unhandled configurable parameter '{key}'.")

    return params


def build_feature_dataset(
    prices: pd.DataFrame,
    horizon_days: int = 21,
    target_type: str = TARGET_TYPE_REGRESSION,
) -> pd.DataFrame:
    """Build a supervised ML dataset from historical prices.

    This wrapper preserves the original public API while delegating to the
    structured pipeline.
    """
    return ReturnPredictionPipeline(
        horizon_days=horizon_days,
        target_type=target_type,
    ).build_dataset(prices)


def train_return_model(
    dataset: pd.DataFrame,
    target_type: str | None = None,
) -> tuple[MLModel, pd.DataFrame]:
    """Train the model zoo and return the best model plus fold metrics."""
    result = train_model_zoo(dataset, target_type=target_type)
    return result.best_model, result.fold_metrics


def train_model_zoo(
    dataset: pd.DataFrame,
    target_type: str | None = None,
) -> MLTrainingResult:
    """Train all suitable models and return comparison-ready outputs."""
    target_column = _infer_target_column(dataset)
    inferred_target_type = _infer_target_type(target_column)
    resolved_target_type = (
        normalize_target_type(target_type)
        if target_type is not None
        else inferred_target_type
    )
    if resolved_target_type != inferred_target_type:
        raise ValueError(
            "The requested target_type does not match the dataset target column "
            f"'{target_column}'."
        )
    horizon_days = _infer_horizon_days(target_column)
    feature_columns = _infer_feature_columns(dataset)
    pipeline = ReturnPredictionPipeline(
        horizon_days=horizon_days,
        feature_columns=feature_columns,
        target_column_name=target_column,
        target_type=resolved_target_type,
    )
    return pipeline.train_model_zoo(dataset)


def predict_latest_returns(
    prices: pd.DataFrame,
    model: MLModel,
    horizon_days: int | None = None,
    target_type: str | None = None,
) -> pd.DataFrame:
    """Predict latest horizon returns for each selected ticker."""
    resolved_horizon = horizon_days or int(
        getattr(model, "_nivesh_horizon_days", DEFAULT_HORIZON_DAYS)
    )
    resolved_target_type = (
        normalize_target_type(target_type)
        if target_type is not None
        else getattr(model, "_nivesh_target_type", TARGET_TYPE_REGRESSION)
    )
    pipeline = ReturnPredictionPipeline(
        horizon_days=resolved_horizon,
        target_type=resolved_target_type,
    )
    return pipeline.predict_latest(prices, model)


def latest_feature_snapshot(
    prices: pd.DataFrame,
    horizon_days: int = 21,
) -> pd.DataFrame:
    """Return the latest fully populated feature row for each ticker."""
    pipeline = ReturnPredictionPipeline(horizon_days=horizon_days)
    features = pipeline.generate_features(prices)
    if features.empty:
        return pd.DataFrame(
            columns=[DATE_COLUMN, TICKER_COLUMN, *pipeline.feature_columns]
        )

    snapshot = (
        features.sort_values([DATE_COLUMN, TICKER_COLUMN])
        .groupby(TICKER_COLUMN, group_keys=False)
        .tail(1)
        .copy()
    )
    columns = [DATE_COLUMN, TICKER_COLUMN, *pipeline.feature_columns]
    return snapshot[columns].reset_index(drop=True)


def build_ml_signal_weights(
    prediction_df: pd.DataFrame,
    method: str,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
) -> pd.DataFrame:
    """Convert latest ML predictions into a constrained long-only allocation.

    Parameters
    ----------
    prediction_df:
        Latest prediction rows with a ``Ticker`` column and at least one
        prediction or probability column.
    method:
        Weighting method. Supported values are the keys or labels in
        ``ML_WEIGHT_METHOD_LABELS``.
    max_weight:
        Maximum single-stock allocation, e.g. ``0.25`` for 25%.
    min_weight:
        Drop any position below this threshold before final normalization.
    """
    if not isinstance(prediction_df, pd.DataFrame):
        raise TypeError("prediction_df must be a pandas DataFrame.")
    if prediction_df.empty:
        raise ValueError("prediction_df cannot be empty.")
    if TICKER_COLUMN not in prediction_df.columns:
        raise ValueError("prediction_df must contain a Ticker column.")

    resolved_method = _normalize_weighting_method(method)
    signals = _extract_ml_signal(prediction_df)

    if resolved_method == ML_WEIGHT_SOFTMAX:
        raw_scores = _softmax_scores(signals)
    elif resolved_method == ML_WEIGHT_RANK_TOP_K:
        raw_scores = _rank_top_k_scores(signals, max_weight)
    elif resolved_method == ML_WEIGHT_POSITIVE_ONLY:
        raw_scores = signals.clip(lower=0)
    elif resolved_method == ML_WEIGHT_VOLATILITY_ADJUSTED:
        volatility = _extract_volatility_for_weights(prediction_df)
        raw_scores = _safe_divide(signals, volatility).clip(lower=0)
    else:
        raise ValueError(f"Unsupported ML weighting method '{method}'.")

    weights = _constrain_long_only_weights(
        raw_scores,
        max_weight=max_weight,
        min_weight=min_weight,
    )

    return (
        pd.DataFrame(
            {
                TICKER_COLUMN: prediction_df[TICKER_COLUMN].astype(str).values,
                "ML Signal": signals.values,
                "Raw Weight Score": raw_scores.values,
                "Weight": weights.values,
                "Weighting Method": ML_WEIGHT_METHOD_LABELS[resolved_method],
            }
        )
        .query("Weight > 0")
        .sort_values("Weight", ascending=False)
        .reset_index(drop=True)
    )


def model_feature_importance(
    model: MLModel,
    feature_columns: tuple[str, ...] | list[str] | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Return feature importance or linear coefficients for a fitted model."""
    columns = list(
        feature_columns
        or getattr(model, "_nivesh_feature_columns", DEFAULT_MODEL_FEATURE_COLUMNS)
    )
    estimator = _final_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype="float64")
        explanation_type = "Tree Feature Importance"
        table = pd.DataFrame(
            {
                "Feature": columns,
                "Importance": values,
                "Absolute Importance": np.abs(values),
                "Explanation Type": explanation_type,
            }
        )
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype="float64")
        if coefficients.ndim > 1:
            coefficients = coefficients[0]
        table = pd.DataFrame(
            {
                "Feature": columns,
                "Coefficient": coefficients,
                "Absolute Importance": np.abs(coefficients),
                "Explanation Type": "Linear Coefficient",
            }
        )
    else:
        return pd.DataFrame(
            columns=["Feature", "Importance", "Absolute Importance", "Explanation Type"]
        )

    table = table.sort_values("Absolute Importance", ascending=False).reset_index(drop=True)
    if top_n is not None:
        table = table.head(top_n)
    return table


def add_prediction_confidence(
    prediction_df: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    feature_snapshot: pd.DataFrame,
    target_type: str,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Add confidence scores, labels, and risk warnings to predictions."""
    if prediction_df.empty:
        return prediction_df.copy()

    resolved_target_type = normalize_target_type(target_type)
    enriched = prediction_df.copy()
    features = feature_snapshot.copy()
    if not features.empty:
        feature_columns = [
            TICKER_COLUMN,
            "Volatility 20D",
            "SMA20 / SMA50 - 1",
            _return_feature_name(20),
            _return_feature_name(60),
            "Rolling Max Drawdown 60D",
        ]
        available_columns = [column for column in feature_columns if column in features.columns]
        enriched = enriched.merge(
            features[available_columns],
            on=TICKER_COLUMN,
            how="left",
        )

    signal = _extract_ml_signal(enriched).abs()
    volatility = pd.to_numeric(
        enriched.get("Volatility 20D", pd.Series(np.nan, index=enriched.index)),
        errors="coerce",
    )
    validation_score = _validation_quality_score(
        validation_metrics,
        resolved_target_type,
        model_name,
    )
    magnitude_score = _percent_rank_score(signal)
    volatility_score = 1 - _percent_rank_score(volatility.fillna(volatility.median()))

    enriched["Validation Quality Score"] = validation_score
    enriched["Prediction Strength Score"] = magnitude_score
    enriched["Volatility Safety Score"] = volatility_score.fillna(0.5)
    enriched["Confidence Score"] = (
        0.45 * enriched["Validation Quality Score"]
        + 0.35 * enriched["Prediction Strength Score"]
        + 0.20 * enriched["Volatility Safety Score"]
    ).clip(0, 1)
    enriched["Confidence Label"] = pd.cut(
        enriched["Confidence Score"],
        bins=[-0.01, 0.45, 0.70, 1.01],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    enriched["Risk Warnings"] = _prediction_risk_warnings(
        enriched,
        validation_score=validation_score,
    )
    return enriched


def build_ml_report_card(
    *,
    selected_model: str,
    target_type: str,
    horizon_days: int,
    training_rows: int,
    feature_count: int,
    model_comparison: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact ML report card for Streamlit display."""
    resolved_target_type = normalize_target_type(target_type)
    selected_row = model_comparison[model_comparison["Model"] == selected_model]
    selected_summary = selected_row.iloc[0] if not selected_row.empty else pd.Series(dtype=object)
    metric_column = _report_card_metric_column(resolved_target_type, selected_summary)
    selected_folds = fold_metrics[fold_metrics["Model"] == selected_model].copy()
    if metric_column in selected_folds.columns and not selected_folds.empty:
        higher_is_better = resolved_target_type in {
            TARGET_TYPE_CLASSIFICATION,
            TARGET_TYPE_RANKING,
        }
        ordered = selected_folds.sort_values(metric_column, ascending=not higher_is_better)
        best_fold = int(ordered.iloc[0]["Fold"])
        worst_fold = int(ordered.iloc[-1]["Fold"])
        validation_metric = selected_summary.get(f"Avg {metric_column}", np.nan)
    else:
        best_fold = np.nan
        worst_fold = np.nan
        validation_metric = np.nan

    top_features = ""
    if "Feature" in feature_importance.columns:
        top_features = ", ".join(feature_importance.head(5)["Feature"].astype(str))
    rows = [
        ("Selected Model", selected_model),
        ("Target Type", TARGET_TYPE_LABELS[resolved_target_type]),
        ("Prediction Horizon", f"{horizon_days} trading days"),
        ("Training Rows", training_rows),
        ("Feature Count", feature_count),
        ("Primary Validation Metric", metric_column),
        ("Average Validation Metric", validation_metric),
        ("Best Fold", best_fold),
        ("Worst Fold", worst_fold),
        ("Top Predictive Features", top_features or "N/A"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def target_column_for_horizon(
    horizon_days: int,
    target_type: str = TARGET_TYPE_REGRESSION,
) -> str:
    """Return the supervised target name for a prediction horizon."""
    resolved_target_type = normalize_target_type(target_type)
    if resolved_target_type == TARGET_TYPE_CLASSIFICATION:
        return f"Target {horizon_days}D Direction"
    if resolved_target_type == TARGET_TYPE_RISK_ADJUSTED:
        return f"Target {horizon_days}D Risk-Adjusted Return"
    if resolved_target_type == TARGET_TYPE_RANKING:
        return f"Target {horizon_days}D Return Rank"
    return f"Target {horizon_days}D Return"


def prediction_column_for_horizon(
    horizon_days: int,
    target_type: str = TARGET_TYPE_REGRESSION,
) -> str:
    """Return the prediction column name for a prediction horizon."""
    resolved_target_type = normalize_target_type(target_type)
    if resolved_target_type == TARGET_TYPE_CLASSIFICATION:
        return f"Predicted {horizon_days}D Direction"
    if resolved_target_type == TARGET_TYPE_RISK_ADJUSTED:
        return f"Predicted {horizon_days}D Risk-Adjusted Return"
    if resolved_target_type == TARGET_TYPE_RANKING:
        return f"Predicted {horizon_days}D Return Rank"
    return f"Predicted {horizon_days}D Return"


def probability_column_for_horizon(horizon_days: int) -> str:
    """Return the positive-direction probability column for classification."""
    return f"Probability {horizon_days}D Up"


def normalize_target_type(target_type: str) -> str:
    """Normalize user-facing target type text to an internal target key."""
    normalized = str(target_type).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "return": TARGET_TYPE_REGRESSION,
        "future_return": TARGET_TYPE_REGRESSION,
        "future_return_regression": TARGET_TYPE_REGRESSION,
        "direction": TARGET_TYPE_CLASSIFICATION,
        "direction_classification": TARGET_TYPE_CLASSIFICATION,
        "risk_adjusted": TARGET_TYPE_RISK_ADJUSTED,
        "risk_adjusted_return": TARGET_TYPE_RISK_ADJUSTED,
        "quantile": TARGET_TYPE_RANKING,
        "rank": TARGET_TYPE_RANKING,
        "ranking": TARGET_TYPE_RANKING,
        "cross_sectional_ranking": TARGET_TYPE_RANKING,
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in TARGET_TYPE_OPTIONS:
        options = ", ".join(TARGET_TYPE_OPTIONS)
        raise ValueError(f"Unsupported ML target type '{target_type}'. Choose one of: {options}.")
    return resolved


def _normalize_weighting_method(method: str) -> str:
    """Normalize an ML weighting method key or display label."""
    normalized = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    label_aliases = {
        label.lower().replace("-", "_").replace(" ", "_"): key
        for key, label in ML_WEIGHT_METHOD_LABELS.items()
    }
    aliases = {
        **label_aliases,
        "softmax": ML_WEIGHT_SOFTMAX,
        "rank": ML_WEIGHT_RANK_TOP_K,
        "rank_top_k": ML_WEIGHT_RANK_TOP_K,
        "top_k": ML_WEIGHT_RANK_TOP_K,
        "positive": ML_WEIGHT_POSITIVE_ONLY,
        "positive_only": ML_WEIGHT_POSITIVE_ONLY,
        "positive_signal_only": ML_WEIGHT_POSITIVE_ONLY,
        "volatility_adjusted": ML_WEIGHT_VOLATILITY_ADJUSTED,
        "vol_adjusted": ML_WEIGHT_VOLATILITY_ADJUSTED,
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in ML_WEIGHT_METHOD_LABELS:
        options = ", ".join(ML_WEIGHT_METHOD_LABELS)
        raise ValueError(f"Unsupported ML weighting method '{method}'. Choose one of: {options}.")
    return resolved


def _final_estimator(model: MLModel) -> MLModel:
    """Return the underlying estimator for sklearn pipelines."""
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def _validation_quality_score(
    validation_metrics: pd.DataFrame,
    target_type: str,
    model_name: str | None,
) -> float:
    """Scale validation performance to a 0-1 confidence component."""
    metrics = validation_metrics.copy()
    if model_name and "Model" in metrics.columns:
        metrics = metrics[metrics["Model"] == model_name]

    if metrics.empty:
        return 0.35

    resolved_target_type = normalize_target_type(target_type)
    if resolved_target_type == TARGET_TYPE_CLASSIFICATION:
        roc_auc_values = pd.to_numeric(
            metrics.get("ROC-AUC", pd.Series(dtype="float64")),
            errors="coerce",
        )
        metric = "ROC-AUC" if roc_auc_values.notna().any() else "F1"
        value = pd.to_numeric(
            metrics.get(metric, pd.Series(dtype="float64")),
            errors="coerce",
        ).mean()
        if metric == "ROC-AUC":
            return float(np.clip((value - 0.5) / 0.25, 0, 1))
        return float(np.clip(value, 0, 1))

    if resolved_target_type == TARGET_TYPE_RANKING:
        spearman_values = pd.to_numeric(
            metrics.get("Spearman Correlation", pd.Series(dtype="float64")),
            errors="coerce",
        )
        metric = (
            "Spearman Correlation"
            if spearman_values.notna().any()
            else "Top-3 Hit Rate"
        )
        value = pd.to_numeric(
            metrics.get(metric, pd.Series(dtype="float64")),
            errors="coerce",
        ).mean()
        if metric == "Spearman Correlation":
            return float(np.clip((value + 1) / 2, 0, 1))
        return float(np.clip(value, 0, 1))

    rmse = pd.to_numeric(metrics.get("RMSE", pd.Series(dtype="float64")), errors="coerce").mean()
    mae = pd.to_numeric(metrics.get("MAE", pd.Series(dtype="float64")), errors="coerce").mean()
    if pd.isna(rmse) or rmse <= 0:
        return 0.35
    relative_error = rmse / max(abs(mae), 1e-6)
    return float(np.clip(1 / (1 + relative_error), 0, 1))


def _percent_rank_score(values: pd.Series) -> pd.Series:
    """Return 0-1 percentile rank scores with defensive missing handling."""
    clean_values = pd.to_numeric(values, errors="coerce")
    if clean_values.notna().sum() <= 1:
        return pd.Series(0.5, index=values.index, dtype="float64")
    return clean_values.rank(method="average", pct=True).fillna(0.5)


def _prediction_risk_warnings(
    prediction_frame: pd.DataFrame,
    *,
    validation_score: float,
) -> pd.Series:
    """Create human-readable risk warning labels per prediction row."""
    warnings: list[str] = []
    volatility = pd.to_numeric(
        prediction_frame.get("Volatility 20D", pd.Series(np.nan, index=prediction_frame.index)),
        errors="coerce",
    )
    high_volatility_cutoff = volatility.quantile(0.75) if volatility.notna().any() else np.nan

    trend = pd.to_numeric(
        prediction_frame.get("SMA20 / SMA50 - 1", pd.Series(np.nan, index=prediction_frame.index)),
        errors="coerce",
    )
    return_20d = pd.to_numeric(
        prediction_frame.get(_return_feature_name(20), pd.Series(np.nan, index=prediction_frame.index)),
        errors="coerce",
    )
    return_60d = pd.to_numeric(
        prediction_frame.get(_return_feature_name(60), pd.Series(np.nan, index=prediction_frame.index)),
        errors="coerce",
    )

    for index in prediction_frame.index:
        row_warnings: list[str] = []
        if pd.notna(high_volatility_cutoff) and volatility.loc[index] >= high_volatility_cutoff:
            row_warnings.append("High volatility")
        if validation_score < 0.45:
            row_warnings.append("Weak validation")
        trend_value = trend.loc[index]
        return_20_value = return_20d.loc[index]
        return_60_value = return_60d.loc[index]
        if (
            pd.notna(trend_value)
            and pd.notna(return_20_value)
            and pd.notna(return_60_value)
            and (
                np.sign(trend_value) != np.sign(return_20_value)
                or np.sign(return_20_value) != np.sign(return_60_value)
            )
        ):
            row_warnings.append("Unstable trend")
        warnings.append(", ".join(row_warnings) if row_warnings else "None")

    return pd.Series(warnings, index=prediction_frame.index)


def _report_card_metric_column(target_type: str, selected_summary: pd.Series) -> str:
    """Choose the report card validation metric for the selected model."""
    if target_type == TARGET_TYPE_CLASSIFICATION:
        return "ROC-AUC" if pd.notna(selected_summary.get("Avg ROC-AUC", np.nan)) else "F1"
    if target_type == TARGET_TYPE_RANKING:
        return (
            "Spearman Correlation"
            if pd.notna(selected_summary.get("Avg Spearman Correlation", np.nan))
            else "Top-3 Hit Rate"
        )
    return "RMSE"


def _extract_ml_signal(prediction_df: pd.DataFrame) -> pd.Series:
    """Choose the best prediction signal column for portfolio construction."""
    frame = prediction_df.reset_index(drop=True)
    probability_columns = [
        column for column in frame.columns if column.startswith("Probability ")
    ]
    if probability_columns:
        probability = pd.to_numeric(frame[probability_columns[0]], errors="coerce")
        return probability - 0.5

    predicted_columns = [
        column
        for column in frame.columns
        if column.startswith("Predicted ") and "Annualized" not in column
    ]
    if predicted_columns:
        return pd.to_numeric(frame[predicted_columns[0]], errors="coerce")

    raise ValueError("prediction_df does not contain a usable ML prediction signal.")


def _extract_volatility_for_weights(prediction_df: pd.DataFrame) -> pd.Series:
    """Find a volatility column to scale ML signals for portfolio weights."""
    for column in ["Volatility 20D", "Annual Volatility", "Annualized Volatility"]:
        if column in prediction_df.columns:
            volatility = pd.to_numeric(prediction_df[column], errors="coerce")
            return volatility.replace(0, np.nan)
    raise ValueError(
        "Volatility-adjusted ML weights require a Volatility 20D or annual volatility column."
    )


def _softmax_scores(signals: pd.Series) -> pd.Series:
    """Convert arbitrary signals into stable positive softmax scores."""
    clean_signals = pd.to_numeric(signals, errors="coerce").fillna(0.0)
    signal_std = float(clean_signals.std())
    if signal_std > 0 and not np.isnan(signal_std):
        scaled = (clean_signals - clean_signals.mean()) / signal_std
    else:
        scaled = clean_signals - clean_signals.mean()
    exp_scores = np.exp(np.clip(scaled, -5, 5))
    return pd.Series(exp_scores, index=signals.index, dtype="float64")


def _rank_top_k_scores(signals: pd.Series, max_weight: float) -> pd.Series:
    """Create descending rank scores for the top-k predicted tickers."""
    clean_signals = pd.to_numeric(signals, errors="coerce").fillna(-np.inf)
    min_assets_for_cap = int(np.ceil(1 / max_weight)) if max_weight > 0 else 1
    top_k = min(len(clean_signals), max(5, min_assets_for_cap))
    top_index = clean_signals.nlargest(top_k).index
    scores = pd.Series(0.0, index=signals.index, dtype="float64")
    scores.loc[top_index] = np.arange(len(top_index), 0, -1, dtype="float64")
    return scores


def _constrain_long_only_weights(
    raw_scores: pd.Series,
    *,
    max_weight: float,
    min_weight: float,
) -> pd.Series:
    """Normalize, threshold, and cap long-only portfolio weights."""
    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be greater than 0 and less than or equal to 1.")
    if min_weight < 0 or min_weight >= 1:
        raise ValueError("min_weight must be greater than or equal to 0 and less than 1.")
    if min_weight > max_weight:
        raise ValueError("min_weight cannot be greater than max_weight.")

    raw = pd.to_numeric(raw_scores, errors="coerce").replace([np.inf, -np.inf], np.nan)
    raw = raw.fillna(0.0).clip(lower=0.0).astype("float64")
    if raw.sum() <= 0:
        raise ValueError("No positive ML signals are available for this weighting method.")

    weights = raw / raw.sum()
    if min_weight > 0:
        small_positions = (weights > 0) & (weights < min_weight)
        candidate = weights.mask(small_positions, 0.0)
        if candidate.sum() > 0 and int((candidate > 0).sum()) * max_weight >= 1:
            raw = raw.where(candidate > 0, 0.0)

    active = raw > 0
    active_count = int(active.sum())
    if active_count == 0:
        raise ValueError("All ML weights were removed by the minimum threshold.")
    if active_count * max_weight < 1 - 1e-9:
        raise ValueError(
            "The selected ML signals cannot satisfy the max_weight cap. "
            "Increase max_weight or choose a broader weighting method."
        )

    capped_weights = pd.Series(0.0, index=raw.index, dtype="float64")
    remaining_raw = raw.loc[active].copy()
    remaining_weight = 1.0

    while not remaining_raw.empty:
        if remaining_raw.sum() <= 0:
            provisional = pd.Series(
                remaining_weight / len(remaining_raw),
                index=remaining_raw.index,
                dtype="float64",
            )
        else:
            provisional = remaining_weight * remaining_raw / remaining_raw.sum()

        over_cap = provisional > max_weight
        if not over_cap.any():
            capped_weights.loc[provisional.index] = provisional
            break

        capped_index = provisional[over_cap].index
        capped_weights.loc[capped_index] = max_weight
        remaining_weight -= max_weight * len(capped_index)
        remaining_raw = remaining_raw.drop(capped_index)

        if remaining_weight < -1e-9:
            raise ValueError("ML portfolio weights could not satisfy the max_weight cap.")
        if remaining_weight <= 1e-12:
            break

    total_weight = float(capped_weights.sum())
    if total_weight <= 0:
        raise ValueError("ML portfolio weights could not be normalized.")
    capped_weights = capped_weights / total_weight

    if (capped_weights > max_weight + 1e-8).any():
        raise ValueError("ML portfolio weights could not satisfy the max_weight cap.")
    return capped_weights


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two aligned series while treating zero denominators as missing."""
    return numerator / denominator.replace(0, np.nan)


def _relative_strength_index(prices: pd.Series, window: int) -> pd.Series:
    """Compute trailing RSI from prices using only historical observations."""
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    rsi = rsi.mask((average_gain == 0) & (average_loss > 0), 0.0)
    return rsi


def _moving_average_convergence_divergence(prices: pd.Series) -> pd.Series:
    """Compute normalized MACD with trailing exponential moving averages."""
    fast_ema = prices.ewm(
        span=MACD_FAST_SPAN,
        adjust=False,
        min_periods=MACD_SLOW_SPAN,
    ).mean()
    slow_ema = prices.ewm(
        span=MACD_SLOW_SPAN,
        adjust=False,
        min_periods=MACD_SLOW_SPAN,
    ).mean()
    return _safe_divide(fast_ema - slow_ema, prices)


def _bollinger_band_position(
    prices: pd.Series,
    *,
    window: int,
    std_multiplier: float,
) -> pd.Series:
    """Return trailing Bollinger Band position where 0 is lower and 1 is upper."""
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    lower_band = rolling_mean - std_multiplier * rolling_std
    upper_band = rolling_mean + std_multiplier * rolling_std
    return _safe_divide(prices - lower_band, upper_band - lower_band)


def _rolling_max_drawdown(prices: pd.Series, window: int) -> pd.Series:
    """Compute the maximum drawdown observed inside each trailing window."""
    return prices.rolling(window).apply(_max_drawdown_from_window, raw=True)


def _max_drawdown_from_window(values: np.ndarray) -> float:
    """Return the worst peak-to-trough drawdown in one price window."""
    if len(values) == 0 or np.isnan(values).any():
        return np.nan
    running_peak = np.maximum.accumulate(values)
    drawdowns = values / running_peak - 1
    return float(np.min(drawdowns))


def _add_cross_sectional_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add same-date cross-sectional ranks across the selected tickers."""
    ranked = features.copy()
    ranked["Ticker Return Rank"] = ranked.groupby(DATE_COLUMN)[
        _return_feature_name(20)
    ].rank(method="average", pct=True)
    ranked["Volatility Rank"] = ranked.groupby(DATE_COLUMN)[
        _volatility_feature_name(20)
    ].rank(method="average", pct=True)
    ranked["Momentum Rank"] = ranked.groupby(DATE_COLUMN)[
        _momentum_feature_name(60)
    ].rank(method="average", pct=True)
    return ranked


def _positive_class_probability(model: MLModel, x_data: pd.DataFrame) -> np.ndarray:
    """Return probability estimates for the positive direction class."""
    if not hasattr(model, "predict_proba"):
        return np.asarray(model.predict(x_data), dtype=float)

    probabilities = model.predict_proba(x_data)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    if len(classes) == 1:
        return np.ones(len(x_data)) if classes[0] == 1 else np.zeros(len(x_data))
    return np.full(len(x_data), np.nan)


def _safe_roc_auc(actual: pd.Series, probability: np.ndarray) -> float:
    """Calculate ROC-AUC when both classes are present; otherwise return NaN."""
    if actual.nunique(dropna=True) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(actual, probability))
    except ValueError:
        return float("nan")


def _ranking_validation_metrics(
    validation_frame: pd.DataFrame,
    *,
    actual_column: str,
    prediction_column: str,
    top_k: int = 3,
) -> tuple[float, float]:
    """Return date-wise Spearman correlation and top-k hit rate."""
    spearman_values: list[float] = []
    hit_rates: list[float] = []

    for _, group in validation_frame.groupby(DATE_COLUMN):
        if len(group) < 2:
            continue

        if group[actual_column].nunique(dropna=True) > 1 and group[
            prediction_column
        ].nunique(dropna=True) > 1:
            spearman = group[actual_column].corr(
                group[prediction_column],
                method="spearman",
            )
            if pd.notna(spearman):
                spearman_values.append(float(spearman))

        effective_k = min(top_k, len(group))
        actual_top = set(
            group.nlargest(effective_k, actual_column)[TICKER_COLUMN].astype(str)
        )
        predicted_top = set(
            group.nlargest(effective_k, prediction_column)[TICKER_COLUMN].astype(str)
        )
        if actual_top:
            hit_rates.append(len(actual_top.intersection(predicted_top)) / effective_k)

    spearman_average = float(np.mean(spearman_values)) if spearman_values else float("nan")
    hit_rate_average = float(np.mean(hit_rates)) if hit_rates else float("nan")
    return spearman_average, hit_rate_average


def _metric_columns_for_target(target_type: str) -> list[str]:
    """Return validation metric columns for a target type."""
    resolved_target_type = normalize_target_type(target_type)
    if resolved_target_type == TARGET_TYPE_CLASSIFICATION:
        return ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    if resolved_target_type == TARGET_TYPE_RANKING:
        return ["Spearman Correlation", "Top-3 Hit Rate"]
    return ["MAE", "RMSE", "R2"]


def _empty_metric_values(target_type: str) -> dict[str, float]:
    """Return NaN metric placeholders for a failed model fold."""
    return {column: float("nan") for column in _metric_columns_for_target(target_type)}


def _selection_metric_for_target(
    target_type: str,
    comparison: pd.DataFrame,
) -> tuple[str, bool]:
    """Return the comparison column and sort direction used to pick a model."""
    resolved_target_type = normalize_target_type(target_type)
    if resolved_target_type == TARGET_TYPE_CLASSIFICATION:
        if "Avg ROC-AUC" in comparison and comparison["Avg ROC-AUC"].notna().any():
            return "Avg ROC-AUC", True
        return "Avg F1", True
    if resolved_target_type == TARGET_TYPE_RANKING:
        if (
            "Avg Spearman Correlation" in comparison
            and comparison["Avg Spearman Correlation"].notna().any()
        ):
            return "Avg Spearman Correlation", True
        return "Avg Top-3 Hit Rate", True
    return "Avg RMSE", False


def _attach_model_metadata(
    model: MLModel,
    *,
    horizon_days: int,
    target_type: str,
    feature_columns: tuple[str, ...],
    model_name: str,
) -> None:
    """Store lightweight prediction metadata on fitted sklearn models."""
    setattr(model, "_nivesh_horizon_days", horizon_days)
    setattr(model, "_nivesh_target_type", target_type)
    setattr(model, "_nivesh_feature_columns", tuple(feature_columns))
    setattr(model, "_nivesh_model_name", model_name)


def _empty_feature_frame(feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Return an empty feature frame with the expected display columns."""
    return pd.DataFrame(columns=[DATE_COLUMN, TICKER_COLUMN, *feature_columns])


def _infer_target_column(dataset: pd.DataFrame) -> str:
    """Infer the target column from a supervised ML dataset."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame.")
    if TARGET_COLUMN in dataset.columns:
        return TARGET_COLUMN

    candidates = [
        column
        for column in dataset.columns
        if column.startswith("Target ") and "D " in column
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError("ML dataset is missing a target return column.")
    raise ValueError(
        "ML dataset contains multiple target columns; pass a dataset with one target."
    )


def _infer_horizon_days(target_column: str) -> int:
    """Infer the horizon from target columns such as 'Target 21D Return'."""
    try:
        return int(target_column.split("Target ", 1)[1].split("D", 1)[0])
    except (IndexError, ValueError):
        return DEFAULT_HORIZON_DAYS


def _infer_target_type(target_column: str) -> str:
    """Infer the target type from a target column name."""
    if "Direction" in target_column:
        return TARGET_TYPE_CLASSIFICATION
    if "Risk-Adjusted" in target_column:
        return TARGET_TYPE_RISK_ADJUSTED
    if "Rank" in target_column:
        return TARGET_TYPE_RANKING
    return TARGET_TYPE_REGRESSION


def _infer_feature_columns(dataset: pd.DataFrame) -> tuple[str, ...]:
    """Select the richest compatible feature set available in a dataset."""
    full_feature_set = [
        column for column in DEFAULT_MODEL_FEATURE_COLUMNS if column in dataset.columns
    ]
    if len(full_feature_set) == len(DEFAULT_MODEL_FEATURE_COLUMNS):
        return tuple(full_feature_set)

    legacy_feature_set = [
        column for column in LEGACY_FEATURE_COLUMNS if column in dataset.columns
    ]
    if len(legacy_feature_set) == len(LEGACY_FEATURE_COLUMNS):
        return tuple(legacy_feature_set)

    if full_feature_set:
        return tuple(full_feature_set)

    raise ValueError("ML dataset does not contain usable feature columns.")


def _feature_columns_for_model(
    model: MLModel,
    latest_rows: pd.DataFrame,
    fallback_columns: tuple[str, ...],
) -> list[str]:
    """Use model feature names when available to support older fitted models."""
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None:
        model_features = fallback_columns

    feature_columns = [str(column) for column in model_features]
    missing = [column for column in feature_columns if column not in latest_rows.columns]
    if missing:
        raise ValueError(
            "Latest feature frame is missing model feature columns: "
            f"{', '.join(missing)}"
        )
    return feature_columns
