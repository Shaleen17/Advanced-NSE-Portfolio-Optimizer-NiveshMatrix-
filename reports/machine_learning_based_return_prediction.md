# Machine Learning Based Return Prediction

This section adds a basic machine learning layer to the portfolio optimization project. The goal is educational: predict each selected NSE stock's next 21 trading day return and compare an ML-predicted-return portfolio with a historical mean-return optimized portfolio and an equal weight portfolio.

The input data is adjusted close price data downloaded from Yahoo Finance. Daily returns are calculated as the percentage change from one trading day to the next. The prediction target is the return from today's close to the close 21 trading days later.

Feature guide:
- 5-day return: the stock's return over roughly one trading week. It captures very short-term momentum.
- 10-day return: the stock's return over roughly two trading weeks. It smooths short-term movement slightly more than the 5-day return.
- 20-day return: the stock's return over roughly one trading month. It captures monthly momentum.
- 50-day moving average ratio: today's price divided by the 50-day average price, minus 1. A positive value means price is above its medium-term trend.
- 100-day moving average ratio: today's price divided by the 100-day average price, minus 1. This gives a slower trend signal.
- 20-day volatility: the standard deviation of daily returns over the last 20 trading days. Higher volatility means recent price movement has been less stable.
- RSI 14D: a momentum indicator comparing recent gains with recent losses. It can describe whether recent price action has been strong or weak, but it is not reliable as a standalone trading signal.

Random train-test splitting is wrong for time series because it allows the model to train on future market regimes and test on earlier dates. That creates look-ahead bias. This project uses time-series splits, where each test period happens after its training period, which better matches how a real forecasting workflow would operate.

Two simple models are trained: Linear Regression and Random Forest Regressor. Linear Regression checks whether the features have a simple linear relationship with future returns. Random Forest can capture nonlinear relationships, but it can also overfit if it learns patterns that only existed in the past.

Average validation results from time-series splits:
- Best model by MAE: Linear Regression
- MAE: 0.0506
- RMSE: 0.0655
- Directional accuracy: 52.74%

Overfitting risk is high in financial machine learning. A model may look good on historical data because it memorized noise, one market cycle, or a temporary relationship. Simpler models, time-series validation, and conservative interpretation help reduce this risk, but they do not remove it.

ML predictions in finance are noisy because stock prices react to new information, macro events, earnings, liquidity, sentiment, regulations, and random market behavior. Historical price features contain only a small part of that information. For that reason, the ML portfolio should be treated as an experiment, not as a guarantee of profit.
