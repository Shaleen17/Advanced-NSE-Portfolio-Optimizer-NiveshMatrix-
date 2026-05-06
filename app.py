"""Advanced NSE Portfolio Optimizer Streamlit application."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    BRAND_NAME,
    CHART_COLORS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TRANSACTION_COST,
    LOGO_PATH,
    NSE_TICKERS,
    PROJECT_NAME,
    RISK_FREE_RATE,
    ensure_project_folders,
)
from src.backtest import compare_backtest_to_equal_weight, run_monthly_rebalance_backtest
from src.black_litterman import black_litterman_allocation
from src.data_loader import DataLoadError, get_benchmark_returns, get_price_data
from src.factor_investing import build_factor_portfolio_weights, build_factor_score_table
from src.metrics import (
    build_allocation_table,
    build_strategy_comparison,
    calculate_annual_covariance,
    calculate_annual_returns,
    calculate_annual_volatility,
    calculate_correlation_matrix,
    calculate_daily_returns,
    calculate_portfolio_performance,
    equal_weight_vector,
    random_weight_vector,
    summarize_assets,
)
from src.ml_models import build_feature_dataset, predict_latest_returns, train_return_model
from src.optimizer import (
    OptimizationError,
    optimize_max_sharpe,
    optimize_min_volatility,
    run_monte_carlo_simulation,
)
from src.reporting import (
    dataframe_to_csv_bytes,
    final_conclusion,
    formula_reference,
    future_scope_items,
    methodology_steps,
    project_file_purpose,
)
from src.risk import build_risk_table, build_strategy_return_frame
from src.visualization import (
    allocation_chart,
    correlation_heatmap,
    cumulative_return_chart,
    drawdown_chart,
    efficient_frontier_chart,
    factor_score_chart,
    ml_prediction_chart,
    price_trend_chart,
    returns_distribution_chart,
    turnover_chart,
)


st.set_page_config(
    page_title=f"{BRAND_NAME} | {PROJECT_NAME}",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    """Inject the pure black fintech dashboard theme."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {CHART_COLORS["background"]};
            color: {CHART_COLORS["text"]};
        }}
        [data-testid="stSidebar"] {{
            background: {CHART_COLORS["background"]};
            border-right: 1px solid {CHART_COLORS["border"]};
        }}
        h1, h2, h3, h4, h5, h6, p, span, div, label {{
            color: {CHART_COLORS["text"]};
        }}
        .block-container {{
            padding-top: 1.5rem;
            max-width: 1400px;
        }}
        .hero {{
            background: linear-gradient(135deg, #000000 0%, #101010 100%);
            border: 1px solid {CHART_COLORS["border"]};
            border-radius: 8px;
            padding: 22px;
            margin-bottom: 18px;
        }}
        .hero-title {{
            font-size: 34px;
            font-weight: 800;
            line-height: 1.15;
        }}
        .hero-subtitle {{
            color: {CHART_COLORS["muted"]};
            font-size: 15px;
            margin-top: 8px;
        }}
        .metric-card {{
            background: {CHART_COLORS["panel"]};
            border: 1px solid {CHART_COLORS["border"]};
            border-radius: 8px;
            padding: 16px;
            min-height: 98px;
        }}
        .metric-label {{
            color: {CHART_COLORS["muted"]};
            font-size: 13px;
        }}
        .metric-value {{
            font-size: 25px;
            font-weight: 800;
            margin-top: 8px;
        }}
        .positive {{
            color: {CHART_COLORS["profit"]};
        }}
        .negative {{
            color: {CHART_COLORS["loss"]};
        }}
        .info-card {{
            background: {CHART_COLORS["panel_2"]};
            border: 1px solid {CHART_COLORS["border"]};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }}
        .small-muted {{
            color: {CHART_COLORS["muted"]};
            font-size: 13px;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {CHART_COLORS["border"]};
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, signed_value: float | None = None) -> None:
    """Render a custom metric card with green/red signed value styling."""
    value_class = ""
    if signed_value is not None:
        if signed_value > 0:
            value_class = "positive"
        elif signed_value < 0:
            value_class = "negative"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {value_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str) -> None:
    """Render an explanatory card."""
    st.markdown(
        f"""
        <div class="info-card">
            <strong>{title}</strong>
            <div class="small-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_percent(value: float) -> str:
    """Format a float as a percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def format_number(value: float) -> str:
    """Format a float with four decimals."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def style_numeric_table(
    dataframe: pd.DataFrame,
    percent_columns: list[str] | None = None,
    number_columns: list[str] | None = None,
):
    """Style numeric tables with green positives and red negatives."""
    percent_columns = percent_columns or []
    number_columns = number_columns or []
    formatters = {column: "{:.2%}" for column in percent_columns if column in dataframe.columns}
    formatters.update({column: "{:.4f}" for column in number_columns if column in dataframe.columns})

    def color_signed(value):
        if isinstance(value, (int, float, np.number)) and not pd.isna(value):
            if value > 0:
                return f"color: {CHART_COLORS['profit']}"
            if value < 0:
                return f"color: {CHART_COLORS['loss']}"
        return f"color: {CHART_COLORS['text']}"

    return dataframe.style.format(formatters).applymap(color_signed)


@st.cache_data(show_spinner=False)
def load_prices_cached(
    tickers: tuple[str, ...],
    start_date: date,
    end_date: date,
    use_cache_first: bool,
) -> pd.DataFrame:
    """Cache price loading for faster Streamlit reruns."""
    return get_price_data(list(tickers), start_date, end_date, use_cache_first)


@st.cache_data(show_spinner=False)
def load_benchmark_cached(start_date: date, end_date: date, use_cache_first: bool) -> pd.Series | None:
    """Cache benchmark loading for faster Streamlit reruns."""
    return get_benchmark_returns(start_date, end_date, use_cache_first)


def sidebar_inputs() -> tuple[list[str], date, date, bool, int, int, float, float, int]:
    """Collect sidebar settings."""
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    st.sidebar.title(BRAND_NAME)

    selected_tickers = st.sidebar.multiselect(
        "NSE stock universe",
        options=NSE_TICKERS,
        default=NSE_TICKERS[:12],
    )
    start_date = st.sidebar.date_input("Start date", value=DEFAULT_START_DATE)
    end_date = st.sidebar.date_input("End date", value=DEFAULT_END_DATE)
    use_cache_first = st.sidebar.checkbox("Use cached data first", value=True)
    portfolio_count = st.sidebar.slider("Monte Carlo portfolios", 500, 10000, 2500, 500)
    random_seed = st.sidebar.number_input("Random seed", min_value=1, value=42, step=1)
    risk_free_rate = st.sidebar.slider("Risk-free rate", 0.0, 0.15, RISK_FREE_RATE, 0.005)
    transaction_cost = st.sidebar.slider(
        "Transaction cost assumption",
        0.0,
        0.01,
        DEFAULT_TRANSACTION_COST,
        0.0005,
        format="%.4f",
    )
    factor_size = st.sidebar.slider("Factor portfolio size", 2, 20, 10, 1)
    return (
        selected_tickers,
        start_date,
        end_date,
        use_cache_first,
        portfolio_count,
        int(random_seed),
        risk_free_rate,
        transaction_cost,
        factor_size,
    )


def render_hero() -> None:
    """Render the main dashboard heading."""
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{PROJECT_NAME}</div>
            <div class="hero-subtitle">
                Python and Streamlit dashboard for Indian NSE portfolio optimization using
                Modern Portfolio Theory, Efficient Frontier analysis, SciPy optimization,
                backtesting, risk analytics, Black-Litterman concepts, factor investing,
                and machine learning experiments.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit dashboard."""
    ensure_project_folders()
    inject_css()

    (
        selected_tickers,
        start_date,
        end_date,
        use_cache_first,
        portfolio_count,
        random_seed,
        risk_free_rate,
        transaction_cost,
        factor_size,
    ) = sidebar_inputs()

    render_hero()

    try:
        prices = load_prices_cached(tuple(selected_tickers), start_date, end_date, use_cache_first)
        daily_returns = calculate_daily_returns(prices)
        annual_returns = calculate_annual_returns(daily_returns)
        annual_volatility = calculate_annual_volatility(daily_returns)
        annual_covariance = calculate_annual_covariance(daily_returns)
        correlation_matrix = calculate_correlation_matrix(daily_returns)
        benchmark_returns = load_benchmark_cached(start_date, end_date, use_cache_first)

        equal_weights = equal_weight_vector(len(prices.columns))
        random_weights = random_weight_vector(len(prices.columns), random_seed)
        max_sharpe_weights = optimize_max_sharpe(annual_returns, annual_covariance, risk_free_rate)
        min_volatility_weights = optimize_min_volatility(annual_covariance)
        random_results = run_monte_carlo_simulation(
            annual_returns,
            annual_covariance,
            portfolio_count=portfolio_count,
            risk_free_rate=risk_free_rate,
            seed=random_seed,
        )
    except (DataLoadError, OptimizationError, ValueError) as error:
        st.error(str(error))
        st.stop()

    strategy_weights = {
        "Equal Weight": equal_weights,
        "Random Portfolio": random_weights,
        "Maximum Sharpe": max_sharpe_weights,
        "Minimum Volatility": min_volatility_weights,
    }
    comparison = build_strategy_comparison(
        strategy_weights, annual_returns, annual_covariance, risk_free_rate
    )
    strategy_returns = build_strategy_return_frame(daily_returns, strategy_weights)
    risk_table = build_risk_table(strategy_returns, benchmark_returns, risk_free_rate)
    best_strategy = comparison.iloc[0]

    top_cards = st.columns(4)
    with top_cards[0]:
        metric_card("Selected Stocks", str(len(prices.columns)))
    with top_cards[1]:
        metric_card("Trading Days", str(len(prices)))
    with top_cards[2]:
        metric_card("Best Strategy", best_strategy["Strategy"], best_strategy["Sharpe Ratio"])
    with top_cards[3]:
        metric_card("Best Sharpe", format_number(best_strategy["Sharpe Ratio"]), best_strategy["Sharpe Ratio"])

    tabs = st.tabs(
        [
            "Basic",
            "Intermediate",
            "Advanced",
            "Expert Risk",
            "Backtesting",
            "Black-Litterman",
            "Factor Investing",
            "Machine Learning",
            "Report",
            "Downloads",
        ]
    )

    with tabs[0]:
        st.subheader("Basic: Understanding Portfolio Optimization")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            info_card(
                "What is portfolio optimization?",
                "It is the process of choosing stock weights to balance expected return and risk.",
            )
        with col_b:
            info_card(
                "Why is stock data required?",
                "Historical prices are needed to estimate returns, volatility, covariance, and drawdowns.",
            )
        with col_c:
            info_card(
                "What is yfinance?",
                "yfinance is a Python library that retrieves market data from Yahoo Finance for research use.",
            )

        st.markdown("#### Adjusted Close Price Data")
        st.dataframe(prices.head(), use_container_width=True)

        missing_table = pd.DataFrame(
            {
                "Ticker": prices.columns,
                "Missing Values After Cleaning": prices.isna().sum().values,
            }
        )
        st.markdown("#### Missing Value Handling")
        st.dataframe(missing_table, use_container_width=True, hide_index=True)
        st.pyplot(price_trend_chart(prices), use_container_width=True)

    with tabs[1]:
        st.subheader("Intermediate: Returns, Risk, Covariance, and Diversification")
        info_card(
            "Diversification",
            "Diversification means combining stocks that do not move exactly together. "
            "Correlation and covariance help measure this relationship.",
        )

        st.markdown("#### Daily Returns")
        st.dataframe(daily_returns.head(), use_container_width=True)

        asset_summary = summarize_assets(prices)
        st.markdown("#### Annual Expected Return and Annual Volatility")
        st.dataframe(
            style_numeric_table(
                asset_summary,
                percent_columns=["Total Return", "Expected Annual Return", "Annual Volatility"],
            ),
            use_container_width=True,
        )

        st.markdown("#### Covariance Matrix")
        st.dataframe(annual_covariance, use_container_width=True)
        st.markdown("#### Correlation Matrix")
        st.dataframe(correlation_matrix, use_container_width=True)
        st.pyplot(correlation_heatmap(correlation_matrix), use_container_width=True)

        random_performance = calculate_portfolio_performance(
            random_weights, annual_returns, annual_covariance, risk_free_rate
        )
        cards = st.columns(3)
        with cards[0]:
            metric_card("Random Portfolio Return", format_percent(random_performance["Expected Annual Return"]), random_performance["Expected Annual Return"])
        with cards[1]:
            metric_card("Random Portfolio Risk", format_percent(random_performance["Annual Risk"]))
        with cards[2]:
            metric_card("Random Portfolio Sharpe", format_number(random_performance["Sharpe Ratio"]), random_performance["Sharpe Ratio"])
        st.pyplot(returns_distribution_chart(daily_returns), use_container_width=True)

    with tabs[2]:
        st.subheader("Advanced: Monte Carlo, Efficient Frontier, and SciPy Optimization")
        info_card(
            "Optimization constraints",
            "The optimized portfolios are long-only. Total weights must equal 1, no short selling is allowed, and each stock weight is between 0 and 1.",
        )
        st.pyplot(efficient_frontier_chart(random_results, comparison), use_container_width=True)

        st.markdown("#### Strategy Comparison")
        st.dataframe(
            style_numeric_table(
                comparison,
                percent_columns=["Expected Annual Return", "Annual Risk"],
                number_columns=["Sharpe Ratio"],
            ),
            use_container_width=True,
        )

        allocations = {
            "Equal Weight": build_allocation_table(prices.columns.tolist(), equal_weights),
            "Random Portfolio": build_allocation_table(prices.columns.tolist(), random_weights),
            "Maximum Sharpe": build_allocation_table(prices.columns.tolist(), max_sharpe_weights),
            "Minimum Volatility": build_allocation_table(prices.columns.tolist(), min_volatility_weights),
        }
        selected_allocation = st.selectbox("Allocation to visualize", list(allocations.keys()))
        selected_table = allocations[selected_allocation]
        st.dataframe(
            style_numeric_table(selected_table, percent_columns=["Weight"]),
            use_container_width=True,
        )
        st.pyplot(
            allocation_chart(selected_table, "Weight", f"{selected_allocation} Allocation"),
            use_container_width=True,
        )

    with tabs[3]:
        st.subheader("Expert: Advanced Risk Metrics")
        st.dataframe(
            style_numeric_table(
                risk_table,
                percent_columns=[
                    "Annualized Return",
                    "Annualized Volatility",
                    "Maximum Drawdown",
                    "Daily VaR 95%",
                    "Daily CVaR 95%",
                    "Tracking Error",
                ],
                number_columns=[
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Calmar Ratio",
                    "Beta vs NIFTY 50",
                    "Information Ratio",
                ],
            ),
            use_container_width=True,
        )
        st.pyplot(cumulative_return_chart(strategy_returns, "Strategy Cumulative Returns"), use_container_width=True)
        st.pyplot(drawdown_chart(strategy_returns, "Strategy Drawdowns"), use_container_width=True)

    with tabs[4]:
        st.subheader("Expert: Monthly Rebalancing Backtest")
        try:
            backtest_returns, backtest_weights, turnover_table = run_monthly_rebalance_backtest(
                daily_returns,
                transaction_cost=transaction_cost,
            )
            backtest_frame, backtest_metrics = compare_backtest_to_equal_weight(
                daily_returns,
                backtest_returns,
                benchmark_returns,
            )
            st.dataframe(
                style_numeric_table(
                    backtest_metrics,
                    percent_columns=[
                        "Annualized Return",
                        "Annualized Volatility",
                        "Maximum Drawdown",
                        "Daily VaR 95%",
                        "Daily CVaR 95%",
                        "Tracking Error",
                    ],
                    number_columns=["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"],
                ),
                use_container_width=True,
            )
            st.pyplot(cumulative_return_chart(backtest_frame, "Backtest Cumulative Returns"), use_container_width=True)
            st.pyplot(turnover_chart(turnover_table), use_container_width=True)
            with st.expander("View monthly backtest weights"):
                st.dataframe(backtest_weights, use_container_width=True)
        except ValueError as error:
            st.warning(str(error))

    with tabs[5]:
        st.subheader("Expert: Black-Litterman Model")
        try:
            bl_weights, bl_table = black_litterman_allocation(prices)
            bl_allocation = pd.DataFrame(
                {"Ticker": prices.columns, "Black-Litterman Weight": bl_weights}
            ).sort_values("Black-Litterman Weight", ascending=False)
            st.dataframe(
                style_numeric_table(
                    bl_table,
                    percent_columns=[
                        "Historical Annual Return",
                        "Equilibrium Return",
                        "Momentum View Return",
                        "Black-Litterman Return",
                        "Black-Litterman Weight",
                    ],
                ),
                use_container_width=True,
            )
            st.pyplot(
                allocation_chart(bl_allocation, "Black-Litterman Weight", "Black-Litterman Allocation"),
                use_container_width=True,
            )
        except Exception as error:
            st.warning(f"Black-Litterman model could not be calculated: {error}")

    with tabs[6]:
        st.subheader("Expert: Factor Investing Analysis")
        factor_scores = build_factor_score_table(prices)
        factor_weights = build_factor_portfolio_weights(factor_scores, factor_size)
        st.dataframe(
            style_numeric_table(
                factor_scores,
                percent_columns=["Momentum 6M", "Momentum 12M", "Annual Volatility", "Trend Ratio"],
                number_columns=["Momentum Score", "Low Volatility Score", "Trend Score", "Overall Factor Score"],
            ),
            use_container_width=True,
        )
        st.pyplot(factor_score_chart(factor_scores), use_container_width=True)
        st.markdown("#### Factor Portfolio Weights")
        st.dataframe(
            style_numeric_table(
                factor_weights,
                percent_columns=["Factor Weight"],
                number_columns=["Overall Factor Score"],
            ),
            use_container_width=True,
        )

    with tabs[7]:
        st.subheader("Expert: Machine Learning Return Prediction")
        info_card(
            "Important ML note",
            "Machine learning predictions are experimental. Stock returns are noisy, and this model should not be treated as a trading signal or profit guarantee.",
        )
        if st.button("Run ML return prediction"):
            try:
                dataset = build_feature_dataset(prices)
                model, ml_metrics = train_return_model(dataset)
                predictions = predict_latest_returns(prices, model)
                st.markdown("#### Validation Metrics")
                st.dataframe(
                    style_numeric_table(
                        ml_metrics,
                        percent_columns=["Directional Accuracy"],
                        number_columns=["MAE", "RMSE"],
                    ),
                    use_container_width=True,
                )
                st.markdown("#### Latest Predictions")
                st.dataframe(
                    style_numeric_table(
                        predictions,
                        percent_columns=["Predicted 21D Return", "Annualized Predicted Return"],
                    ),
                    use_container_width=True,
                )
                st.pyplot(ml_prediction_chart(predictions), use_container_width=True)
            except Exception as error:
                st.warning(f"ML prediction could not be completed: {error}")
        else:
            st.caption("Click the button to train the model. This may take a little time.")

    with tabs[8]:
        st.subheader("College Project Report Content")
        st.markdown("#### Methodology")
        for step in methodology_steps():
            st.markdown(f"- {step}")
        st.markdown("#### Formula Reference")
        st.dataframe(formula_reference(), use_container_width=True, hide_index=True)
        st.markdown("#### System File Map")
        st.dataframe(project_file_purpose(), use_container_width=True, hide_index=True)
        st.markdown("#### Final Conclusion")
        st.write(final_conclusion())
        st.markdown("#### Future Scope")
        for item in future_scope_items():
            st.markdown(f"- {item}")
        st.markdown("#### Educational Disclaimer")
        st.warning(
            "This project is for educational and academic use only. It is not financial advice, "
            "not investment advice, and does not guarantee returns."
        )

    with tabs[9]:
        st.subheader("Downloadable CSV Outputs")
        st.download_button(
            "Download strategy comparison",
            data=dataframe_to_csv_bytes(comparison),
            file_name="strategy_comparison.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download risk metrics",
            data=dataframe_to_csv_bytes(risk_table),
            file_name="risk_metrics.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download asset summary",
            data=dataframe_to_csv_bytes(summarize_assets(prices)),
            file_name="asset_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download Monte Carlo portfolios",
            data=dataframe_to_csv_bytes(random_results),
            file_name="monte_carlo_portfolios.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
