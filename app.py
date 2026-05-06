"""Advanced NSE Portfolio Optimizer Streamlit application."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    AUTH_IMAGE_PATH,
    BRAND_NAME,
    CHART_COLORS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TRANSACTION_COST,
    HOME_IMAGE_PATH,
    LOGO_PATH,
    NSE_TICKERS,
    PROJECT_NAME,
    RISK_FREE_RATE,
    ensure_project_folders,
)
from src.auth import authenticate_user, create_user, image_to_data_uri
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
        #MainMenu, footer, [data-testid="stToolbar"], .stDeployButton {{
            display: none !important;
            visibility: hidden !important;
        }}
        .stApp {{
            background: {CHART_COLORS["background"]};
            color: {CHART_COLORS["text"]};
        }}
        [data-testid="stSidebar"] {{
            background:
                radial-gradient(circle at 25% 0%, rgba(255, 0, 45, 0.22), transparent 30%),
                linear-gradient(180deg, #000000 0%, #090000 45%, #160006 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.10);
            box-shadow: 18px 0 40px rgba(255, 0, 45, 0.10);
        }}
        [data-testid="stSidebar"] img {{
            max-width: 160px;
            margin: 10px auto 18px auto;
            display: block;
        }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: #FFFFFF !important;
            letter-spacing: 0;
        }}
        [data-testid="stSidebar"] label {{
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }}
        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
            background: #FF0033 !important;
            color: #FFFFFF !important;
            border-radius: 7px !important;
        }}
        [data-testid="stSidebar"] .stCheckbox [data-testid="stMarkdownContainer"] p {{
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] div[data-baseweb="select"],
        [data-testid="stSidebar"] input {{
            background-color: #141414 !important;
            color: #FFFFFF !important;
            border-color: rgba(255, 255, 255, 0.14) !important;
        }}
        [data-testid="stSidebar"] .stSlider div[role="slider"] {{
            background-color: #FF0033 !important;
            border-color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] .stSlider div[data-testid="stTickBar"] {{
            background: rgba(255, 255, 255, 0.18) !important;
        }}
        [data-testid="stSidebar"] button {{
            border-radius: 8px !important;
            border: 1px solid rgba(255, 0, 45, 0.65) !important;
            color: #FFFFFF !important;
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
        .stApp:has(.home-marker) {{
            background:
                radial-gradient(circle at 18% 16%, rgba(255, 0, 51, 0.16), transparent 22%),
                radial-gradient(circle at 80% 18%, rgba(255, 181, 28, 0.12), transparent 20%),
                linear-gradient(135deg, #000000 0%, #0A0A0A 52%, #160006 100%) !important;
        }}
        .block-container:has(.home-marker) {{
            max-width: none !important;
            padding: 0 !important;
        }}
        .public-site {{
            min-height: 100vh;
            width: min(1760px, 92vw);
            margin: 0 auto;
            padding: 28px 0 56px;
        }}
        .public-nav {{
            min-height: 82px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 24px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .public-brand {{
            display: flex;
            align-items: center;
            gap: 14px;
            font-size: 24px;
            font-weight: 900;
        }}
        .public-brand img {{
            width: 48px;
            height: 48px;
            object-fit: contain;
        }}
        .public-nav-right {{
            display: flex;
            align-items: center;
            gap: 28px;
        }}
        .public-nav-copy {{
            color: #C9C9C9 !important;
            font-size: 15px;
        }}
        .public-nav-actions {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .public-nav-actions a {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 112px;
            min-height: 44px;
            padding: 0 22px;
            border-radius: 999px !important;
            text-decoration: none !important;
            font-weight: 850 !important;
            font-size: 15px;
            transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
        }}
        .public-nav-actions a:hover {{
            transform: translateY(-1px);
        }}
        .public-login {{
            background: transparent !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.24) !important;
        }}
        .public-signup {{
            background: #FFB51C !important;
            color: #000000 !important;
            border: 1px solid #FFB51C !important;
            box-shadow: 0 12px 28px rgba(255, 181, 28, 0.20);
        }}
        .public-hero {{
            display: grid;
            grid-template-columns: minmax(0, 0.92fr) minmax(520px, 1.08fr);
            align-items: center;
            gap: min(5vw, 78px);
            min-height: calc(100vh - 166px);
            padding: 58px 0 12px;
        }}
        .public-eyebrow {{
            color: #FFB51C !important;
            font-size: 14px;
            font-weight: 900;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 20px;
        }}
        .public-title {{
            font-size: clamp(52px, 5.4vw, 88px);
            line-height: 1.04;
            font-weight: 900;
            max-width: 780px;
        }}
        .public-title span {{
            color: #FF0033 !important;
        }}
        .public-subtitle {{
            color: #D7D7D7 !important;
            font-size: clamp(18px, 1.6vw, 24px);
            line-height: 1.55;
            max-width: 690px;
            margin-top: 24px;
        }}
        .public-cta-row {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin-top: 32px;
        }}
        .public-cta-row a {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 52px;
            padding: 0 28px;
            border-radius: 999px;
            text-decoration: none !important;
            font-weight: 900;
        }}
        .public-primary {{
            background: #FFB51C;
            color: #000000 !important;
            box-shadow: 0 18px 40px rgba(255, 181, 28, 0.20);
        }}
        .public-secondary {{
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.18);
        }}
        .public-feature-row {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-top: 34px;
            max-width: 760px;
        }}
        .public-feature {{
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 8px;
            padding: 16px;
        }}
        .public-feature strong {{
            color: #FFFFFF !important;
            display: block;
            font-size: 16px;
            margin-bottom: 6px;
        }}
        .public-feature span {{
            color: #BDBDBD !important;
            font-size: 13px;
            line-height: 1.4;
        }}
        .public-visual {{
            min-height: min(66vh, 660px);
            border-radius: 28px;
            background-size: cover;
            background-position: left center;
            border: 1px solid rgba(255, 255, 255, 0.12);
            position: relative;
            overflow: hidden;
            box-shadow: 0 28px 95px rgba(0, 0, 0, 0.72), inset 0 0 120px rgba(0, 0, 0, 0.45);
        }}
        .public-visual::after {{
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(0, 0, 0, 0.18), rgba(0, 0, 0, 0.50)),
                radial-gradient(circle at 88% 16%, rgba(255, 181, 28, 0.16), transparent 24%);
        }}
        .public-stat-strip {{
            position: absolute;
            left: 28px;
            right: 28px;
            bottom: 28px;
            z-index: 2;
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }}
        .public-stat {{
            background: rgba(0, 0, 0, 0.62);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            padding: 14px;
            backdrop-filter: blur(14px);
        }}
        .public-stat strong {{
            display: block;
            color: #FFFFFF !important;
            font-size: 22px;
            margin-bottom: 4px;
        }}
        .public-stat span {{
            color: #CFCFCF !important;
            font-size: 12px;
        }}
        @media (max-width: 980px) {{
            .public-site {{
                width: min(92vw, 680px);
                padding-top: 18px;
            }}
            .public-nav {{
                align-items: center;
                min-height: 74px;
            }}
            .public-nav-copy {{
                display: none;
            }}
            .public-nav-actions a {{
                min-width: auto;
                min-height: 38px;
                padding: 0 14px;
            }}
            .public-hero {{
                grid-template-columns: 1fr;
                min-height: 0;
                padding-top: 34px;
            }}
            .public-feature-row {{
                grid-template-columns: 1fr;
            }}
            .public-cta-row {{
                flex-direction: column;
                align-items: stretch;
            }}
            .public-visual {{
                min-height: 460px;
            }}
            .public-stat-strip {{
                grid-template-columns: 1fr;
            }}
        }}
        .stApp:has(.auth-marker) {{
            background:
                radial-gradient(circle at 16% 12%, rgba(255, 255, 255, 0.07), transparent 20%),
                radial-gradient(circle at 76% 82%, rgba(255, 181, 28, 0.10), transparent 24%),
                linear-gradient(135deg, #272727 0%, #101010 46%, #030303 100%) !important;
        }}
        .block-container:has(.auth-marker) {{
            max-width: none !important;
            padding: clamp(22px, 4vh, 54px) 0 !important;
        }}
        .block-container:has(.auth-marker) [data-testid="stHorizontalBlock"] {{
            width: min(1680px, 92vw);
            min-height: clamp(680px, 82vh, 860px);
            margin: 0 auto;
            padding: 18px;
            background: #000000;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 38px;
            box-shadow: 0 34px 100px rgba(0, 0, 0, 0.78);
            gap: clamp(24px, 3vw, 58px) !important;
            align-items: stretch;
            overflow: hidden;
        }}
        .block-container:has(.auth-marker) [data-testid="column"] {{
            min-height: clamp(644px, 78vh, 824px);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .block-container:has(.auth-marker) [data-testid="column"]:first-child > div {{
            height: 100%;
        }}
        .block-container:has(.auth-marker) [data-testid="column"]:nth-child(2) > div {{
            width: min(520px, 90%);
            margin: auto;
        }}
        .auth-visual {{
            min-height: clamp(644px, 78vh, 824px);
            height: 100%;
            border-radius: 28px;
            position: relative;
            overflow: hidden;
            background: #080808;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: inset -90px 0 150px rgba(0, 0, 0, 0.22);
        }}
        .auth-visual img {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: left center;
            opacity: 0.92;
        }}
        .auth-visual::after {{
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(0,0,0,0.12), rgba(0,0,0,0.55)),
                radial-gradient(circle at 16% 18%, rgba(0, 255, 255, 0.16), transparent 26%);
            pointer-events: none;
        }}
        .auth-visual-copy {{
            position: absolute;
            left: clamp(26px, 4vw, 62px);
            bottom: clamp(28px, 5vw, 72px);
            z-index: 2;
            max-width: 520px;
        }}
        .auth-visual-copy span {{
            display: inline-flex;
            color: #FFB51C !important;
            font-size: 13px;
            font-weight: 900;
            text-transform: uppercase;
            margin-bottom: 16px;
        }}
        .auth-visual-copy strong {{
            display: block;
            color: #FFFFFF !important;
            font-size: clamp(38px, 4vw, 58px);
            line-height: 1.08;
            letter-spacing: 0;
        }}
        .auth-heading-wrap {{
            text-align: center;
            margin-bottom: 34px;
        }}
        .auth-title {{
            text-align: center;
            color: #F4F4F4 !important;
            font-size: clamp(38px, 4vw, 58px);
            font-weight: 850;
            line-height: 1.05;
            margin-bottom: 16px;
        }}
        .auth-subtitle {{
            color: #D9D9D9 !important;
            text-align: center;
            font-size: clamp(16px, 1.15vw, 20px);
            line-height: 1.45;
        }}
        .auth-form-card {{
            background: rgba(255, 255, 255, 0.025);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 28px;
            box-shadow: 0 28px 70px rgba(0, 0, 0, 0.42);
        }}
        .block-container:has(.auth-marker) [data-testid="stForm"] {{
            background: rgba(255, 255, 255, 0.025) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 18px !important;
            padding: 28px !important;
            box-shadow: 0 28px 70px rgba(0, 0, 0, 0.42) !important;
        }}
        .block-container:has(.auth-marker) [data-testid="stForm"] [data-testid="stVerticalBlock"] {{
            gap: 0.72rem !important;
        }}
        .block-container:has(.auth-marker) [data-testid="stForm"] label p,
        .block-container:has(.auth-marker) [data-testid="stTextInput"] label p {{
            color: #EDEDED !important;
            font-size: 14px !important;
            font-weight: 850 !important;
            margin-bottom: 7px !important;
        }}
        .block-container:has(.auth-marker) [data-testid="stTextInput"] input {{
            background: #1C1C1C !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            min-height: 54px;
            font-size: 17px !important;
            padding: 0 18px !important;
            box-shadow: none !important;
        }}
        .block-container:has(.auth-marker) [data-testid="stTextInput"] input:focus {{
            border: 1px solid rgba(255, 181, 28, 0.62) !important;
            box-shadow: 0 0 0 2px rgba(255, 181, 28, 0.16) !important;
        }}
        .block-container:has(.auth-marker) [data-testid="stFormSubmitButton"] button {{
            background: #FFB51C !important;
            color: #000000 !important;
            border: 1px solid #FFB51C !important;
            border-radius: 12px !important;
            min-height: 56px;
            margin-top: 14px;
            font-size: 16px !important;
            font-weight: 900 !important;
            box-shadow: 0 16px 32px rgba(255, 181, 28, 0.22) !important;
        }}
        .block-container:has(.auth-marker) [data-testid="stCheckbox"] label,
        .block-container:has(.auth-marker) [data-testid="stCheckbox"] p {{
            color: #EDEDED !important;
            font-size: 15px !important;
        }}
        .auth-switch {{
            text-align: center;
            color: #D6D6D6 !important;
            margin-top: 26px;
            font-size: 16px;
        }}
        .auth-switch span {{
            color: #FFB51C;
            font-weight: 800;
        }}
        .block-container:has(.auth-marker) [data-testid="stButton"] button {{
            background: transparent !important;
            border: 0 !important;
            color: #FFB51C !important;
            box-shadow: none !important;
            font-size: 16px !important;
            font-weight: 850 !important;
            min-height: 28px;
            padding: 0 !important;
        }}
        @media (max-width: 920px) {{
            .block-container:has(.auth-marker) {{
                padding: 18px 0 !important;
            }}
            .block-container:has(.auth-marker) [data-testid="stHorizontalBlock"] {{
                width: min(94vw, 620px);
                min-height: 0;
                padding: 10px;
                border-radius: 28px;
            }}
            .block-container:has(.auth-marker) [data-testid="column"] {{
                min-height: auto;
            }}
            .block-container:has(.auth-marker) [data-testid="column"]:nth-child(2) > div {{
                width: 92%;
                padding: 28px 0 18px;
            }}
            .auth-visual {{
                min-height: 360px;
                border-radius: 22px;
            }}
            .auth-visual-copy strong {{
                font-size: 34px;
            }}
            .auth-form-card {{
                padding: 20px;
            }}
            .auth-heading-wrap {{
                margin-bottom: 24px;
            }}
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {CHART_COLORS["border"]};
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_auth_state() -> None:
    """Initialize Streamlit authentication session state."""
    st.session_state.setdefault("auth_mode", "login")
    st.session_state.setdefault("auth_view", "home")
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("user", None)


def switch_auth_mode(mode: str) -> None:
    """Switch between login and signup modes."""
    st.session_state.auth_mode = mode
    st.session_state.auth_view = "auth"


def show_auth_page(mode: str) -> None:
    """Open the authentication page in a chosen mode."""
    st.session_state.auth_mode = mode
    st.session_state.auth_view = "auth"


def show_public_home() -> None:
    """Return unauthenticated users to the public home page."""
    st.session_state.auth_view = "home"
    st.session_state.auth_mode = "login"
    st.query_params.clear()


def sync_auth_view_from_url() -> None:
    """Open authentication view when the public website URL asks for it."""
    requested_auth = st.query_params.get("auth")
    if (
        not st.session_state.authenticated
        and st.session_state.auth_view == "home"
        and requested_auth in {"login", "signup"}
    ):
        st.session_state.auth_mode = requested_auth
        st.session_state.auth_view = "auth"


def logout() -> None:
    """Clear the authenticated session."""
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.auth_mode = "login"
    st.session_state.auth_view = "home"
    st.query_params.clear()
    st.rerun()


def render_public_home() -> None:
    """Render the public website home screen before authentication."""
    image_uri = image_to_data_uri(HOME_IMAGE_PATH) if HOME_IMAGE_PATH.exists() else ""
    logo_uri = image_to_data_uri(LOGO_PATH) if LOGO_PATH.exists() else ""

    st.markdown(
        f"""
        <div class="home-marker"></div>
        <div class="public-site">
            <nav class="public-nav">
                <div class="public-brand">
                    {'<img src="' + logo_uri + '" alt="' + BRAND_NAME + ' logo" />' if logo_uri else ''}
                    <span>{BRAND_NAME}</span>
                </div>
                <div class="public-nav-right">
                    <div class="public-nav-copy">Modern NSE portfolio intelligence dashboard</div>
                    <div class="public-nav-actions">
                        <a class="public-login" href="?auth=login" target="_self">Login</a>
                        <a class="public-signup" href="?auth=signup" target="_self">Sign up</a>
                    </div>
                </div>
            </nav>
            <section class="public-hero">
                <div>
                    <div class="public-eyebrow">NSE portfolio optimizer</div>
                    <div class="public-title">
                        Build smarter portfolios with <span>{BRAND_NAME}</span>
                    </div>
                    <div class="public-subtitle">
                        Analyze Indian NSE stocks using Modern Portfolio Theory, Efficient Frontier
                        optimization, backtesting, risk analytics, and research-friendly dashboard views.
                    </div>
                    <div class="public-cta-row">
                        <a class="public-primary" href="?auth=signup" target="_self">Start free</a>
                        <a class="public-secondary" href="?auth=login" target="_self">Login to dashboard</a>
                    </div>
                    <div class="public-feature-row">
                        <div class="public-feature">
                            <strong>Optimize</strong>
                            <span>Maximum Sharpe and minimum volatility allocation workflows.</span>
                        </div>
                        <div class="public-feature">
                            <strong>Measure Risk</strong>
                            <span>Volatility, drawdown, VaR, CVaR, beta, and tracking metrics.</span>
                        </div>
                        <div class="public-feature">
                            <strong>Present</strong>
                            <span>Clean Streamlit dashboard for GitHub, college demos, and reports.</span>
                        </div>
                    </div>
                </div>
                <div class="public-visual" style="background-image:
                    linear-gradient(180deg, rgba(0,0,0,0.08), rgba(0,0,0,0.45)),
                    url('{image_uri}');">
                    <div class="public-stat-strip">
                        <div class="public-stat"><strong>50</strong><span>NSE stocks</span></div>
                        <div class="public-stat"><strong>MPT</strong><span>Optimization model</span></div>
                        <div class="public-stat"><strong>Risk</strong><span>Advanced analytics</span></div>
                    </div>
                </div>
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auth_page() -> None:
    """Render MongoDB-backed login and signup page."""
    initialize_auth_state()
    mode = st.session_state.auth_mode
    is_signup = mode == "signup"
    image_uri = image_to_data_uri(AUTH_IMAGE_PATH) if AUTH_IMAGE_PATH.exists() else ""

    st.markdown('<div class="auth-marker"></div>', unsafe_allow_html=True)
    visual_col, form_col = st.columns([1.0, 1.0], gap="large")

    with visual_col:
        st.markdown(
            f"""
            <div class="auth-visual" role="img" aria-label="{BRAND_NAME} authentication visual">
                <img src="{image_uri}" alt="{BRAND_NAME} authentication page" />
                <div class="auth-visual-copy">
                    <span>NSE intelligence workspace</span>
                    <strong>Build disciplined portfolios with {BRAND_NAME}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with form_col:
        st.markdown(
            f"""
            <div class="auth-heading-wrap">
            <div class="auth-title">{'Create account' if is_signup else 'Login'}</div>
            <div class="auth-subtitle">
                {'Start your portfolio intelligence workspace' if is_signup else 'Enter your credentials to access your account'}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("signup_form" if is_signup else "login_form", clear_on_submit=False):
            name = ""
            if is_signup:
                name = st.text_input("Full name", placeholder="Shaleen Singh")
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="Minimum 8 characters")
            remember = st.checkbox("Remember me", value=True)
            submitted = st.form_submit_button(
                "Create account" if is_signup else "Login",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            with st.spinner("Checking credentials..."):
                result = create_user(name, email, password) if is_signup else authenticate_user(email, password)
            if result.success:
                st.session_state.authenticated = True
                st.session_state.user = result.user
                st.session_state.remember_me = remember
                st.success(result.message)
                st.rerun()
            else:
                st.error(result.message)

        if is_signup:
            st.markdown('<div class="auth-switch">Already a member?</div>', unsafe_allow_html=True)
            st.button("Login instead", use_container_width=True, on_click=switch_auth_mode, args=("login",))
        else:
            st.markdown('<div class="auth-switch">Not a member? <span>Create an account</span></div>', unsafe_allow_html=True)
            st.button("Create an account", use_container_width=True, on_click=switch_auth_mode, args=("signup",))


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
    user = st.session_state.get("user") or {}
    st.sidebar.markdown(f"## {BRAND_NAME}")
    st.sidebar.markdown(
        f"""
        <div class="info-card">
            <strong>{user.get('name') or 'NiveshMatrix User'}</strong>
            <div class="small-muted">{user.get('email') or 'Authenticated session'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Logout", use_container_width=True):
        logout()

    st.sidebar.markdown("### Portfolio Controls")

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
    initialize_auth_state()
    sync_auth_view_from_url()

    if not st.session_state.authenticated and st.session_state.auth_view == "auth":
        render_auth_page()
        st.stop()
    if not st.session_state.authenticated:
        render_public_home()
        st.stop()

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
