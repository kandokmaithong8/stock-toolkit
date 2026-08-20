"""
streamlit_app.py
Interactive dashboard for the stock toolkit. Designed to run on Streamlit
Community Cloud (free tier: ~1 CPU, ~1GB RAM) with no local machine
involved — see README.md for the GitHub + Streamlit Cloud deployment guide.

Three tabs:
  1. Portfolio Analyzer — return/risk metrics + mean-variance optimization
  2. Forecast Demo — train a small LSTM/GRU live, in-browser (capped small
     for cloud compute limits; use the CLI/GitHub Actions for serious runs)
  3. Live Predictions — reads predictions_log.csv (kept up to date by a
     scheduled GitHub Action) and shows the model's real track record
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Bridge Streamlit Cloud secrets -> environment variables, so the existing
# env-var-based settrade_source.py works unmodified.
# --------------------------------------------------------------------------- #
for _key in ("SETTRADE_APP_ID", "SETTRADE_APP_SECRET", "SETTRADE_BROKER_ID", "SETTRADE_APP_CODE"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

from data_utils import fetch_multi_close
from portfolio import PortfolioConfig, analyze_portfolio, compute_returns, optimize_weights
from forecasting import ForecastConfig, build_dataset, train_model, evaluate

st.set_page_config(page_title="Stock Toolkit", layout="wide")
st.title("📈 Stock Toolkit")
st.caption(
    "Educational research tool for the methods surveyed in the DL-for-finance "
    "literature — not investment advice. See README.md for full context."
)

tab_portfolio, tab_forecast, tab_log = st.tabs(
    ["Portfolio Analyzer", "Forecast Demo", "Live Predictions"]
)

# --------------------------------------------------------------------------- #
# Tab 1: Portfolio Analyzer
# --------------------------------------------------------------------------- #
with tab_portfolio:
    st.subheader("Portfolio return, risk, and allocation")

    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_input = st.text_input(
            "Tickers (space or comma separated)", value="AAPL MSFT GOOGL BND",
            help="Use bare SET symbols (e.g. PTT KBANK) with the settrade source, "
                 "or Yahoo symbols (e.g. AAPL, PTT.BK) with the yahoo source."
        )
    with col2:
        source = st.selectbox("Data source", ["yahoo", "settrade"], key="pf_source")

    tickers = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()]

    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input("Start date", value=pd.Timestamp("2018-01-01")).strftime("%Y-%m-%d")
    with c2:
        risk_free = st.number_input("Risk-free rate (annual)", value=0.02, step=0.005, format="%.3f")
    with c3:
        optimize_choice = st.selectbox("Optimize for", ["(none — use equal weight)", "max_sharpe", "min_volatility"])

    run_pf = st.button("Analyze portfolio", type="primary")

    if run_pf and tickers:
        try:
            with st.spinner("Fetching data and computing metrics..."):
                cfg = PortfolioConfig(tickers=tickers, start=start, source=source, risk_free_rate=risk_free)
                result = analyze_portfolio(cfg)

                opt_result = None
                if optimize_choice != "(none — use equal weight)":
                    prices = fetch_multi_close(tickers, start, source=source)
                    returns = compute_returns(prices)
                    opt_result = optimize_weights(returns, objective=optimize_choice, risk_free_rate=risk_free)

            m = result["portfolio"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Annual return", f"{m['annual_return_pct']:.1f}%")
            mc2.metric("Annual volatility", f"{m['annual_volatility_pct']:.1f}%")
            mc3.metric("Sharpe ratio", f"{m['sharpe_ratio']:.2f}")
            mc4.metric("Max drawdown", f"{m['max_drawdown_pct']:.1f}%")

            st.markdown("**Per-asset breakdown**")
            per_asset_df = pd.DataFrame(result["per_asset"]).T
            st.dataframe(per_asset_df, use_container_width=True)

            st.markdown("**Correlation matrix**")
            corr_df = pd.DataFrame(result["correlation_matrix"])
            fig = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)

            if opt_result:
                st.markdown(f"**Optimized weights ({optimize_choice})**")
                w_df = pd.DataFrame(list(opt_result["weights"].items()), columns=["Ticker", "Weight"])
                fig2 = px.bar(w_df, x="Ticker", y="Weight")
                st.plotly_chart(fig2, use_container_width=True)
                oc1, oc2, oc3 = st.columns(3)
                oc1.metric("Expected annual return", f"{opt_result['expected_annual_return_pct']:.1f}%")
                oc2.metric("Expected volatility", f"{opt_result['expected_annual_volatility_pct']:.1f}%")
                oc3.metric("Expected Sharpe", f"{opt_result['expected_sharpe']:.2f}")

        except Exception as e:
            st.error(f"Couldn't complete the analysis: {e}")

# --------------------------------------------------------------------------- #
# Tab 2: Forecast Demo
# --------------------------------------------------------------------------- #
with tab_forecast:
    st.subheader("Train a small LSTM/GRU live, in-browser")
    st.caption(
        "Kept small (short lookback, few epochs) to stay responsive on free "
        "cloud compute. For a real evaluation, run forecasting.py --walk-forward "
        "locally, in Colab, or via GitHub Actions with a larger budget."
    )

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        fc_ticker = st.text_input("Ticker", value="AAPL")
        fc_source = st.selectbox("Data source", ["yahoo", "settrade"], key="fc_source")
    with fc2:
        fc_start = st.date_input(
            "History start",
            value=pd.Timestamp.today() - pd.Timedelta(days=730),
            max_value=pd.Timestamp.today() - pd.Timedelta(days=200),
            help="Start of the TRAINING history, not a forecast date. Needs "
                 "roughly 1-2+ years so there's enough data after technical "
                 "indicators warm up and after the train/val/test split.",
        ).strftime("%Y-%m-%d")
        fc_model = st.selectbox("Model", ["lstm", "gru"])
    with fc3:
        fc_epochs = st.slider("Epochs", min_value=5, max_value=60, value=25,
                               help="Capped at 60 to keep cloud runs fast")
        fc_lookback = st.slider("Lookback (days)", min_value=10, max_value=60, value=30)

    run_fc = st.button("Train & forecast", type="primary")

    if run_fc and fc_ticker:
        try:
            with st.spinner(f"Training {fc_model.upper()} on {fc_ticker}... this can take a minute on free compute"):
                cfg = ForecastConfig(
                    ticker=fc_ticker.strip().upper(), start=fc_start, source=fc_source,
                    model_type=fc_model, epochs=fc_epochs, lookback=fc_lookback,
                )
                data = build_dataset(cfg)
                model, _ = train_model(cfg, data)
                results = evaluate(model, data)

            beats = results["beats_naive_baseline"]
            st.markdown(
                f"{'✅' if beats else '⚠️'} **This model {'beat' if beats else 'did NOT beat'} "
                f"the naive baseline** (RMSE {results['rmse']:.3f} vs. naive {results['naive_baseline_rmse']:.3f})"
                + ("" if beats else " — treat any apparent accuracy with real skepticism.")
            )

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("RMSE", f"{results['rmse']:.3f}")
            rc2.metric("MAPE", f"{results['mape_pct']:.2f}%")
            rc3.metric("Directional accuracy", f"{results['directional_accuracy_pct']:.1f}%")

            plot_df = pd.DataFrame({
                "Date": pd.to_datetime(results["dates"]),
                "Actual": results["actual"],
                "Predicted": results["predicted"],
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Actual"], name="Actual", mode="lines"))
            fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Predicted"], name="Predicted", mode="lines"))
            fig.update_layout(title=f"{fc_ticker.upper()} — predicted vs. actual (test period)",
                               xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Training failed: {e}")

# --------------------------------------------------------------------------- #
# Tab 3: Live Predictions Log
# --------------------------------------------------------------------------- #
with tab_log:
    st.subheader("Live out-of-sample prediction track record")
    st.caption(
        "Updated daily by a scheduled GitHub Action running predict_logger.py — "
        "every prediction here was logged BEFORE the outcome was known."
    )

    log_path = os.path.join(os.path.dirname(__file__), "predictions_log.csv")

    if not os.path.exists(log_path):
        st.info(
            "No predictions_log.csv yet. It's created automatically once the "
            "GitHub Action in .github/workflows/daily_predictions.yml has run "
            "at least once — see README.md for setup, or trigger it manually "
            "from the Actions tab in your repo."
        )
    else:
        log_df = pd.read_csv(log_path)
        if log_df.empty:
            st.info("Log file exists but is empty — waiting on the first scheduled run.")
        else:
            resolved = log_df.dropna(subset=["actual_price"])

            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("Total predictions logged", len(log_df))
            lc2.metric("Resolved (outcome known)", len(resolved))
            if len(resolved) > 0:
                hit_rate = resolved["direction_correct"].astype(bool).mean() * 100
                lc3.metric("Directional hit rate", f"{hit_rate:.1f}%")
            else:
                lc3.metric("Directional hit rate", "—")

            ticker_filter = st.multiselect(
                "Filter by ticker", options=sorted(log_df["ticker"].unique()),
                default=sorted(log_df["ticker"].unique())
            )
            filtered = log_df[log_df["ticker"].isin(ticker_filter)] if ticker_filter else log_df

            if not filtered.dropna(subset=["actual_price"]).empty:
                plot_df = filtered.dropna(subset=["actual_price"]).copy()
                plot_df["target_date"] = pd.to_datetime(plot_df["target_date"])
                fig = px.scatter(
                    plot_df, x="target_date", y="abs_pct_error", color="ticker",
                    title="Prediction error over time (lower is better)",
                    labels={"abs_pct_error": "Absolute % error", "target_date": "Target date"},
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Full log**")
            display_df = filtered.sort_values("run_date", ascending=False).copy()
            if "plausible" in display_df.columns:
                display_df["plausible"] = display_df["plausible"].map(
                    {True: "✅", False: "⚠️ flagged"}
                ).fillna("—")
            st.dataframe(display_df, use_container_width=True)
