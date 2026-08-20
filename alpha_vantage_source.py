"""
alpha_vantage_source.py
Data source adapter: Alpha Vantage (https://www.alphavantage.co) — a free,
official (not scraped) stock data API. 20+ years of daily history on the
free tier, but only 25 requests/day (5/minute), so it's best for a handful
of tickers checked once a day rather than frequent/bulk pulls.

SETUP
-----
1. Get a free key at https://www.alphavantage.co/support/#api-key (just an
   email address, no credit card).
2. Set it as an environment variable:
     export ALPHAVANTAGE_API_KEY="your-key"

NOTE ON TESTING
---------------
This adapter follows Alpha Vantage's documented TIME_SERIES_DAILY endpoint
and response shape exactly (verified against their official docs), but
couldn't be exercised against a live API call in the environment this was
written in (outbound network here is restricted to a small allowlist that
doesn't include alphavantage.co). The JSON parsing was tested against a
synthetic response matching their documented schema. If Alpha Vantage ever
changes their response format, you'll get a clear KeyError pointing at
what's missing — validate with your own key before relying on this.
"""

from __future__ import annotations

import os

import pandas as pd
import requests

BASE_URL = "https://www.alphavantage.co/query"


def fetch_ohlcv(ticker: str, start: str, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a ticker from Alpha Vantage.

    Only daily ("1d") is supported by this adapter — Alpha Vantage's
    intraday endpoints use a different function/response shape.

    Returns a DataFrame indexed by date with columns: Open, High, Low,
    Close, Volume — matching data_utils.fetch_ohlcv's shape.
    """
    if interval not in ("1d", "1day", "day"):
        raise ValueError(f"alpha_vantage_source only supports daily data, got interval={interval!r}")

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing ALPHAVANTAGE_API_KEY. Get a free key at "
            "https://www.alphavantage.co/support/#api-key and set it as an "
            "environment variable."
        )

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "full",   # 20+ years, vs. "compact" (last 100 points)
        "apikey": api_key,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if "Note" in payload:
        raise RuntimeError(f"Alpha Vantage rate limit hit: {payload['Note']}")
    if "Information" in payload:
        raise RuntimeError(f"Alpha Vantage: {payload['Information']}")
    if "Error Message" in payload:
        raise ValueError(f"Alpha Vantage error for {ticker}: {payload['Error Message']}")

    series_key = "Time Series (Daily)"
    if series_key not in payload:
        raise KeyError(
            f"Unexpected Alpha Vantage response shape — missing '{series_key}'. "
            f"Got keys: {list(payload.keys())}"
        )

    raw = payload[series_key]
    df = pd.DataFrame.from_dict(raw, orient="index")
    df = df.rename(columns={
        "1. open": "Open", "2. high": "High", "3. low": "Low",
        "4. close": "Close", "5. volume": "Volume",
    })
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.astype(float).sort_index()

    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    if df.empty:
        raise ValueError(f"No Alpha Vantage data for {ticker} in the requested date range.")

    return df[["Open", "High", "Low", "Close", "Volume"]]
