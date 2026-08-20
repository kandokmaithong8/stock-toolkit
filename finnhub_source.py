"""
finnhub_source.py
Data source adapter: Finnhub (https://finnhub.io) — an official stock data
API with a generous 60 calls/minute free tier for quotes, fundamentals, and
news.

IMPORTANT LIMITATION — READ BEFORE USING FOR HISTORICAL DAILY DATA
--------------------------------------------------------------------
As of recent Finnhub API changes (documented in multiple 2025 GitHub
issues on finnhubio/Finnhub-API, e.g. #546 "Free plan does not offer US
Stock candles?"), the /stock/candle endpoint — which is what this toolkit
needs for historical OHLCV — now returns a 403 "You don't have access to
this resource" for FREE-TIER US stock requests. Many older tutorials still
describe it as free; that access appears to have been removed since. This
adapter is included for completeness and in case you have a paid Finnhub
plan (or it covers non-US symbols on your account), but for free-tier US
equities, prefer alpha_vantage_source.py or twelve_data_source.py instead.
If you hit the 403 here, that's expected given the above — it's not a bug
in this code.

SETUP
-----
1. Get a free key at https://finnhub.io (email signup).
2. Set it as an environment variable:
     export FINNHUB_API_KEY="your-key"
"""

from __future__ import annotations

import os

import pandas as pd
import requests

BASE_URL = "https://finnhub.io/api/v1/stock/candle"

_RESOLUTION_MAP = {
    "1d": "D", "1day": "D", "day": "D",
    "1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60",
    "1wk": "W", "1w": "W",
}


def fetch_ohlcv(ticker: str, start: str, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker from Finnhub's /stock/candle endpoint.

    Returns a DataFrame indexed by date with columns: Open, High, Low,
    Close, Volume — matching data_utils.fetch_ohlcv's shape.

    Raises a clear error (not a silent empty result) if the free-tier
    restriction described in this module's docstring is hit.
    """
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing FINNHUB_API_KEY. Get a free key at https://finnhub.io "
            "and set it as an environment variable."
        )

    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp()) if end else int(pd.Timestamp.today().timestamp())

    params = {
        "symbol": ticker,
        "resolution": _RESOLUTION_MAP.get(interval, interval),
        "from": start_ts,
        "to": end_ts,
        "token": api_key,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)

    if resp.status_code == 403:
        raise PermissionError(
            f"Finnhub returned 403 for {ticker} — this matches the known "
            f"free-tier restriction on US stock candle data described in "
            f"this module's docstring. Try alpha_vantage_source.py or "
            f"twelve_data_source.py instead, or use a paid Finnhub plan."
        )
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("s") == "no_data":
        raise ValueError(f"Finnhub returned no data for {ticker} in the requested date range.")
    if payload.get("s") != "ok":
        raise ValueError(f"Finnhub error for {ticker}: {payload}")

    df = pd.DataFrame({
        "Date": pd.to_datetime(payload["t"], unit="s"),
        "Open": payload["o"], "High": payload["h"], "Low": payload["l"],
        "Close": payload["c"], "Volume": payload["v"],
    })
    df = df.set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]
