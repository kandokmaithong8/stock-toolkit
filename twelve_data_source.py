"""
twelve_data_source.py
Data source adapter: Twelve Data (https://twelvedata.com) — a free,
official stock data API with the most comfortable free-tier headroom for
this toolkit's use case (800 requests/day, 8/minute).

SETUP
-----
1. Get a free key at https://twelvedata.com (email signup, no credit card).
2. Set it as an environment variable:
     export TWELVEDATA_API_KEY="your-key"

NOTE ON TESTING
---------------
This adapter follows Twelve Data's documented /time_series endpoint and
response shape exactly (verified against their official docs), but
couldn't be exercised against a live API call in the environment this was
written in (outbound network here is restricted to a small allowlist that
doesn't include api.twelvedata.com). The JSON parsing was tested against a
synthetic response matching their documented schema. Validate with your own
key before relying on this for anything important.
"""

from __future__ import annotations

import os

import pandas as pd
import requests

BASE_URL = "https://api.twelvedata.com/time_series"

_INTERVAL_MAP = {
    "1d": "1day", "1day": "1day", "day": "1day",
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "60m": "1h", "1h": "1h",
    "1wk": "1week", "1w": "1week",
}


def fetch_ohlcv(ticker: str, start: str, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker from Twelve Data.

    Returns a DataFrame indexed by date with columns: Open, High, Low,
    Close, Volume — matching data_utils.fetch_ohlcv's shape.
    """
    api_key = os.environ.get("TWELVEDATA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing TWELVEDATA_API_KEY. Get a free key at "
            "https://twelvedata.com and set it as an environment variable."
        )

    params = {
        "symbol": ticker,
        "interval": _INTERVAL_MAP.get(interval, interval),
        "outputsize": 5000,   # max allowed per request
        "apikey": api_key,
    }
    if start:
        params["start_date"] = start
    if end:
        params["end_date"] = end

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") == "error":
        raise ValueError(f"Twelve Data error for {ticker}: {payload.get('message', payload)}")
    if "values" not in payload:
        raise KeyError(
            f"Unexpected Twelve Data response shape — missing 'values'. "
            f"Got keys: {list(payload.keys())}"
        )

    df = pd.DataFrame(payload["values"])
    if df.empty:
        raise ValueError(f"No Twelve Data data for {ticker} in the requested date range.")

    df = df.rename(columns={
        "datetime": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()   # Twelve Data returns newest-first

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    return df[["Open", "High", "Low", "Close", "Volume"]]
