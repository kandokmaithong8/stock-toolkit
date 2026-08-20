"""
settrade_source.py
Data source adapter: Stock Exchange of Thailand (SET) via the official
Settrade Open API, using the `settrade-v2` SDK (pip install settrade-v2).

WHY THIS EXISTS
----------------
Yahoo Finance (the default source in data_utils.py) is free but unofficial:
Yahoo doesn't publish a supported API, and third-party libraries like
yfinance call internal endpoints that can change or rate-limit without
notice. Settrade is a subsidiary of the Stock Exchange of Thailand itself,
so this is the officially sanctioned way to pull SET market data
programmatically.

ONE-TIME SETUP
--------------
1. Register a sandbox account at https://developer.settrade.com
2. Generate an Application Id / Application Secret from the developer
   portal ("app_id" / "app_secret" below). It's shown only once — save it.
3. Get your `broker_id` and `app_code`. As of this writing, the equity
   market-data/trading API is only available through a handful of partner
   brokers (e.g. Globlex, Yuanta, Country Group, Classic Ausiris) — you'll
   need an account with one of them to get real (non-sandbox) values.
4. Set the following as environment variables (recommended, so credentials
   never end up in code or shell history):

     export SETTRADE_APP_ID="your-app-id"
     export SETTRADE_APP_SECRET="your-app-secret"
     export SETTRADE_BROKER_ID="your-broker-id"
     export SETTRADE_APP_CODE="your-app-code"

WHAT'S VERIFIED VS. NOT
------------------------
The `settrade-v2` package (v2.2.1) was installed and inspected directly to
write this: `Investor.__init__(app_id, app_secret, app_code, broker_id,
is_auto_queue)`, the `Investor.MarketData()` factory, and
`MarketData.get_candlestick(symbol, interval, limit, start, end,
normalized)` all match the SDK's actual source exactly — that part is
solid, not guessed from blog posts.

What could NOT be verified: the exact JSON field names in the response
body, since that requires a live authenticated call against Settrade's
servers (not reachable from the environment this was written in). The
`rename_map` below covers the field names documented in Settrade's own
usage examples (time/open/high/low/close/volume) with a clear error if
your response uses different keys — if you hit that error, print(raw) once
to see the actual field names and extend the map accordingly.
"""

from __future__ import annotations

import os

import pandas as pd

try:
    from settrade_v2 import Investor
except ImportError:
    Investor = None  # surfaced as a clear error only when actually used

_INTERVAL_MAP = {
    "1d": "1d", "1day": "1d", "day": "1d",
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "60m": "60m", "1h": "60m",
    "1wk": "1w", "1w": "1w",
}

_investor_cache = None


def _get_investor():
    """Create (and cache) an authenticated Investor session from env vars."""
    global _investor_cache
    if _investor_cache is not None:
        return _investor_cache

    if Investor is None:
        raise ImportError(
            "settrade-v2 is not installed. Run: pip install settrade-v2"
        )

    creds = {
        "app_id": os.environ.get("SETTRADE_APP_ID"),
        "app_secret": os.environ.get("SETTRADE_APP_SECRET"),
        "broker_id": os.environ.get("SETTRADE_BROKER_ID"),
        "app_code": os.environ.get("SETTRADE_APP_CODE"),
    }
    missing = [k.upper() for k, v in creds.items() if not v]
    if missing:
        raise EnvironmentError(
            "Missing Settrade credentials: "
            + ", ".join(f"SETTRADE_{m}" if not m.startswith("SETTRADE") else m for m in missing)
            + ". See the setup steps in settrade_source.py's module docstring."
        )

    _investor_cache = Investor(
        app_id=creds["app_id"],
        app_secret=creds["app_secret"],
        broker_id=creds["broker_id"],
        app_code=creds["app_code"],
        is_auto_queue=False,
    )
    return _investor_cache


def fetch_ohlcv(ticker: str, start: str, end: str | None = None,
                 interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a SET-listed symbol (e.g. "PTT", "KBANK", "AOT",
    "CPALL" — no ".BK" suffix, that's a Yahoo-ism) via the Settrade Open API.

    Returns a DataFrame indexed by date with columns: Open, High, Low,
    Close, Volume — the same shape as data_utils.fetch_ohlcv(source="yahoo"),
    so both sources are drop-in interchangeable elsewhere in this toolkit.
    """
    investor = _get_investor()
    market_data = investor.MarketData()

    settrade_interval = _INTERVAL_MAP.get(interval, interval)

    raw = market_data.get_candlestick(
        symbol=ticker.upper(),
        interval=settrade_interval,
        start=start,
        end=end,
        normalized=True,
    )

    df = pd.DataFrame(raw)
    if df.empty:
        raise ValueError(
            f"No data returned for {ticker} from Settrade. "
            f"Check the symbol (no '.BK' suffix) and date range."
        )

    # Normalize field names across observed SDK response variants.
    rename_map = {
        "time": "Date", "timestamp": "Date", "date": "Date",
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise KeyError(
            f"Settrade response is missing expected columns {missing_cols}. "
            f"Actual columns: {list(df.columns)}. Your SDK version may use "
            f"different field names — extend rename_map in settrade_source.py."
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]
