"""
data_utils.py
Shared data-fetching and feature-engineering helpers for the forecasting
and portfolio modules.

Two data sources are supported, selected via `source=`:
  - "yahoo"    (default) — yfinance, free/unofficial, global tickers.
  - "settrade" — official Settrade Open API for SET-listed Thai equities.
                 See settrade_source.py for credential setup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

VALID_SOURCES = ("yahoo", "settrade")


def _check_source(source: str) -> None:
    if source not in VALID_SOURCES:
        raise ValueError(f"Unknown source '{source}'. Must be one of {VALID_SOURCES}.")


def fetch_ohlcv(ticker: str, start: str, end: str | None = None,
                 interval: str = "1d", source: str = "yahoo") -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker.

    source="yahoo":    ticker is a Yahoo symbol (e.g. "AAPL", "PTT.BK").
    source="settrade": ticker is a bare SET symbol (e.g. "PTT", "KBANK") —
                        requires Settrade Open API credentials, see
                        settrade_source.py.

    Returns a DataFrame indexed by date with columns:
    Open, High, Low, Close, Volume
    """
    _check_source(source)

    if source == "settrade":
        import settrade_source
        return settrade_source.fetch_ohlcv(ticker, start, end, interval)

    df = yf.download(ticker, start=start, end=end, interval=interval,
                      auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol and date range.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    return df


def fetch_multi_close(tickers: list[str], start: str, end: str | None = None,
                       source: str = "yahoo") -> pd.DataFrame:
    """
    Download close prices for multiple tickers and align them into a
    single DataFrame (columns = tickers).
    """
    _check_source(source)

    if source == "settrade":
        import settrade_source
        series = {}
        for t in tickers:
            df = settrade_source.fetch_ohlcv(t, start, end, interval="1d")
            series[t] = df["Close"]
        close = pd.DataFrame(series).dropna(how="all").ffill().dropna()
        if close.empty:
            raise ValueError("No overlapping price data found for the given tickers/date range.")
        return close

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        # single ticker fallback
        close = raw[["Close"]]
        close.columns = tickers
    close = close.dropna(how="all").ffill().dropna()
    if close.empty:
        raise ValueError("No overlapping price data found for the given tickers/date range.")
    return close


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a standard set of technical indicators to an OHLCV DataFrame.
    All indicators are computed causally (no look-ahead) using only
    past/current data at each row.

    Both raw (price-level) and relative (scale-invariant) versions of the
    price-based indicators are included. forecasting.py's FEATURE_COLUMNS
    deliberately uses only the relative versions (return_1d, close_to_sma_*,
    bb_position, macd_*_norm, etc.) as model inputs — a StandardScaler fit
    on raw price levels (Close, SMA, EMA, MACD, Bollinger Bands) breaks down
    the moment a stock's price has drifted meaningfully since the training
    period, because "today" then sits outside the range the scaler was
    fit on and the model extrapolates wildly. Relative features stay in a
    roughly stable range regardless of the stock's absolute price level or
    how much it's grown, so they don't have this failure mode. The raw
    columns are kept here for anyone who wants them (e.g. plotting), just
    not fed to the model.
    """
    out = df.copy()

    out["return_1d"] = out["Close"].pct_change()
    out["log_return_1d"] = np.log(out["Close"]).diff()

    for window in (5, 10, 20, 50):
        out[f"sma_{window}"] = out["Close"].rolling(window).mean()
        out[f"ema_{window}"] = out["Close"].ewm(span=window, adjust=False).mean()
        # Relative versions: how far price sits from the moving average, as
        # a fraction — stays in a stable range (~-0.2..0.2 typically)
        # regardless of the stock's absolute price level.
        out[f"close_to_sma_{window}"] = out["Close"] / out[f"sma_{window}"] - 1
        out[f"close_to_ema_{window}"] = out["Close"] / out[f"ema_{window}"] - 1

    out["volatility_10"] = out["log_return_1d"].rolling(10).std()
    out["volatility_20"] = out["log_return_1d"].rolling(20).std()

    # RSI (14) — already bounded 0-100, scale-invariant by construction
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9) — raw MACD is in price units (an EMA difference), so
    # it scales with the stock's price level just like SMA/EMA. Normalize
    # by Close to get a relative version.
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["macd_norm"] = out["macd"] / out["Close"]
    out["macd_signal_norm"] = out["macd_signal"] / out["Close"]
    out["macd_hist_norm"] = out["macd_hist"] / out["Close"]

    # Bollinger Bands (20, 2 std)
    mid = out["Close"].rolling(20).mean()
    std = out["Close"].rolling(20).std()
    out["bb_upper"] = mid + 2 * std
    out["bb_lower"] = mid - 2 * std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / mid   # already relative
    # Where Close sits within the bands, as a 0-1 fraction (already relative)
    band_range = (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
    out["bb_position"] = (out["Close"] - out["bb_lower"]) / band_range

    # Volume features
    out["volume_change"] = out["Volume"].pct_change()             # already relative
    out["volume_sma_10"] = out["Volume"].rolling(10).mean()
    out["volume_ratio"] = out["Volume"] / out["volume_sma_10"].replace(0, np.nan) - 1

    out = out.dropna()
    return out
