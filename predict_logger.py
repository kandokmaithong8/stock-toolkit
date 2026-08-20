"""
predict_logger.py
Maintains predictions_log.csv: a running, honest track record of live
out-of-sample forecasts vs. what actually happened.

Designed to run on a schedule via GitHub Actions (see
.github/workflows/daily_predictions.yml) so it works with zero local
machine — GitHub's servers run it, and it commits the updated CSV back to
the repo, which the Streamlit dashboard reads.

Each run does two things:
1. BACKFILL: for previously logged predictions whose target_date has now
   passed and whose actual_price is still blank, fetch the real close price
   and fill in actual_price / error / abs_pct_error / direction_correct.
2. LOG: generate a fresh prediction for each configured ticker and append
   it as a new row (with actual_price left blank, to be backfilled later).

Usage:
    python predict_logger.py --tickers AAPL MSFT --source yahoo
    python predict_logger.py --tickers PTT KBANK --source settrade
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from data_utils import fetch_ohlcv
from forecasting import ForecastConfig, predict_next

LOG_PATH = Path(__file__).parent / "predictions_log.csv"
LOG_COLUMNS = [
    "run_date", "ticker", "source", "model_type", "horizon",
    "as_of_date", "target_date", "last_close", "predicted_price",
    "predicted_change_pct", "implied_move_in_std_devs", "plausible",
    "actual_price", "error", "abs_pct_error", "direction_correct",
]


def _load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def _save_log(df: pd.DataFrame) -> None:
    df.to_csv(LOG_PATH, index=False)


def backfill_actuals(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in actual_price for past predictions whose target date has arrived."""
    if df.empty:
        return df

    # Loaded-from-CSV numeric columns default to float64, which can't hold a
    # bool (direction_correct) without a dtype error — widen them first.
    for col in ("actual_price", "error", "abs_pct_error", "direction_correct"):
        if col in df.columns:
            df[col] = df[col].astype(object)

    today = pd.Timestamp.today().normalize()
    pending = df[df["actual_price"].isna() & (pd.to_datetime(df["target_date"]) <= today)]

    for idx, row in pending.iterrows():
        target_date = pd.to_datetime(row["target_date"])
        try:
            # Small forward window in case target_date landed on a holiday —
            # take the first available close on/after target_date.
            window = fetch_ohlcv(
                row["ticker"],
                start=(target_date - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                end=(target_date + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
                source=row["source"],
            )
            available = window[window.index >= target_date]
            if available.empty:
                continue  # target date still hasn't traded yet (e.g. long holiday)
            actual_price = float(available["Close"].iloc[0])
        except Exception as e:
            print(f"  Could not backfill {row['ticker']} @ {row['target_date']}: {e}")
            continue

        predicted = row["predicted_price"]
        last_close = row["last_close"]
        error = predicted - actual_price
        abs_pct_error = abs(error) / actual_price * 100
        predicted_dir = predicted > last_close
        actual_dir = actual_price > last_close

        df.loc[idx, "actual_price"] = actual_price
        df.loc[idx, "error"] = error
        df.loc[idx, "abs_pct_error"] = abs_pct_error
        df.loc[idx, "direction_correct"] = bool(predicted_dir == actual_dir)
        print(f"  Backfilled {row['ticker']} target {row['target_date']}: "
              f"predicted {predicted:.2f}, actual {actual_price:.2f}, "
              f"error {abs_pct_error:.2f}%")

    return df


def log_new_predictions(df: pd.DataFrame, tickers: list[str], source: str,
                         model_type: str, horizon: int, lookback: int,
                         start: str, epochs: int,
                         hidden_size: int = 64, num_layers: int = 2,
                         dropout: float = 0.2) -> pd.DataFrame:
    run_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    new_rows = []

    for ticker in tickers:
        # Skip if we already logged a prediction for this ticker today
        already_done = ((df["ticker"] == ticker) & (df["run_date"] == run_date)).any()
        if already_done:
            print(f"  Skipping {ticker} — already logged today.")
            continue

        try:
            cfg = ForecastConfig(
                ticker=ticker, start=start, source=source,
                horizon=horizon, lookback=lookback,
                model_type=model_type, epochs=epochs,
                hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
            )
            pred = predict_next(cfg)
        except Exception as e:
            print(f"  Failed to predict {ticker}: {e}")
            continue

        new_rows.append({
            "run_date": run_date,
            "ticker": pred["ticker"],
            "source": pred["source"],
            "model_type": pred["model_type"],
            "horizon": pred["horizon"],
            "as_of_date": pred["as_of_date"],
            "target_date": pred["target_date"],
            "last_close": pred["last_close"],
            "predicted_price": pred["predicted_price"],
            "predicted_change_pct": pred["predicted_change_pct"],
            "implied_move_in_std_devs": pred["implied_move_in_std_devs"],
            "plausible": pred["plausible"],
            "actual_price": None,
            "error": None,
            "abs_pct_error": None,
            "direction_correct": None,
        })
        flag = "" if pred["plausible"] else "  ⚠️ FLAGGED AS IMPLAUSIBLE"
        print(f"  Logged {ticker}: as-of {pred['as_of_date']} close {pred['last_close']:.2f} "
              f"-> predicted {pred['predicted_price']:.2f} on {pred['target_date']} "
              f"({pred['predicted_change_pct']:+.2f}%){flag}")

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


def main():
    p = argparse.ArgumentParser(description="Log live out-of-sample predictions and backfill outcomes")
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--source", choices=["yahoo", "settrade", "alpha_vantage", "twelve_data", "finnhub"], default="yahoo")
    p.add_argument("--model-type", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--lookback", type=int, default=60,
                    help="Overridden by --hidden-size etc. if you pass tuned values from forecasting.py --tune")
    p.add_argument("--start", default="2018-01-01", help="Training history start date")
    p.add_argument("--epochs", type=int, default=50,
                    help="Kept modest by default so this finishes quickly in CI")
    p.add_argument("--hidden-size", type=int, default=64,
                    help="LSTM/GRU hidden size — pass the value from forecasting.py --tune for a tuned model")
    p.add_argument("--num-layers", type=int, default=2,
                    help="LSTM/GRU layer count — pass the value from forecasting.py --tune for a tuned model")
    p.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout rate — pass the value from forecasting.py --tune for a tuned model")
    args = p.parse_args()

    df = _load_log()

    print("Backfilling actual outcomes for past predictions...")
    df = backfill_actuals(df)

    print("Generating today's predictions...")
    df = log_new_predictions(
        df, tickers=args.tickers, source=args.source,
        model_type=args.model_type, horizon=args.horizon,
        lookback=args.lookback, start=args.start, epochs=args.epochs,
        hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout,
    )

    _save_log(df)
    print(f"\nLog saved to {LOG_PATH} ({len(df)} total rows).")


if __name__ == "__main__":
    main()
