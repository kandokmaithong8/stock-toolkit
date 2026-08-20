"""
forecasting.py
Deep learning (LSTM/GRU) forecaster for stock closing prices, following
the general architecture patterns surveyed in recent DL-for-finance
literature (recurrent models on multivariate OHLCV + technical features).

IMPORTANT — READ BEFORE USING FOR ANYTHING REAL:
This is a research/educational tool, not a trading signal generator.
Backtested accuracy on historical price data is easy to overstate due to
overfitting, regime change, and non-stationarity. Nothing here should be
used as the sole basis for an investment decision. See README.md.

Usage:
    python forecasting.py --ticker AAPL --start 2015-01-01 --horizon 1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from data_utils import fetch_ohlcv, add_technical_indicators

FEATURE_COLUMNS = [
    "Close", "Volume", "return_1d", "log_return_1d",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_5", "ema_10", "ema_20", "ema_50",
    "volatility_10", "volatility_20",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "volume_change", "volume_sma_10",
]
TARGET_COLUMN = "Close"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class ForecastConfig:
    ticker: str
    start: str
    end: str | None = None
    source: str = "yahoo"         # "yahoo" or "settrade" (see settrade_source.py)
    horizon: int = 1              # trading days ahead to predict
    lookback: int = 60            # sequence length fed to the RNN
    model_type: str = "lstm"      # "lstm" or "gru"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    train_frac: float = 0.70
    val_frac: float = 0.15        # remainder is test
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    patience: int = 10            # early stopping
    seed: int = 42
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))


# --------------------------------------------------------------------------- #
# Data preparation (chronological, leakage-safe)
# --------------------------------------------------------------------------- #
def _fit_scalers(train_df: pd.DataFrame, feature_columns: list[str]):
    x_scaler = StandardScaler().fit(train_df[feature_columns])
    y_scaler = StandardScaler().fit(train_df[["target"]])
    return x_scaler, y_scaler


def _to_sequences(df: pd.DataFrame, feature_columns: list[str], lookback: int,
                   x_scaler: StandardScaler, y_scaler: StandardScaler):
    x = x_scaler.transform(df[feature_columns])
    y = y_scaler.transform(df[["target"]]).ravel()
    dates = df.index[lookback:]
    X_seq, y_seq = [], []
    for i in range(lookback, len(df)):
        X_seq.append(x[i - lookback:i])
        y_seq.append(y[i])
    return (np.array(X_seq, dtype=np.float32),
            np.array(y_seq, dtype=np.float32),
            dates)


def prepare_features(cfg: ForecastConfig) -> pd.DataFrame:
    """Fetch data and engineer features + horizon-ahead target column."""
    raw = fetch_ohlcv(cfg.ticker, cfg.start, cfg.end, source=cfg.source)
    feats = add_technical_indicators(raw)
    feats["target"] = feats[TARGET_COLUMN].shift(-cfg.horizon)
    return feats


def build_dataset(cfg: ForecastConfig):
    feats = prepare_features(cfg).dropna()

    n = len(feats)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)

    train_df = feats.iloc[:n_train]
    val_df = feats.iloc[n_train:n_train + n_val]
    test_df = feats.iloc[n_train + n_val:]

    # Fit scalers ONLY on training data to avoid look-ahead leakage
    x_scaler, y_scaler = _fit_scalers(train_df, cfg.feature_columns)

    # Include trailing `lookback` rows from the previous split so val/test
    # sequences aren't artificially shortened (still no leakage: scaler was
    # fit on train only, and future info never enters past sequences).
    val_ctx = pd.concat([train_df.iloc[-cfg.lookback:], val_df])
    test_ctx = pd.concat([val_df.iloc[-cfg.lookback:], test_df])

    X_train, y_train, dates_train = _to_sequences(train_df, cfg.feature_columns, cfg.lookback, x_scaler, y_scaler)
    X_val, y_val, dates_val = _to_sequences(val_ctx, cfg.feature_columns, cfg.lookback, x_scaler, y_scaler)
    X_test, y_test, dates_test = _to_sequences(test_ctx, cfg.feature_columns, cfg.lookback, x_scaler, y_scaler)

    return {
        "X_train": X_train, "y_train": y_train, "dates_train": dates_train,
        "X_val": X_val, "y_val": y_val, "dates_val": dates_val,
        "X_test": X_test, "y_test": y_test, "dates_test": dates_test,
        "x_scaler": x_scaler, "y_scaler": y_scaler,
        "raw_close": feats[TARGET_COLUMN],
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class RecurrentForecaster(nn.Module):
    def __init__(self, n_features: int, cfg: ForecastConfig):
        super().__init__()
        rnn_cls = nn.LSTM if cfg.model_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size // 2, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]        # final time step's hidden state
        return self.head(last).squeeze(-1)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train_model(cfg: ForecastConfig, data: dict, device: str = "cpu"):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = RecurrentForecaster(n_features=len(cfg.feature_columns), cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    X_train = torch.from_numpy(data["X_train"]).to(device)
    y_train = torch.from_numpy(data["y_train"]).to(device)
    X_val = torch.from_numpy(data["X_val"]).to(device)
    y_val = torch.from_numpy(data["y_val"]).to(device)

    n = X_train.shape[0]
    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = X_train[idx], y_train[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate(model, data: dict, device: str = "cpu"):
    model.eval()
    X_test = torch.from_numpy(data["X_test"]).to(device)
    with torch.no_grad():
        pred_scaled = model(X_test).cpu().numpy()

    y_scaler = data["y_scaler"]
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    actual = y_scaler.inverse_transform(data["y_test"].reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    mae = float(np.mean(np.abs(pred - actual)))
    mape = float(np.mean(np.abs((pred - actual) / actual)) * 100)

    # Directional accuracy vs. the price `horizon` days earlier
    prev_close = data["raw_close"].reindex(data["dates_test"]).values
    actual_dir = np.sign(actual - prev_close)
    pred_dir = np.sign(pred - prev_close)
    directional_acc = float(np.mean(actual_dir == pred_dir) * 100)

    # Naive baseline: "tomorrow = today" — a real model should beat this
    naive_rmse = float(np.sqrt(np.mean((prev_close - actual) ** 2)))

    return {
        "rmse": rmse, "mae": mae, "mape_pct": mape,
        "directional_accuracy_pct": directional_acc,
        "naive_baseline_rmse": naive_rmse,
        "beats_naive_baseline": rmse < naive_rmse,
        "dates": [d.strftime("%Y-%m-%d") for d in data["dates_test"]],
        "predicted": pred.tolist(),
        "actual": actual.tolist(),
    }


def _evaluate_fold(model, X_test, y_test, y_scaler, dates_test, raw_close, device="cpu"):
    """Same metric logic as evaluate(), factored out for reuse across folds."""
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_test).to(device)).cpu().numpy()

    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    mae = float(np.mean(np.abs(pred - actual)))

    prev_close = raw_close.reindex(dates_test).values
    actual_dir = np.sign(actual - prev_close)
    pred_dir = np.sign(pred - prev_close)
    directional_acc = float(np.mean(actual_dir == pred_dir) * 100)
    naive_rmse = float(np.sqrt(np.mean((prev_close - actual) ** 2)))

    return {
        "rmse": rmse, "mae": mae,
        "directional_accuracy_pct": directional_acc,
        "naive_baseline_rmse": naive_rmse,
        "beats_naive_baseline": rmse < naive_rmse,
        "n_test_points": int(len(actual)),
    }


# --------------------------------------------------------------------------- #
# Walk-forward (rolling-origin) validation
# --------------------------------------------------------------------------- #
def walk_forward_evaluate(cfg: ForecastConfig, n_folds: int = 5,
                           fold_epochs: int = 30, min_train_frac: float = 0.5,
                           device: str | None = None) -> dict:
    """
    Rolling-origin evaluation: train on an expanding window of past data,
    test on the next contiguous unseen block, repeat n_folds times moving
    forward through the series. Much closer to how the model would actually
    be used than a single train/val/test split — a model that only looks
    good on one lucky split will get exposed here.

    NOTE: retrains a model from scratch each fold, so this is
    n_folds times more expensive than a single build_dataset()+train_model()
    run. fold_epochs defaults lower than the single-split default (100) to
    keep this tractable — increase it if you have the compute budget.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    feats = prepare_features(cfg).dropna()
    n = len(feats)
    first_test_start = int(n * min_train_frac)
    remaining = n - first_test_start
    if remaining < n_folds * cfg.lookback * 2:
        raise ValueError(
            f"Not enough data for {n_folds} folds with lookback={cfg.lookback} "
            f"and min_train_frac={min_train_frac}. Use fewer folds, a smaller "
            f"lookback, or a longer date range."
        )
    fold_size = remaining // n_folds

    fold_results = []
    for fold in range(n_folds):
        test_start = first_test_start + fold * fold_size
        test_end = n if fold == n_folds - 1 else test_start + fold_size

        train_df = feats.iloc[:test_start]
        test_df = feats.iloc[test_start:test_end]
        # trailing context so the first test sequences aren't truncated
        test_ctx = pd.concat([train_df.iloc[-cfg.lookback:], test_df])

        x_scaler, y_scaler = _fit_scalers(train_df, cfg.feature_columns)
        X_train, y_train, _ = _to_sequences(train_df, cfg.feature_columns, cfg.lookback, x_scaler, y_scaler)
        X_test, y_test, dates_test = _to_sequences(test_ctx, cfg.feature_columns, cfg.lookback, x_scaler, y_scaler)

        fold_cfg = ForecastConfig(**{**cfg.__dict__, "epochs": fold_epochs, "patience": max(5, fold_epochs // 4)})
        model = RecurrentForecaster(n_features=len(cfg.feature_columns), cfg=fold_cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=fold_cfg.lr)
        loss_fn = nn.MSELoss()

        Xt = torch.from_numpy(X_train).to(device)
        yt = torch.from_numpy(y_train).to(device)
        n_train_pts = Xt.shape[0]

        model.train()
        for _epoch in range(fold_epochs):
            perm = torch.randperm(n_train_pts)
            for i in range(0, n_train_pts, fold_cfg.batch_size):
                idx = perm[i:i + fold_cfg.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()

        metrics = _evaluate_fold(model, X_test, y_test, y_scaler, dates_test, feats[TARGET_COLUMN], device)
        metrics["fold"] = fold + 1
        metrics["train_start"] = str(train_df.index[0].date())
        metrics["train_end"] = str(train_df.index[-1].date())
        metrics["test_start"] = str(test_df.index[0].date())
        metrics["test_end"] = str(test_df.index[-1].date())
        fold_results.append(metrics)
        print(f"  Fold {fold + 1}/{n_folds}: "
              f"train {metrics['train_start']}..{metrics['train_end']}, "
              f"test {metrics['test_start']}..{metrics['test_end']} — "
              f"RMSE {metrics['rmse']:.3f} (naive {metrics['naive_baseline_rmse']:.3f}), "
              f"dir. acc {metrics['directional_accuracy_pct']:.1f}%")

    rmses = [f["rmse"] for f in fold_results]
    dir_accs = [f["directional_accuracy_pct"] for f in fold_results]
    beats = [f["beats_naive_baseline"] for f in fold_results]

    summary = {
        "n_folds": n_folds,
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
        "mean_directional_accuracy_pct": float(np.mean(dir_accs)),
        "folds_beating_naive_baseline": int(sum(beats)),
        "folds_beating_naive_baseline_pct": float(np.mean(beats) * 100),
    }
    return {"summary": summary, "folds": fold_results}


# --------------------------------------------------------------------------- #
# Live (out-of-sample) prediction — for the daily prediction logger
# --------------------------------------------------------------------------- #
def predict_next(cfg: ForecastConfig, device: str | None = None) -> dict:
    """
    Train on all available history and predict `cfg.horizon` trading days
    beyond the most recent available bar — i.e. a genuine out-of-sample
    forecast with no ground truth yet available to check it against.
    Intended for predict_logger.py: log the prediction now, compare against
    the real price once the target date has passed.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    full_feats = prepare_features(cfg)          # has NaN target for last `horizon` rows
    train_feats = full_feats.dropna()             # rows with a known target, for training

    n = len(train_feats)
    n_val = max(cfg.lookback + 1, int(n * cfg.val_frac))
    train_df = train_feats.iloc[:-n_val]
    val_df = train_feats.iloc[-n_val:]

    x_scaler, y_scaler = _fit_scalers(train_df, cfg.feature_columns)
    val_ctx = pd.concat([train_df.iloc[-cfg.lookback:], val_df])
    X_train, y_train, _ = _to_sequences(train_df, cfg.feature_columns, cfg.lookback, x_scaler, y_scaler)
    X_val, y_val, _ = _to_sequences(val_ctx, cfg.feature_columns, cfg.lookback, x_scaler, y_scaler)

    data = {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}
    model, _ = train_model(cfg, data, device=device)

    # Build the input window from the MOST RECENT `lookback` rows of the full
    # feature set (including rows whose target is unknown/NaN — we only use
    # their input features, not their target, so this is not leakage).
    last_window = full_feats[cfg.feature_columns].iloc[-cfg.lookback:]
    x_last = x_scaler.transform(last_window)
    X_pred = torch.from_numpy(x_last[np.newaxis, :, :].astype(np.float32)).to(device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_pred).cpu().numpy()
    predicted_price = float(y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0])

    last_date = full_feats.index[-1]
    last_close = float(full_feats[TARGET_COLUMN].iloc[-1])
    # Approximate target date: last_date + horizon business days (actual SET/
    # exchange holidays aren't modeled — the logger's backfill step is
    # tolerant of this, see predict_logger.py).
    target_date = (last_date + pd.tseries.offsets.BDay(cfg.horizon))

    return {
        "ticker": cfg.ticker,
        "source": cfg.source,
        "model_type": cfg.model_type,
        "horizon": cfg.horizon,
        "as_of_date": last_date.strftime("%Y-%m-%d"),
        "target_date": target_date.strftime("%Y-%m-%d"),
        "last_close": last_close,
        "predicted_price": predicted_price,
        "predicted_change_pct": (predicted_price - last_close) / last_close * 100,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="LSTM/GRU stock price forecaster")
    p.add_argument("--ticker", required=True)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--source", choices=["yahoo", "settrade"], default="yahoo",
                    help="'settrade' requires SETTRADE_* env vars — see settrade_source.py")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--model-type", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--walk-forward", action="store_true",
                    help="Run rolling-origin walk-forward validation instead of a single split")
    p.add_argument("--folds", type=int, default=5, help="Number of folds for --walk-forward")
    p.add_argument("--fold-epochs", type=int, default=30, help="Epochs per fold for --walk-forward")
    p.add_argument("--out", default="forecast_results.json")
    args = p.parse_args()

    cfg = ForecastConfig(
        ticker=args.ticker, start=args.start, end=args.end, source=args.source,
        horizon=args.horizon, lookback=args.lookback,
        model_type=args.model_type, epochs=args.epochs,
    )

    if args.walk_forward:
        print(f"Running {args.folds}-fold walk-forward validation for {cfg.ticker}...")
        results = walk_forward_evaluate(cfg, n_folds=args.folds, fold_epochs=args.fold_epochs)
        print(json.dumps(results["summary"], indent=2))
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Full results written to {args.out}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fetching data for {cfg.ticker}...")
    data = build_dataset(cfg)
    print(f"Train/Val/Test sequences: "
          f"{len(data['X_train'])}/{len(data['X_val'])}/{len(data['X_test'])}")

    print(f"Training {cfg.model_type.upper()} model on {device}...")
    model, history = train_model(cfg, data, device=device)

    results = evaluate(model, data, device=device)
    print(json.dumps({k: v for k, v in results.items()
                       if k not in ("dates", "predicted", "actual")}, indent=2))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
