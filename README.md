# Stock Toolkit

Two independent tools, sharing a data-utils module:

1. **`forecasting.py`** — LSTM/GRU price forecaster (PyTorch), in the spirit of
   the recurrent-architecture approaches (LSTM/GRU/CNN-LSTM) that dominate the
   recent DL-for-financial-forecasting literature.
2. **`portfolio.py`** — Portfolio analytics + mean-variance optimizer
   (max Sharpe / min volatility / efficient frontier).

## Setup

```bash
pip install -r requirements.txt
```

## Data sources

Both tools accept `--source yahoo` (default) or `--source settrade`.

- **`yahoo`** — free, unofficial (via `yfinance`), works for global tickers
  including Thai stocks under the `.BK` suffix (e.g. `PTT.BK`).
- **`settrade`** — official Stock Exchange of Thailand data via the
  [Settrade Open API](https://developer.settrade.com/open-api/), a
  subsidiary of SET itself. Requires credentials — see `settrade_source.py`
  for one-time setup (register a sandbox account, generate an app id/secret,
  set them as `SETTRADE_APP_ID`, `SETTRADE_APP_SECRET`, `SETTRADE_BROKER_ID`,
  `SETTRADE_APP_CODE` env vars). Use bare SET symbols with this source
  (`PTT`, not `PTT.BK`).

```bash
python forecasting.py --ticker PTT --source settrade --start 2018-01-01
python portfolio.py --tickers PTT KBANK AOT --source settrade --start 2018-01-01
```

The `settrade_source.py` module's constructor and API calls were verified
directly against the installed `settrade-v2` SDK source, but the response
parsing couldn't be tested against a live authenticated call in the
environment this was built in — validate it against your own sandbox
credentials, and see the module docstring if you hit a field-name mismatch.

## 1. Forecasting

```bash
python forecasting.py --ticker AAPL --start 2015-01-01 --horizon 1 --model-type lstm
```

- Fetches OHLCV data, engineers ~20 features (SMA/EMA, RSI, MACD, Bollinger
  Bands, volatility, volume features).
- Chronological train/val/test split — **no shuffling**, so no future
  information leaks into training (a common bug in naive implementations).
- Scalers are fit on the training split only.
- Reports RMSE / MAE / MAPE / directional accuracy, **and always compares
  against a naive "tomorrow = today" baseline** — if your model doesn't beat
  that, it isn't adding value. Don't skip this check.

Key options: `--horizon` (days ahead), `--lookback` (sequence window),
`--model-type lstm|gru`, `--epochs`.

## 2. Portfolio analyzer / optimizer

```bash
python portfolio.py --tickers AAPL MSFT GOOGL BND --weights 0.3 0.3 0.2 0.2 \
    --start 2018-01-01 --optimize max_sharpe
```

Computes annualized return/volatility, Sharpe, Sortino, max drawdown,
historical VaR(95%), and a correlation matrix for your current weights, then
(optionally) solves for the long-only max-Sharpe or min-volatility weights
via SLSQP mean-variance optimization.

## Honest limitations — please read

This toolkit follows patterns from the DL-for-finance literature (e.g. the
architectures surveyed in Giantsidi & Tarantola, *Deep Learning for Financial
Forecasting: A Review of Recent Trends*, 2025), but that same literature is
upfront that the field has real, unresolved gaps: poor robustness in extreme
market conditions, no standardized evaluation protocol, weak interpretability,
and a large gap between backtested and live performance.

Concretely:

- **Beating a naive baseline on historical data is not the same as being
  profitable.** Transaction costs, slippage, and the fact that markets are
  close to efficient mean most published "high accuracy" results don't survive
  contact with live trading.
- **Backtests are easy to fool yourself with.** Watch for lookahead bias
  (this code guards against it structurally, but changing the code carelessly
  can reintroduce it), survivorship bias in your ticker list, and overfitting
  to one time period.
- **Mean-variance optimization is sensitive to estimation error** — small
  changes in your return/covariance estimates can produce very different
  "optimal" weights. Treat the optimizer's output as one input, not a verdict.
- **Nothing here is financial advice.** These are research/engineering tools
  for exploring the methods the literature describes, not a system to size
  real positions off of without independent judgment (and ideally professional
  advice for anything material).

## Extending it

- Swap in a Transformer encoder (e.g. PatchTST-style patching) in
  `forecasting.py` in place of `RecurrentForecaster` — the data pipeline
  already produces fixed-length sequences, so the model class is the only
  thing that needs to change.
- ~~Add walk-forward (rolling-origin) re-training~~ — done, see below.
- Add sentiment/fundamental features to `data_utils.add_technical_indicators`
  — the paper flags multi-modal feature integration as a strong recent trend.

## Walk-forward validation

A single train/val/test split can look good or bad by luck. Walk-forward
validation retrains on an expanding window and tests on the next unseen
block, repeated across several folds — much closer to how the model would
actually be used, and it will expose a model that only looks good on one
lucky split.

```bash
python forecasting.py --ticker AAPL --walk-forward --folds 5 --fold-epochs 30
```

Prints per-fold RMSE / directional accuracy / naive-baseline comparison,
plus an aggregate summary (mean RMSE, % of folds beating the naive
baseline). This is `n_folds`× more expensive than a single run — expect
several minutes, not seconds.

## Live prediction tracking

`predict_logger.py` generates a genuine out-of-sample prediction (no ground
truth exists yet) and logs it to `predictions_log.csv`. On each run it also
backfills `actual_price` for any past predictions whose target date has now
passed — building an honest, tamper-proof track record over time, since
every prediction was logged *before* the outcome was known.

```bash
python predict_logger.py --tickers AAPL MSFT --source yahoo --epochs 50
```

Run this on a schedule (see the GitHub Actions + Streamlit deployment guide
below) and `streamlit_app.py`'s "Live Predictions" tab will chart the
model's real accuracy over time — the only evaluation that can't be
accidentally leakage-contaminated.

---

## Deploying with zero local machine (GitHub + Streamlit Community Cloud)

This setup uses **GitHub Actions** as a free, scheduled "server" that keeps
`predictions_log.csv` updated, and **Streamlit Community Cloud** as a free
host for the interactive dashboard. Both run entirely in the cloud — you
never need to run anything on your own computer.

### 1. Create a GitHub repository (via the web, no git needed)

1. Go to [github.com/new](https://github.com/new), name it (e.g.
   `stock-toolkit`), set it to **Public** (required for free Streamlit
   Community Cloud hosting), and click **Create repository**.
2. On the empty repo page, click **uploading an existing file**.
3. Drag in every file from this toolkit: `data_utils.py`, `settrade_source.py`,
   `forecasting.py`, `portfolio.py`, `predict_logger.py`, `streamlit_app.py`,
   `requirements.txt`, `README.md`, `predictions_log.csv`.
4. For the `.github/workflows/daily_predictions.yml` file specifically: click
   **Add file → Create new file** instead of uploading, and type the full
   path `.github/workflows/daily_predictions.yml` into the filename box —
   GitHub auto-creates the folders. Paste the file's contents in and commit.
5. Click **Commit changes** to finish the initial upload.

### 2. (Optional) Add Settrade credentials as repo secrets

Only needed if `predict_logger.py`'s workflow uses `--source settrade`.

1. In your repo: **Settings → Secrets and variables → Actions → New repository secret**.
2. Add four secrets: `SETTRADE_APP_ID`, `SETTRADE_APP_SECRET`,
   `SETTRADE_BROKER_ID`, `SETTRADE_APP_CODE` — same values as the local
   setup in `settrade_source.py`'s docstring.

### 3. Turn on and test the GitHub Action

1. Go to the **Actions** tab in your repo. If prompted, click **"I understand
   my workflows, enable them."**
2. Click **Daily Predictions** in the left sidebar, then **Run workflow**
   (this is the `workflow_dispatch` trigger — lets you test without waiting
   for the schedule) → **Run workflow** again to confirm.
3. Wait ~1–3 minutes, refresh, and check the run succeeded (green check).
   Open it to see the logs — you should see "Logged AAPL: ..." etc.
4. Check the repo's file list: `predictions_log.csv` should now have new
   rows, committed by `github-actions[bot]`.
5. From now on it runs automatically on the schedule in the workflow file
   (weekdays at 17:00 Bangkok time by default) — edit the `cron:` line and
   the `--tickers`/`--source` flags in the workflow file to customize.

### 4. Deploy the dashboard on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account (authorize Streamlit to access your repos).
2. Click **New app** (or **Create app**).
3. Pick your repository, the branch (`main`), and set **Main file path** to
   `streamlit_app.py`.
4. If you're using Settrade: click **Advanced settings → Secrets** and paste
   in TOML format:
   ```toml
   SETTRADE_APP_ID = "your-app-id"
   SETTRADE_APP_SECRET = "your-app-secret"
   SETTRADE_BROKER_ID = "your-broker-id"
   SETTRADE_APP_CODE = "your-app-code"
   ```
5. Click **Deploy**. First build takes a few minutes (installing PyTorch
   etc.) — subsequent redeploys are faster.
6. You'll get a URL like `https://your-app-name.streamlit.app` — that's your
   live dashboard.

### 5. How it stays up to date

- The GitHub Action commits a fresh `predictions_log.csv` to your repo on
  its schedule.
- Streamlit Community Cloud watches your repo and **automatically
  redeploys** on new commits — so the "Live Predictions" tab refreshes
  itself with no manual steps after the first setup.
- To change tracked tickers, edit the `--tickers` line in
  `.github/workflows/daily_predictions.yml` directly on GitHub (Edit this
  file → Commit) — no local machine needed for that either.

### A note on free-tier limits

- Streamlit Community Cloud's free tier is CPU-only with ~1GB RAM — the
  "Forecast Demo" tab is deliberately capped (short lookback, ≤60 epochs) to
  stay responsive there. For serious walk-forward runs, use the GitHub
  Action (it gets a full GitHub-hosted runner) or run locally/in Colab.
- `requirements.txt` pins CPU-only PyTorch wheels via
  `--extra-index-url https://download.pytorch.org/whl/cpu` — the default
  PyTorch install pulls several GB of CUDA packages that aren't needed here
  and would blow past Streamlit Cloud's build limits.
