"""
portfolio.py
Portfolio analytics: return/risk metrics for a set of holdings, and
mean-variance weight optimization (max Sharpe, min volatility, efficient
frontier). Educational tool — not investment advice.

Usage:
    python portfolio.py --tickers AAPL MSFT GOOGL BND --weights 0.3 0.3 0.2 0.2 \
        --start 2018-01-01
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_utils import fetch_multi_close

TRADING_DAYS = 252


@dataclass
class PortfolioConfig:
    tickers: list[str]
    weights: list[float] | None = None      # None -> equal weight
    start: str = "2018-01-01"
    end: str | None = None
    source: str = "yahoo"                    # "yahoo" or "settrade"
    risk_free_rate: float = 0.02             # annualized


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def annualize_return(daily_returns: pd.Series) -> float:
    return float((1 + daily_returns.mean()) ** TRADING_DAYS - 1)


def annualize_vol(daily_returns: pd.Series) -> float:
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    excess = daily_returns - risk_free_rate / TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float) -> float:
    excess = daily_returns - risk_free_rate / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf")
    return float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(cum_returns: pd.Series) -> float:
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return float(drawdown.min())


def value_at_risk(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR: worst expected daily loss at the given confidence level."""
    return float(np.percentile(daily_returns, (1 - confidence) * 100))


def portfolio_daily_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return returns.dot(weights)


def analyze_portfolio(cfg: PortfolioConfig) -> dict:
    prices = fetch_multi_close(cfg.tickers, cfg.start, cfg.end, source=cfg.source)
    returns = compute_returns(prices)

    weights = np.array(cfg.weights) if cfg.weights else np.repeat(1 / len(cfg.tickers), len(cfg.tickers))
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {weights.sum():.4f}")

    port_returns = portfolio_daily_returns(returns, weights)
    cum_returns = (1 + port_returns).cumprod()

    per_asset = {}
    for i, t in enumerate(cfg.tickers):
        r = returns[t]
        per_asset[t] = {
            "weight": float(weights[i]),
            "annual_return_pct": annualize_return(r) * 100,
            "annual_volatility_pct": annualize_vol(r) * 100,
            "sharpe": sharpe_ratio(r, cfg.risk_free_rate),
        }

    result = {
        "period": {"start": str(returns.index.min().date()), "end": str(returns.index.max().date())},
        "tickers": cfg.tickers,
        "weights": weights.tolist(),
        "portfolio": {
            "annual_return_pct": annualize_return(port_returns) * 100,
            "annual_volatility_pct": annualize_vol(port_returns) * 100,
            "sharpe_ratio": sharpe_ratio(port_returns, cfg.risk_free_rate),
            "sortino_ratio": sortino_ratio(port_returns, cfg.risk_free_rate),
            "max_drawdown_pct": max_drawdown(cum_returns) * 100,
            "value_at_risk_95_daily_pct": value_at_risk(port_returns, 0.95) * 100,
            "total_return_pct": (cum_returns.iloc[-1] - 1) * 100,
        },
        "per_asset": per_asset,
        "correlation_matrix": returns.corr().round(3).to_dict(),
    }
    return result


# --------------------------------------------------------------------------- #
# Mean-variance optimization
# --------------------------------------------------------------------------- #
def optimize_weights(returns: pd.DataFrame, objective: str = "max_sharpe",
                      risk_free_rate: float = 0.02,
                      bounds: tuple[float, float] = (0.0, 1.0)) -> dict:
    """
    objective: "max_sharpe" or "min_volatility"
    Long-only, fully-invested (weights sum to 1) mean-variance optimization.
    """
    mean_ret = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    n = len(returns.columns)

    def port_vol(w):
        return float(np.sqrt(w @ cov.values @ w))

    def port_ret(w):
        return float(w @ mean_ret.values)

    def neg_sharpe(w):
        vol = port_vol(w)
        if vol == 0:
            return 0.0
        return -(port_ret(w) - risk_free_rate) / vol

    obj_fn = neg_sharpe if objective == "max_sharpe" else port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = tuple(bounds for _ in range(n))
    x0 = np.repeat(1 / n, n)

    res = minimize(obj_fn, x0, method="SLSQP", bounds=bnds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w = res.x
    return {
        "objective": objective,
        "weights": dict(zip(returns.columns, w.round(4).tolist())),
        "expected_annual_return_pct": port_ret(w) * 100,
        "expected_annual_volatility_pct": port_vol(w) * 100,
        "expected_sharpe": (port_ret(w) - risk_free_rate) / port_vol(w) if port_vol(w) > 0 else 0.0,
    }


def efficient_frontier(returns: pd.DataFrame, n_points: int = 25,
                        bounds: tuple[float, float] = (0.0, 1.0)) -> list[dict]:
    """Trace the efficient frontier by minimizing volatility for a range of target returns."""
    mean_ret = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    n = len(returns.columns)

    target_returns = np.linspace(mean_ret.min(), mean_ret.max(), n_points)
    frontier = []

    for target in target_returns:
        def port_vol(w):
            return float(np.sqrt(w @ cov.values @ w))

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mean_ret.values - t},
        ]
        bnds = tuple(bounds for _ in range(n))
        x0 = np.repeat(1 / n, n)
        res = minimize(port_vol, x0, method="SLSQP", bounds=bnds, constraints=constraints)
        if res.success:
            frontier.append({
                "target_return_pct": target * 100,
                "volatility_pct": port_vol(res.x) * 100,
                "weights": dict(zip(returns.columns, res.x.round(4).tolist())),
            })
    return frontier


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Portfolio risk/return analyzer + optimizer")
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Must sum to 1.0. Defaults to equal weight.")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--source", choices=["yahoo", "settrade"], default="yahoo",
                    help="'settrade' requires SETTRADE_* env vars — see settrade_source.py")
    p.add_argument("--risk-free-rate", type=float, default=0.02)
    p.add_argument("--optimize", choices=["max_sharpe", "min_volatility"], default=None)
    p.add_argument("--out", default="portfolio_results.json")
    args = p.parse_args()

    cfg = PortfolioConfig(tickers=args.tickers, weights=args.weights,
                           start=args.start, end=args.end, source=args.source,
                           risk_free_rate=args.risk_free_rate)

    result = analyze_portfolio(cfg)

    if args.optimize:
        prices = fetch_multi_close(cfg.tickers, cfg.start, cfg.end, source=cfg.source)
        returns = compute_returns(prices)
        result["optimization"] = optimize_weights(
            returns, objective=args.optimize, risk_free_rate=cfg.risk_free_rate
        )

    print(json.dumps(result, indent=2, default=str))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
