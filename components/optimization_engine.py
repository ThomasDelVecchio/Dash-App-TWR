"""
Portfolio Optimization Engine
==============================
Pure computation layer for Efficient Frontier, Monte Carlo projection,
drawdown, and rolling Sharpe analysis.  No Dash / UI code here.

Uses:
  - yfinance (via data_loader.fetch_price_history) for price data
  - PyPortfolioOpt for mean-variance optimisation
  - numpy / pandas for Monte Carlo & analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt.exceptions import OptimizationError

from data_loader import fetch_price_history, load_holdings
from config import RISK_FREE_RATE
from components.monte_carlo import ASSET_CLASS_BENCHMARKS, TICKER_PROXY_BASKETS, _normalize_weights


GAP_BUFFER_DAYS = 30

TICKER_PROXY_OVERRIDES = {
    "SPMO": "MTUM",
    "VXUS": "VEU",
    "BND": "AGG",
    "AVDV": "SCZ",
    "AVUV": "IWN",
    "XMHQ": "MDY",
    "FBTC": "BTC-USD",
    "IBIT": "BTC-USD",
}

KNOWN_BENCHMARK_CLASSES = {
    "SPY": "US Large Cap",
    "VOO": "US Large Cap",
    "QQQ": "US Growth",
    "VXUS": "International Equity",
    "VEU": "International Equity",
    "GLD": "Gold / Precious Metals",
    "BND": "Fixed Income",
    "AGG": "Fixed Income",
    "FBTC": "Digital Assets",
    "IBIT": "Digital Assets",
    "BTC-USD": "Digital Assets",
}


def _build_holdings_map() -> dict:
    try:
        holdings_df = load_holdings()
        if holdings_df.empty:
            return {}
        holdings_df = holdings_df.copy()
        holdings_df["ticker"] = holdings_df["ticker"].astype(str).str.upper()
        if "asset_class" not in holdings_df.columns:
            return {}
        return holdings_df.set_index("ticker")["asset_class"].to_dict()
    except Exception:
        return {}


def _resolve_asset_class(ticker: str, holdings_map: dict) -> str:
    if ticker in holdings_map:
        return holdings_map[ticker]
    if ticker in KNOWN_BENCHMARK_CLASSES:
        return KNOWN_BENCHMARK_CLASSES[ticker]
    for ac, proxy in ASSET_CLASS_BENCHMARKS.items():
        if proxy and str(proxy).upper() == ticker:
            return ac
    return "US Large Cap"


def _resolve_proxy_ticker(asset_class: str) -> str:
    proxy = ASSET_CLASS_BENCHMARKS.get(asset_class)
    if proxy:
        return str(proxy).upper()
    return str(ASSET_CLASS_BENCHMARKS.get("US Large Cap", "SPY")).upper()


def _build_weighted_proxy_return_series(
    prices: pd.DataFrame,
    full_index: pd.DatetimeIndex,
    basket_weights: dict,
) -> tuple[pd.Series | None, pd.Timestamp | None]:
    if prices.empty or not basket_weights:
        return None, None

    normalized = _normalize_weights(basket_weights)
    if not normalized:
        return None, None

    components = [t for t in normalized.keys() if t in prices.columns]
    if not components:
        return None, None

    comp_prices = prices[components].reindex(full_index)
    comp_rets = comp_prices.pct_change()
    weight_series = pd.Series({t: normalized[t] for t in components})

    numer = comp_rets.multiply(weight_series, axis=1).sum(axis=1)
    denom = comp_rets.notna().multiply(weight_series, axis=1).sum(axis=1)
    proxy_ret = numer.div(denom)

    valid_idx = proxy_ret[proxy_ret.notna()].index
    first_valid = valid_idx.min() if len(valid_idx) else None
    return proxy_ret, first_valid


def _format_proxy_basket_label(basket_weights: dict) -> str:
    normalized = _normalize_weights(basket_weights)
    if not normalized:
        return ""
    parts = [f"{ticker}({weight * 100.0:.0f}%)" for ticker, weight in normalized.items()]
    return " + ".join(parts)


def _collect_proxy_tickers(tickers: list[str], holdings_map: dict) -> set[str]:
    proxies = set()
    for ticker in tickers:
        basket = TICKER_PROXY_BASKETS.get(ticker)
        if basket:
            proxies.update({k for k in basket.keys() if k and k != "CASH"})

        override = TICKER_PROXY_OVERRIDES.get(ticker)
        if override and override != "CASH":
            proxies.add(override)

        ac = _resolve_asset_class(ticker, holdings_map)
        proxy = _resolve_proxy_ticker(ac)
        if proxy and proxy != "CASH":
            proxies.add(proxy)

    return {str(t).upper() for t in proxies}


def _snap_to_trading_start(index: pd.DatetimeIndex, requested_start: pd.Timestamp) -> pd.Timestamp:
    if index.empty:
        return requested_start
    pos = index.searchsorted(requested_start)
    if pos >= len(index):
        return index[-1]
    return index[pos]


def _splice_proxy_histories(
    prices: pd.DataFrame,
    requested_start_date: pd.Timestamp,
    tickers: list[str],
    holdings_map: dict,
    gap_buffer_days: int = GAP_BUFFER_DAYS,
) -> tuple[pd.DataFrame, list[dict]]:
    if prices.empty:
        return prices, []

    full_index = prices.index[prices.index >= requested_start_date]
    if full_index.empty:
        return prices, []

    healed = {}
    logs: list[dict] = []
    buffer_cutoff = requested_start_date + pd.Timedelta(days=gap_buffer_days)

    for ticker in tickers:
        if ticker in prices.columns:
            series = prices[ticker].reindex(full_index)
        else:
            series = pd.Series(index=full_index, dtype=float)

        first_valid = series.first_valid_index()
        needs_splice = first_valid is None or first_valid > buffer_cutoff

        if not needs_splice:
            healed[ticker] = series.ffill()
            continue

        asset_class = _resolve_asset_class(ticker, holdings_map)
        basket = TICKER_PROXY_BASKETS.get(ticker)
        proxy_ticker = TICKER_PROXY_OVERRIDES.get(ticker) or _resolve_proxy_ticker(asset_class)
        proxy_series = None
        proxy_first_valid = None
        proxy_label = None

        if basket:
            proxy_series, proxy_first_valid = _build_weighted_proxy_return_series(prices, full_index, basket)
            proxy_label = _format_proxy_basket_label(basket)
            proxy_ret = proxy_series
        else:
            if proxy_ticker == ticker:
                proxy_ticker = _resolve_proxy_ticker("US Large Cap")
            proxy_px = prices[proxy_ticker].reindex(full_index) if proxy_ticker in prices.columns else None
            proxy_series = proxy_px
            proxy_first_valid = proxy_px.first_valid_index() if proxy_px is not None else None
            proxy_label = proxy_ticker
            proxy_ret = proxy_px.pct_change() if proxy_px is not None else None

        status = "Spliced"
        if proxy_series is None or proxy_series.dropna().empty:
            status = "Failed"
            healed[ticker] = series.ffill()
            proxy_label = None
        else:
            if proxy_first_valid is None or proxy_first_valid > buffer_cutoff:
                status = "Partial History"

            orig_ret = series.pct_change()
            combined_ret = orig_ret if proxy_ret is None else orig_ret.fillna(proxy_ret)
            combined_ret = combined_ret.fillna(0.0)
            synthetic = (1.0 + combined_ret).cumprod()
            if not synthetic.empty:
                synthetic.iloc[0] = 1.0
            healed[ticker] = synthetic * 100.0

        logs.append({
            "Ticker": ticker,
            "Status": status,
            "Proxy Used": proxy_label,
            "Asset Class": asset_class,
            "Original Start": first_valid.date().isoformat() if first_valid is not None else None,
            "Proxy Start": proxy_first_valid.date().isoformat() if proxy_first_valid is not None else None,
            "Requested Start": requested_start_date.date().isoformat() if requested_start_date is not None else None,
        })

    if not healed:
        return prices.reindex(full_index), logs

    healed_df = pd.DataFrame(healed, index=full_index)
    return healed_df, logs


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_optimization_prices(
    tickers: List[str],
    years_back: int = 10,
    use_proxy_splice: bool = False,
) -> pd.DataFrame:
    """Return a daily close-price DataFrame for the given tickers.

    Uses ``use_adj_close=False`` to stay consistent with the hybrid
    FMP / yfinance pipeline the rest of the app relies on.
    """
    requested_tickers = sorted({str(t).upper() for t in tickers if str(t).strip() and str(t).upper() != "CASH"})
    if not requested_tickers:
        return pd.DataFrame()

    holdings_map = _build_holdings_map()
    proxy_tickers = _collect_proxy_tickers(requested_tickers, holdings_map) if use_proxy_splice else set()
    fetch_universe = sorted(set(requested_tickers) | proxy_tickers)

    prices = fetch_price_history(fetch_universe, years_back=years_back, use_adj_close=False)
    if prices.empty:
        return prices

    prices = prices[[c for c in prices.columns if c in fetch_universe]]
    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        return prices

    requested_start = prices.index.max() - pd.DateOffset(years=years_back)
    requested_start = _snap_to_trading_start(prices.index, requested_start)

    if use_proxy_splice:
        healed_prices, proxy_log = _splice_proxy_histories(
            prices=prices,
            requested_start_date=requested_start,
            tickers=requested_tickers,
            holdings_map=holdings_map,
        )
        working = healed_prices
    else:
        proxy_log = []
        working = prices.reindex(columns=requested_tickers)
        working = working[working.index >= requested_start]

    working = working.reindex(columns=requested_tickers)
    working = working.dropna(axis=1, how="all")
    working = working.ffill().dropna(how="any")

    working.attrs["proxy_log"] = proxy_log
    working.attrs["proxy_splice_enabled"] = bool(use_proxy_splice)
    working.attrs["requested_start"] = pd.Timestamp(requested_start).strftime("%Y-%m-%d")
    return working


# ============================================================
# EFFICIENT FRONTIER
# ============================================================

def compute_efficient_frontier(
    prices: pd.DataFrame,
    weight_bounds: Tuple[float, float] = (0.0, 1.0),
    sector_mapper: Optional[Dict[str, str]] = None,
    sector_upper: Optional[Dict[str, float]] = None,
    ticker_floors: Optional[Dict[str, float]] = None,
    ticker_caps: Optional[Dict[str, float]] = None,
    n_points: int = 80,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """
    Compute the Efficient Frontier and key optimal portfolios.

    Returns
    -------
    dict with keys:
        frontier_vols   : list[float]   – annualised σ for each frontier point
        frontier_rets   : list[float]   – annualised μ for each frontier point
        max_sharpe      : dict          – {weights, ret, vol, sharpe}
        min_vol         : dict          – {weights, ret, vol, sharpe}
        individual      : list[dict]    – per-asset {ticker, ret, vol}
        mu              : pd.Series     – expected returns
        cov             : pd.DataFrame  – covariance matrix
    """

    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.sample_cov(prices)
    tickers = list(mu.index)

    # ---- Build per-ticker weight bounds ----------------------------
    lower, upper = weight_bounds
    bounds = {}
    for t in tickers:
        lo = lower
        hi = upper
        if ticker_floors and t in ticker_floors:
            lo = ticker_floors[t]
        if ticker_caps and t in ticker_caps:
            hi = ticker_caps[t]
        if lo > hi:
            raise ValueError(f"Infeasible bounds for {t}: lower {lo:.4f} > upper {hi:.4f}")
        bounds[t] = (lo, hi)
    weight_bounds_list = tuple(bounds[t] for t in tickers)

    # ---- Max Sharpe ------------------------------------------------
    try:
        ef_sharpe = EfficientFrontier(mu, cov, weight_bounds=weight_bounds_list)
        if sector_mapper and sector_upper:
            ef_sharpe.add_sector_constraints(sector_mapper, sector_upper)
        ef_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
        sharpe_weights = ef_sharpe.clean_weights()
        sharpe_perf = ef_sharpe.portfolio_performance(
            verbose=False, risk_free_rate=risk_free_rate
        )
        max_sharpe = {
            "weights": dict(sharpe_weights),
            "ret": sharpe_perf[0],
            "vol": sharpe_perf[1],
            "sharpe": sharpe_perf[2],
        }
    except (OptimizationError, ValueError):
        max_sharpe = None

    # ---- Minimum Volatility ----------------------------------------
    try:
        ef_minvol = EfficientFrontier(mu, cov, weight_bounds=weight_bounds_list)
        if sector_mapper and sector_upper:
            ef_minvol.add_sector_constraints(sector_mapper, sector_upper)
        ef_minvol.min_volatility()
        minvol_weights = ef_minvol.clean_weights()
        minvol_perf = ef_minvol.portfolio_performance(
            verbose=False, risk_free_rate=risk_free_rate
        )
        min_vol = {
            "weights": dict(minvol_weights),
            "ret": minvol_perf[0],
            "vol": minvol_perf[1],
            "sharpe": minvol_perf[2],
        }
    except (OptimizationError, ValueError):
        min_vol = None

    # ---- Frontier curve (efficient_risk at many target vols) -------
    # Determine vol range from the two anchor portfolios and individual assets
    daily_rets = prices.pct_change().dropna()
    asset_vols = daily_rets.std() * np.sqrt(252)

    vol_min = min_vol["vol"] if min_vol else float(asset_vols.min()) * 0.8
    vol_max = float(asset_vols.max()) * 1.3   # extend past the riskiest asset
    if max_sharpe:
        vol_max = max(vol_max, max_sharpe["vol"] * 1.5)
    if min_vol:
        vol_min = min(vol_min, min_vol["vol"] * 0.95)

    target_vols = np.linspace(vol_min, vol_max, n_points)
    frontier_vols: List[float] = []
    frontier_rets: List[float] = []

    for tv in target_vols:
        try:
            ef_pt = EfficientFrontier(mu, cov, weight_bounds=weight_bounds_list)
            if sector_mapper and sector_upper:
                ef_pt.add_sector_constraints(sector_mapper, sector_upper)
            ef_pt.efficient_risk(target_volatility=tv, market_neutral=False)
            perf = ef_pt.portfolio_performance(
                verbose=False, risk_free_rate=risk_free_rate
            )
            frontier_vols.append(perf[1])
            frontier_rets.append(perf[0])
        except (OptimizationError, ValueError):
            continue

    # ---- Individual asset risk/return ------------------------------
    individual = []
    for t in tickers:
        ann_ret = float(mu[t])
        ann_vol = float(asset_vols[t]) if t in asset_vols.index else 0.0
        individual.append({"ticker": t, "ret": ann_ret, "vol": ann_vol})

    return {
        "frontier_vols": frontier_vols,
        "frontier_rets": frontier_rets,
        "max_sharpe": max_sharpe,
        "min_vol": min_vol,
        "individual": individual,
        "mu": mu,
        "cov": cov,
    }


def compute_target_volatility_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    target_vol: float,
    weight_bounds: Tuple[float, float] = (0.0, 1.0),
    ticker_floors: Optional[Dict[str, float]] = None,
    ticker_caps: Optional[Dict[str, float]] = None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Optional[dict]:
    """
    Find the maximum-return portfolio for a given target volatility.
    """
    tickers = list(mu.index)
    lower, upper = weight_bounds
    bounds = {}
    for t in tickers:
        lo = lower
        hi = upper
        if ticker_floors and t in ticker_floors:
            lo = ticker_floors[t]
        if ticker_caps and t in ticker_caps:
            hi = ticker_caps[t]
        if lo > hi:
            raise ValueError(f"Infeasible bounds for {t}: lower {lo:.4f} > upper {hi:.4f}")
        bounds[t] = (lo, hi)
    weight_bounds_list = tuple(bounds[t] for t in tickers)

    try:
        ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds_list)
        ef.efficient_risk(target_volatility=target_vol, market_neutral=False)
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(
            verbose=False, risk_free_rate=risk_free_rate
        )
        return {
            "weights": dict(weights),
            "ret": perf[0],
            "vol": perf[1],
            "sharpe": perf[2],
        }
    except (OptimizationError, ValueError):
        return None


# ============================================================
# MONTE CARLO PROJECTION (from optimised weights)
# ============================================================

def run_optimization_monte_carlo(
    mu: pd.Series,
    cov: pd.DataFrame,
    weights: Dict[str, float],
    initial_value: float = 100_000,
    horizon_years: int = 10,
    n_simulations: int = 1_000,
    monthly_contribution: float = 0.0,
    random_seed: int = 42,
) -> dict:
    """
    Project future wealth paths using Geometric Brownian Motion seeded by
    the optimised portfolio's expected return & covariance.

    Returns dict with keys: years, percentiles (10/25/50/75/90),
    final_distribution, metrics.
    """
    # Portfolio expected return & volatility from the weight vector
    w = np.array([weights.get(t, 0.0) for t in mu.index])
    w = w / w.sum() if w.sum() > 0 else w
    port_mu = float(w @ mu.values)
    port_var = float(w @ cov.values @ w)
    port_sigma = np.sqrt(port_var)

    n_steps = horizon_years * 12
    dt = 1 / 12
    rng = np.random.default_rng(random_seed)

    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = initial_value

    drift = (port_mu - 0.5 * port_sigma ** 2) * dt
    shock_std = port_sigma * np.sqrt(dt)

    for t in range(1, n_steps + 1):
        z = rng.standard_normal(n_simulations)
        paths[:, t] = paths[:, t - 1] * np.exp(drift + shock_std * z) + monthly_contribution

    years = np.linspace(0, horizon_years, n_steps + 1).tolist()
    pcts = {}
    for p in [10, 25, 50, 75, 90]:
        pcts[str(p)] = np.percentile(paths, p, axis=0).tolist()

    final = paths[:, -1]
    return {
        "years": years,
        "percentiles": pcts,
        "final_distribution": final.tolist(),
        "metrics": {
            "mu": port_mu,
            "sigma": port_sigma,
            "median_final": float(np.median(final)),
            "var_95": float(np.percentile(final, 5)),
            "cvar_95": float(final[final <= np.percentile(final, 5)].mean()),
        },
    }


# ============================================================
# HISTORICAL BACK-TEST OF OPTIMISED WEIGHTS
# ============================================================

def backtest_optimized_weights(
    prices: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Return a *growth-of-$1* series for a buy-and-hold portfolio with
    the given fixed weights, rebalanced daily (simplification).
    """
    daily_rets = prices.pct_change().dropna()
    tickers = [t for t in weights if t in daily_rets.columns and weights[t] > 0]
    if not tickers:
        return pd.Series(dtype=float)

    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    port_ret = (daily_rets[tickers] * w).sum(axis=1)
    growth = (1 + port_ret).cumprod()
    return growth


def compute_underwater(growth: pd.Series) -> pd.Series:
    """Return drawdown series (negative %) from running peak."""
    if growth.empty:
        return pd.Series(dtype=float)
    hwm = growth.cummax()
    dd = (growth - hwm) / hwm
    return dd * 100  # percentage


def compute_rolling_sharpe(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    window_years: int = 3,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.Series:
    """
    Compute the rolling annualised Sharpe ratio for the optimised
    portfolio over a `window_years` window.
    """
    daily_rets = prices.pct_change().dropna()
    tickers = [t for t in weights if t in daily_rets.columns and weights[t] > 0]
    if not tickers:
        return pd.Series(dtype=float)

    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    port_ret = (daily_rets[tickers] * w).sum(axis=1)

    window = window_years * 252
    rolling_mean = port_ret.rolling(window).mean() * 252
    rolling_std = port_ret.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
    return rolling_sharpe.dropna()
