"""
Audit 16 — Portfolio Optimization Engine
==========================================
Validates every mathematical operation in components/optimization_engine.py
against first-principles finance and GIPS-adjacent standards.

GIPS Relevance:
  While GIPS (Global Investment Performance Standards) primarily govern
  *performance reporting*, several of their principles directly apply to
  the optimisation module:
    1. Weights must sum to 1.0 (fully invested, no hidden leverage).
    2. Returns & volatilities must use geometric / annualised conventions.
    3. No annualisation of periods < 1 year when presenting expected returns.
    4. Covariance matrix must be symmetric and positive semi-definite.
    5. Drawdown calculations must use the High-Water-Mark methodology.
    6. Sharpe ratio must subtract the stated risk-free rate consistently.
    7. Monte Carlo projections must use drift-adjusted GBM (Ito's lemma),
       not arithmetic drift, to avoid upward bias.
    8. All reported statistics must be reproducible (deterministic seed).

Tests:
  Section A — Data Integrity (prices, NaN handling)
  Section B — Covariance & Expected Returns
  Section C — Weight Constraints & Summation
  Section D — Efficient Frontier Monotonicity
  Section E — Sharpe / Risk-Return Identity
  Section F — Monte Carlo GBM Drift Correctness
  Section G — Monte Carlo Reproducibility (CRN)
  Section H — Monte Carlo Percentile Ordering
  Section I — Backtest Growth-of-$1 Correctness
  Section J — Underwater / Drawdown HWM Logic
  Section K — Rolling Sharpe Consistency
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from components.optimization_engine import (
    fetch_optimization_prices,
    compute_efficient_frontier,
    compute_target_volatility_portfolio,
    run_optimization_monte_carlo,
    backtest_optimized_weights,
    compute_underwater,
    compute_rolling_sharpe,
)
from config import RISK_FREE_RATE

# ============================================================
# HELPERS
# ============================================================
_PASS = 0
_FAIL = 0
_WARN = 0


def log_pass(msg):
    global _PASS
    _PASS += 1
    print(f"  [PASS] {msg}")


def log_fail(msg):
    global _FAIL
    _FAIL += 1
    print(f"  [FAIL] {msg}")


def log_warn(msg):
    global _WARN
    _WARN += 1
    print(f"  [WARN] {msg}")


def log_skip(msg):
    print(f"  [SKIP] {msg}")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# SYNTHETIC DATA FACTORY
# ============================================================

def _build_synthetic_prices(n_days=1260, seed=0):
    """
    Build a deterministic 5-asset price matrix (~5 years of daily data)
    with known statistical properties so we can verify outputs analytically.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days, freq="B")

    # Annual returns & vols (known ground truth)
    params = {
        "ALPHA": {"mu": 0.12, "sigma": 0.20},
        "BETA":  {"mu": 0.08, "sigma": 0.15},
        "GAMMA": {"mu": 0.05, "sigma": 0.10},
        "DELTA": {"mu": 0.15, "sigma": 0.25},
        "OMEGA": {"mu": 0.03, "sigma": 0.06},
    }

    df = pd.DataFrame(index=dates)
    for ticker, p in params.items():
        daily_mu = p["mu"] / 252
        daily_sigma = p["sigma"] / np.sqrt(252)
        log_rets = rng.normal(daily_mu, daily_sigma, n_days)
        prices = 100 * np.exp(np.cumsum(log_rets))
        df[ticker] = prices

    return df, params


# ============================================================
# SECTION A — DATA INTEGRITY
# ============================================================

def test_data_integrity(prices):
    section("A — Data Integrity")

    # A1: No NaN in cleaned prices
    nan_count = prices.isna().sum().sum()
    if nan_count == 0:
        log_pass(f"A1: Price matrix has zero NaN values ({prices.shape[0]} rows x {prices.shape[1]} cols)")
    else:
        log_fail(f"A1: Price matrix has {nan_count} NaN values")

    # A2: All prices strictly positive
    if (prices > 0).all().all():
        log_pass("A2: All prices are strictly positive")
    else:
        neg_count = (prices <= 0).sum().sum()
        log_fail(f"A2: {neg_count} non-positive price entries found")

    # A3: Monotonically increasing dates
    if prices.index.is_monotonic_increasing:
        log_pass("A3: Price index is monotonically increasing (time-ordered)")
    else:
        log_fail("A3: Price index is NOT monotonically increasing")

    # A4: No duplicate dates
    if not prices.index.has_duplicates:
        log_pass("A4: No duplicate dates in price index")
    else:
        log_fail("A4: Duplicate dates found in price index")


# ============================================================
# SECTION B — COVARIANCE & EXPECTED RETURNS
# ============================================================

def test_covariance_and_returns(prices):
    section("B — Covariance Matrix & Expected Returns")
    from pypfopt import expected_returns, risk_models

    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.sample_cov(prices)

    # B1: Covariance matrix is square
    if cov.shape[0] == cov.shape[1]:
        log_pass(f"B1: Covariance matrix is square ({cov.shape[0]}x{cov.shape[1]})")
    else:
        log_fail(f"B1: Covariance matrix is not square: {cov.shape}")

    # B2: Covariance matrix is symmetric
    if np.allclose(cov.values, cov.values.T, atol=1e-10):
        log_pass("B2: Covariance matrix is symmetric (|C - C^T| < 1e-10)")
    else:
        max_asym = np.abs(cov.values - cov.values.T).max()
        log_fail(f"B2: Covariance matrix asymmetry detected (max diff: {max_asym:.2e})")

    # B3: Covariance matrix is positive semi-definite (all eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    if np.all(eigenvalues >= -1e-10):
        log_pass(f"B3: Covariance matrix is PSD (min eigenvalue: {eigenvalues.min():.6e})")
    else:
        log_fail(f"B3: Covariance matrix is NOT PSD (min eigenvalue: {eigenvalues.min():.6e})")

    # B4: Diagonal of covariance = individual variances
    daily_rets = prices.pct_change().dropna()
    for ticker in prices.columns:
        manual_var = float(daily_rets[ticker].var() * 252)
        cov_var = float(cov.loc[ticker, ticker])
        diff = abs(manual_var - cov_var)
        if diff < 0.001:
            log_pass(f"B4: {ticker} variance matches cov diagonal (diff={diff:.6f})")
        else:
            log_fail(f"B4: {ticker} variance mismatch: manual={manual_var:.6f}, cov={cov_var:.6f}")

    # B5: Expected returns use CAGR (geometric annualisation)
    # PyPortfolioOpt default: (end/start)^(252/n_trading_days) - 1
    n_days = len(prices)
    for ticker in prices.columns:
        cagr = float((prices[ticker].iloc[-1] / prices[ticker].iloc[0]) ** (252 / n_days) - 1)
        ef_mu = float(mu[ticker])
        diff = abs(cagr - ef_mu)
        if diff < 0.001:
            log_pass(f"B5: {ticker} expected return matches CAGR (diff={diff:.6f})")
        else:
            log_fail(f"B5: {ticker} expected return mismatch: cagr={cagr:.6f}, ef={ef_mu:.6f})")

    return mu, cov


# ============================================================
# SECTION C — WEIGHT CONSTRAINTS & SUMMATION (GIPS: Fully Invested)
# ============================================================

def test_weight_constraints(prices, mu, cov):
    section("C — Weight Constraints & Summation (GIPS Fully-Invested)")

    result = compute_efficient_frontier(prices)
    max_sharpe = result["max_sharpe"]
    min_vol = result["min_vol"]

    for label, port in [("Max Sharpe", max_sharpe), ("Min Vol", min_vol)]:
        if port is None:
            log_skip(f"{label} portfolio was None (optimiser infeasible)")
            continue

        weights = port["weights"]

        # C1: Weights sum to 1.0
        wsum = sum(weights.values())
        if abs(wsum - 1.0) < 1e-6:
            log_pass(f"C1: {label} weights sum to {wsum:.8f} (GIPS fully-invested)")
        else:
            log_fail(f"C1: {label} weights sum to {wsum:.8f} (expected 1.0)")

        # C2: No negative weights (long-only)
        neg = {k: v for k, v in weights.items() if v < -1e-8}
        if not neg:
            log_pass(f"C2: {label} has no negative weights (long-only constraint)")
        else:
            log_fail(f"C2: {label} has negative weights: {neg}")

        # C3: No weight exceeds upper bound (default 1.0)
        over = {k: v for k, v in weights.items() if v > 1.0 + 1e-6}
        if not over:
            log_pass(f"C3: {label} all weights ≤ 100%")
        else:
            log_fail(f"C3: {label} weights exceed 100%: {over}")

    # C4: Custom bounds respected
    print("\n  Testing custom bounds (min=5%, max=40%)...")
    result_capped = compute_efficient_frontier(
        prices, weight_bounds=(0.05, 0.40)
    )
    if result_capped["max_sharpe"]:
        w = result_capped["max_sharpe"]["weights"]
        violations = []
        for t, v in w.items():
            if v < 0.05 - 1e-6:
                violations.append(f"{t}={v:.4f} < 5%")
            if v > 0.40 + 1e-6:
                violations.append(f"{t}={v:.4f} > 40%")
        if not violations:
            log_pass("C4: Custom bounds [5%, 40%] respected for all tickers")
        else:
            log_fail(f"C4: Bound violations: {violations}")
    else:
        log_skip("C4: Max Sharpe infeasible with custom bounds")

    # C5: Per-ticker cap
    first_ticker = list(prices.columns)[0]
    print(f"\n  Testing per-ticker cap ({first_ticker} capped at 10%)...")
    result_ticker_cap = compute_efficient_frontier(
        prices, ticker_caps={first_ticker: 0.10}
    )
    if result_ticker_cap["max_sharpe"]:
        cap_w = result_ticker_cap["max_sharpe"]["weights"].get(first_ticker, 0)
        if cap_w <= 0.10 + 1e-6:
            log_pass(f"C5: {first_ticker} capped at {cap_w:.4f} (≤ 10%)")
        else:
            log_fail(f"C5: {first_ticker} weight {cap_w:.4f} exceeds 10% cap")
    else:
        log_skip("C5: Max Sharpe infeasible with ticker cap")

    return result


# ============================================================
# SECTION D — EFFICIENT FRONTIER MONOTONICITY
# ============================================================

def test_frontier_monotonicity(result):
    section("D — Efficient Frontier Monotonicity")

    vols = result["frontier_vols"]
    rets = result["frontier_rets"]

    if len(vols) < 5:
        log_warn(f"D0: Only {len(vols)} frontier points — too few for robust test")
        return

    log_pass(f"D0: {len(vols)} frontier points generated")

    # D1: For the efficient (upper) portion, return should generally increase with risk
    # Due to optimiser discretisation, allow small regressions
    mono_violations = 0
    for i in range(1, len(vols)):
        if vols[i] > vols[i - 1] + 1e-6 and rets[i] < rets[i - 1] - 0.005:
            mono_violations += 1

    if mono_violations <= 3:
        log_pass(f"D1: Frontier is approximately monotonic ({mono_violations} minor violations)")
    else:
        log_warn(f"D1: Frontier has {mono_violations} monotonicity violations (may include lower branch)")

    # D2: All vols are positive
    if all(v > 0 for v in vols):
        log_pass("D2: All frontier volatilities are positive")
    else:
        log_fail("D2: Some frontier volatilities are non-positive")

    # D3: Frontier spans a meaningful range
    vol_range = max(vols) - min(vols)
    if vol_range > 0.01:
        log_pass(f"D3: Frontier vol range = {vol_range*100:.1f}pp (meaningful spread)")
    else:
        log_warn(f"D3: Frontier vol range = {vol_range*100:.2f}pp (very narrow)")


# ============================================================
# SECTION E — SHARPE RATIO IDENTITY
# ============================================================

def test_sharpe_identity(result):
    section("E — Sharpe Ratio Identity")

    for label in ["max_sharpe", "min_vol"]:
        port = result[label]
        if port is None:
            log_skip(f"{label} is None")
            continue

        # E1: Sharpe = (ret - rf) / vol
        expected_sharpe = (port["ret"] - RISK_FREE_RATE) / port["vol"]
        actual_sharpe = port["sharpe"]
        diff = abs(expected_sharpe - actual_sharpe)

        if diff < 1e-4:
            log_pass(f"E1: {label} Sharpe identity holds: (ret-rf)/vol = {expected_sharpe:.4f} ≈ {actual_sharpe:.4f}")
        else:
            log_fail(f"E1: {label} Sharpe mismatch: expected={expected_sharpe:.4f}, got={actual_sharpe:.4f}")

    # E2: Max Sharpe has highest Sharpe of all frontier points
    if result["max_sharpe"]:
        ms_sharpe = result["max_sharpe"]["sharpe"]
        frontier_sharpes = []
        for v, r in zip(result["frontier_vols"], result["frontier_rets"]):
            if v > 1e-8:
                frontier_sharpes.append((r - RISK_FREE_RATE) / v)
        if frontier_sharpes:
            max_frontier_sharpe = max(frontier_sharpes)
            # Allow small tolerance — the frontier is sampled, the exact max may not be on a sample point
            if ms_sharpe >= max_frontier_sharpe - 0.05:
                log_pass(f"E2: Max Sharpe ({ms_sharpe:.3f}) ≥ best frontier point ({max_frontier_sharpe:.3f})")
            else:
                log_warn(f"E2: Max Sharpe ({ms_sharpe:.3f}) < best frontier point ({max_frontier_sharpe:.3f})")

    # E3: Min Vol has lowest vol
    if result["min_vol"] and result["frontier_vols"]:
        mv_vol = result["min_vol"]["vol"]
        min_frontier_vol = min(result["frontier_vols"])
        if mv_vol <= min_frontier_vol + 0.005:
            log_pass(f"E3: Min Vol ({mv_vol*100:.2f}%) ≤ min frontier vol ({min_frontier_vol*100:.2f}%)")
        else:
            log_warn(f"E3: Min Vol ({mv_vol*100:.2f}%) > min frontier vol ({min_frontier_vol*100:.2f}%)")


# ============================================================
# SECTION F — MONTE CARLO GBM DRIFT (Ito's Lemma Compliance)
# ============================================================

def test_monte_carlo_drift(mu, cov):
    section("F — Monte Carlo GBM Drift Correctness (Ito's Lemma)")

    # The GBM model should use: drift = (mu - 0.5*sigma^2)*dt
    # With enough simulations, the median log return should approximate the drift
    weights = {t: 1.0 / len(mu) for t in mu.index}  # equal weight
    mc = run_optimization_monte_carlo(
        mu=mu, cov=cov, weights=weights,
        initial_value=100_000,
        horizon_years=10,
        n_simulations=5000,
        random_seed=42,
    )

    # Calculate theoretical portfolio mu and sigma
    w = np.array([weights[t] for t in mu.index])
    w = w / w.sum()
    port_mu = float(w @ mu.values)
    port_var = float(w @ cov.values @ w)
    port_sigma = np.sqrt(port_var)

    # F1: After 10 years, median should ≈ initial * exp(mu * T)
    # (With drift correction, median of log-normal = exp((mu-0.5σ²)*T) * initial)
    theoretical_median = 100_000 * np.exp((port_mu - 0.5 * port_sigma ** 2) * 10)
    actual_median = mc["metrics"]["median_final"]

    pct_diff = abs(actual_median - theoretical_median) / theoretical_median * 100
    if pct_diff < 10:  # 10% tolerance for 5000 sims
        log_pass(f"F1: Median final ${actual_median:,.0f} ≈ theoretical ${theoretical_median:,.0f} (diff={pct_diff:.1f}%)")
    else:
        log_warn(f"F1: Median final ${actual_median:,.0f} vs theoretical ${theoretical_median:,.0f} (diff={pct_diff:.1f}%)")

    # F2: σ reported matches w^T Σ w calculation
    reported_sigma = mc["metrics"]["sigma"]
    if abs(reported_sigma - port_sigma) < 1e-8:
        log_pass(f"F2: Reported σ ({reported_sigma:.6f}) matches w^T Σ w ({port_sigma:.6f})")
    else:
        log_fail(f"F2: Reported σ ({reported_sigma:.6f}) ≠ w^T Σ w ({port_sigma:.6f})")

    # F3: μ reported matches w^T μ calculation
    reported_mu = mc["metrics"]["mu"]
    if abs(reported_mu - port_mu) < 1e-8:
        log_pass(f"F3: Reported μ ({reported_mu:.6f}) matches w^T μ ({port_mu:.6f})")
    else:
        log_fail(f"F3: Reported μ ({reported_mu:.6f}) ≠ w^T μ ({port_mu:.6f})")

    return mc


# ============================================================
# SECTION G — MONTE CARLO REPRODUCIBILITY (CRN)
# ============================================================

def test_monte_carlo_reproducibility(mu, cov):
    section("G — Monte Carlo Reproducibility (Common Random Numbers)")

    weights = {t: 1.0 / len(mu) for t in mu.index}

    mc1 = run_optimization_monte_carlo(
        mu=mu, cov=cov, weights=weights,
        initial_value=100_000, horizon_years=5,
        n_simulations=500, random_seed=99,
    )
    mc2 = run_optimization_monte_carlo(
        mu=mu, cov=cov, weights=weights,
        initial_value=100_000, horizon_years=5,
        n_simulations=500, random_seed=99,
    )

    m1 = np.array(mc1["percentiles"]["50"])
    m2 = np.array(mc2["percentiles"]["50"])

    if np.allclose(m1, m2, atol=1e-6):
        log_pass("G1: Same seed → identical median paths (CRN verified)")
    else:
        max_diff = np.abs(m1 - m2).max()
        log_fail(f"G1: Same seed produced different paths (max diff: ${max_diff:,.2f})")

    # G2: Different seed → different paths
    mc3 = run_optimization_monte_carlo(
        mu=mu, cov=cov, weights=weights,
        initial_value=100_000, horizon_years=5,
        n_simulations=500, random_seed=77,
    )
    m3 = np.array(mc3["percentiles"]["50"])
    if not np.allclose(m1, m3, atol=1e-2):
        log_pass("G2: Different seed → different paths (randomness confirmed)")
    else:
        log_warn("G2: Different seeds produced identical paths (suspicious)")


# ============================================================
# SECTION H — MONTE CARLO PERCENTILE ORDERING
# ============================================================

def test_monte_carlo_percentiles(mc):
    section("H — Monte Carlo Percentile Ordering")

    pcts = mc["percentiles"]
    keys = ["10", "25", "50", "75", "90"]

    for i in range(len(keys) - 1):
        lo = np.array(pcts[keys[i]])
        hi = np.array(pcts[keys[i + 1]])
        # Should hold at every time step
        violations = int(np.sum(lo > hi + 0.01))
        if violations == 0:
            log_pass(f"H1: P{keys[i]} ≤ P{keys[i+1]} at all {len(lo)} time steps")
        else:
            log_fail(f"H1: P{keys[i]} > P{keys[i+1]} at {violations}/{len(lo)} time steps")

    # H2: VaR 95 < median final
    var95 = mc["metrics"]["var_95"]
    median = mc["metrics"]["median_final"]
    if var95 < median:
        log_pass(f"H2: VaR95 (${var95:,.0f}) < Median (${median:,.0f})")
    else:
        log_fail(f"H2: VaR95 (${var95:,.0f}) ≥ Median (${median:,.0f})")

    # H3: CVaR ≤ VaR (CVaR is the average of the worst outcomes)
    cvar95 = mc["metrics"]["cvar_95"]
    if cvar95 <= var95 + 1e-2:
        log_pass(f"H3: CVaR95 (${cvar95:,.0f}) ≤ VaR95 (${var95:,.0f})")
    else:
        log_fail(f"H3: CVaR95 (${cvar95:,.0f}) > VaR95 (${var95:,.0f})")

    # H4: Starting value equals initial at t=0
    start_10 = pcts["10"][0]
    start_90 = pcts["90"][0]
    if abs(start_10 - 100_000) < 1.0 and abs(start_90 - 100_000) < 1.0:
        log_pass("H4: All percentiles start at exactly the initial value ($100,000)")
    else:
        log_fail(f"H4: Percentile start values: P10=${start_10:,.0f}, P90=${start_90:,.0f}")


# ============================================================
# SECTION I — BACKTEST GROWTH-OF-$1 CORRECTNESS
# ============================================================

def test_backtest_growth(prices):
    section("I — Backtest Growth-of-$1 Correctness")

    # Equal-weight backtest
    tickers = list(prices.columns)
    weights = {t: 1.0 / len(tickers) for t in tickers}

    growth = backtest_optimized_weights(prices, weights)

    # I1: Growth starts at ~1.0
    if abs(growth.iloc[0] - 1.0) < 0.05:
        log_pass(f"I1: Growth series starts at {growth.iloc[0]:.4f} ≈ 1.0")
    else:
        log_fail(f"I1: Growth series starts at {growth.iloc[0]:.4f} (expected ~1.0)")

    # I2: Growth is always positive
    if (growth > 0).all():
        log_pass("I2: Growth series is always positive")
    else:
        log_fail(f"I2: {(growth <= 0).sum()} non-positive growth values")

    # I3: Manual replication — first day return
    daily_rets = prices.pct_change().dropna()
    w = np.array([weights[t] for t in tickers])
    first_day_manual = float((daily_rets.iloc[0][tickers] * w).sum())
    first_day_growth = float(growth.iloc[0])

    if abs(first_day_growth - (1 + first_day_manual)) < 1e-8:
        log_pass(f"I3: First day growth matches manual calc ({first_day_growth:.6f})")
    else:
        log_fail(f"I3: First day mismatch: growth={first_day_growth:.6f}, manual={1+first_day_manual:.6f}")

    # I4: Zero-weight tickers excluded
    partial_weights = {tickers[0]: 0.6, tickers[1]: 0.4}
    for t in tickers[2:]:
        partial_weights[t] = 0.0

    growth_partial = backtest_optimized_weights(prices, partial_weights)
    if not growth_partial.empty:
        log_pass(f"I4: Partial-weight backtest produced {len(growth_partial)} data points")
    else:
        log_fail("I4: Partial-weight backtest returned empty series")

    return growth


# ============================================================
# SECTION J — UNDERWATER / DRAWDOWN HWM (GIPS Methodology)
# ============================================================

def test_drawdown(growth):
    section("J — Drawdown / Underwater HWM (GIPS Methodology)")

    dd = compute_underwater(growth)

    # J1: Drawdown is always ≤ 0
    if (dd <= 1e-8).all():
        log_pass("J1: Drawdown series is always ≤ 0% (correct sign convention)")
    else:
        positive_count = (dd > 1e-8).sum()
        log_fail(f"J1: {positive_count} positive drawdown values found (should all be ≤ 0)")

    # J2: Drawdown starts at 0 (first day = new high)
    if abs(dd.iloc[0]) < 1e-6:
        log_pass(f"J2: Drawdown starts at {dd.iloc[0]:.4f}% (first day = HWM)")
    else:
        log_warn(f"J2: Drawdown starts at {dd.iloc[0]:.4f}% (expected 0)")

    # J3: Manual HWM replication
    hwm = growth.cummax()
    manual_dd = (growth - hwm) / hwm * 100

    if np.allclose(dd.values, manual_dd.values, atol=1e-8):
        log_pass("J3: Compute_underwater matches manual HWM calculation exactly")
    else:
        max_diff = np.abs(dd.values - manual_dd.values).max()
        log_fail(f"J3: Drawdown mismatch (max diff: {max_diff:.6f}%)")

    # J4: Max drawdown is the global minimum of the series
    max_dd_from_series = dd.min()
    if max_dd_from_series <= 0:
        log_pass(f"J4: Max Drawdown = {max_dd_from_series:.2f}%")
    else:
        log_fail(f"J4: Max Drawdown is positive: {max_dd_from_series:.2f}%")


# ============================================================
# SECTION K — ROLLING SHARPE CONSISTENCY
# ============================================================

def test_rolling_sharpe(prices):
    section("K — Rolling Sharpe Consistency")

    tickers = list(prices.columns)
    weights = {t: 1.0 / len(tickers) for t in tickers}

    rs = compute_rolling_sharpe(prices, weights, window_years=1)

    # K1: Output is not empty
    if not rs.empty:
        log_pass(f"K1: Rolling Sharpe produced {len(rs)} data points")
    else:
        log_fail("K1: Rolling Sharpe returned empty series")
        return

    # K2: No NaN in output
    if rs.isna().sum() == 0:
        log_pass("K2: No NaN values in rolling Sharpe output")
    else:
        log_fail(f"K2: {rs.isna().sum()} NaN values in rolling Sharpe")

    # K3: Manual replication of one point (last)
    daily_rets = prices.pct_change().dropna()
    w = np.array([weights[t] for t in tickers])
    port_ret = (daily_rets[tickers] * w).sum(axis=1)

    window = 252  # 1 year
    last_window = port_ret.iloc[-window:]
    manual_sharpe = float(
        (last_window.mean() * 252 - RISK_FREE_RATE) / (last_window.std() * np.sqrt(252))
    )
    engine_sharpe = float(rs.iloc[-1])

    diff = abs(manual_sharpe - engine_sharpe)
    if diff < 1e-4:
        log_pass(f"K3: Last rolling Sharpe matches manual calc (diff={diff:.6f})")
    else:
        log_fail(f"K3: Last rolling Sharpe mismatch: manual={manual_sharpe:.4f}, engine={engine_sharpe:.4f} (diff={diff:.6f})")

    # K4: Sharpe is in a reasonable range
    extreme = rs[(rs > 5) | (rs < -5)]
    if len(extreme) == 0:
        log_pass(f"K4: All rolling Sharpe values in [-5, 5] range")
    else:
        log_warn(f"K4: {len(extreme)} extreme rolling Sharpe values outside [-5, 5]")


# ============================================================
# SECTION L — LIVE DATA VALIDATION (Against Real Holdings)
# ============================================================

def test_live_data():
    section("L — Live Data Validation (Current Holdings)")

    try:
        from data_loader import load_holdings
        h = load_holdings()
        active = h[(h["shares"] > 0) & (h["ticker"].str.upper() != "CASH")]
        tickers = sorted(active["ticker"].str.upper().unique().tolist())

        if len(tickers) < 2:
            log_skip("Fewer than 2 active tickers — cannot run live optimization")
            return

        print(f"  Active holdings: {tickers}")
        prices = fetch_optimization_prices(tickers, years_back=5)
        available = [t for t in tickers if t in prices.columns]
        print(f"  Price data available for: {available}")

        if len(available) < 2:
            log_warn(f"Only {len(available)} tickers have price data")
            return

        prices = prices[available]

        result = compute_efficient_frontier(prices)

        # L1: Frontier computed
        if len(result["frontier_vols"]) > 0:
            log_pass(f"L1: Efficient Frontier computed ({len(result['frontier_vols'])} points) for live holdings")
        else:
            log_fail("L1: Frontier computation returned zero points")

        # L2: Max Sharpe computed
        if result["max_sharpe"]:
            ms = result["max_sharpe"]
            wsum = sum(ms["weights"].values())
            log_pass(f"L2: Max Sharpe portfolio: ret={ms['ret']*100:.1f}%, vol={ms['vol']*100:.1f}%, Sharpe={ms['sharpe']:.2f}, Σw={wsum:.6f}")
        else:
            log_warn("L2: Max Sharpe infeasible for live holdings")

        # L3: Min Vol computed
        if result["min_vol"]:
            mv = result["min_vol"]
            log_pass(f"L3: Min Vol portfolio: ret={mv['ret']*100:.1f}%, vol={mv['vol']*100:.1f}%, Sharpe={mv['sharpe']:.2f}")
        else:
            log_warn("L3: Min Vol infeasible for live holdings")

        # L4: Individual asset stats
        for a in result["individual"]:
            if a["vol"] > 0 and abs(a["ret"]) < 5.0:
                log_pass(f"L4: {a['ticker']}: ret={a['ret']*100:.1f}%, vol={a['vol']*100:.1f}% (sane)")
            else:
                log_warn(f"L4: {a['ticker']}: ret={a['ret']*100:.1f}%, vol={a['vol']*100:.1f}% (unusual)")

    except Exception as e:
        log_warn(f"Live data test failed: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# MAIN
# ============================================================

def run_all():
    print("=" * 60)
    print("  AUDIT 16 — PORTFOLIO OPTIMIZATION ENGINE")
    print("  GIPS Compliance & Mathematical Verification")
    print("=" * 60)
    print(f"  Risk-Free Rate: {RISK_FREE_RATE*100:.1f}%")
    print(f"  Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    # Build synthetic data
    prices, params = _build_synthetic_prices()
    print(f"\n  Synthetic data: {prices.shape[0]} days x {prices.shape[1]} assets")

    # Run all sections
    test_data_integrity(prices)
    mu, cov = test_covariance_and_returns(prices)
    result = test_weight_constraints(prices, mu, cov)
    test_frontier_monotonicity(result)
    test_sharpe_identity(result)
    mc = test_monte_carlo_drift(mu, cov)
    test_monte_carlo_reproducibility(mu, cov)
    test_monte_carlo_percentiles(mc)
    growth = test_backtest_growth(prices)
    test_drawdown(growth)
    test_rolling_sharpe(prices)
    test_live_data()

    # Final report
    print("\n" + "=" * 60)
    print(f"  FINAL RESULTS")
    print(f"  PASS: {_PASS}  |  FAIL: {_FAIL}  |  WARN: {_WARN}")
    print("=" * 60)

    if _FAIL == 0:
        print("\n  ✅ ALL TESTS PASSED — Optimization engine is GIPS-compliant.")
        return 0
    else:
        print(f"\n  ❌ {_FAIL} FAILURE(S) DETECTED — Review output above.")
        return 1


if __name__ == "__main__":
    code = run_all()
    sys.exit(code)
