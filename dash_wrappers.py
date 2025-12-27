import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import defaultdict

# Import existing modules
from portfolio_engine import (
    run_engine,
    calculate_horizon_pl,
    calculate_ticker_pl,
    calculate_asset_class_pl,
    compute_drawdown_series
)
from data_loader import (
    load_holdings, 
    load_cashflows_external, 
    load_transactions_raw, 
    fetch_price_history,
    load_dividends,
    fetch_etf_sectors
)
from financial_math import (
    get_portfolio_horizon_start,
    compute_period_twr,
    fv_lump,
    fv_contrib,
    modified_dietz_for_ticker_window
)
from report_formatting import fmt_pct_clean, fmt_dollar_clean
import config
from config import TARGET_MONTHLY_CONTRIBUTION, GLOBAL_PALETTE, RISK_FREE_RATE

# ============================================================
# GLOBAL DATA CACHE (Server-Side)
# ============================================================
_DATA_CACHE = None

ASSET_CLASS_PROXIES = {
    "US Large Cap": "SPY",
    "US Growth": "QQQ", 
    "US Small Cap": "IWM",
    "International Equity": "VXUS",
    "Fixed Income": "BND",
    "US Bonds": "BND",
    "Gold / Precious Metals": "GLD",
    "Digital Assets": "BTC-USD"
}

def get_data():
    """Retrieve cached data, initializing if necessary."""
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = run_analytics_engine()
    return _DATA_CACHE

def refresh_data(end_date=None):
    """Force refresh of the data cache."""
    global _DATA_CACHE
    _DATA_CACHE = run_analytics_engine(end_date=end_date)
    return _DATA_CACHE

# ============================================================
# CORE: Run Engine Wrapper
# ============================================================
def run_analytics_engine(end_date=None):
    """
    Runs the core portfolio engine and returns the raw dataframes.
    This should be called on app startup and file upload.
    """
    # Run the base engine
    twr_df, sec_table, class_df, pv, twr_si, twr_si_annualized, pl_si = run_engine(end_date=end_date)
    
    # Load other raw data needed for charts/tables
    cf_ext = load_cashflows_external()
    tx_raw = load_transactions_raw()
    holdings = load_holdings()
    dividends = load_dividends()

    # Time Machine Clipping for Helpers
    if end_date is not None:
        end_date_ts = pd.Timestamp(end_date)
        cf_ext = cf_ext[cf_ext["date"] <= end_date_ts]
        tx_raw = tx_raw[tx_raw["date"] <= end_date_ts]
        dividends = dividends[dividends["date"] <= end_date_ts]
    
    # Calculate true inception (same logic as engine)
    dates = []
    if not cf_ext.empty: dates.append(cf_ext["date"].min())
    if not tx_raw.empty: dates.append(tx_raw["date"].min())
    if not pv.empty: dates.append(pv.index.min())
    
    inception_date = min(dates) if dates else pd.Timestamp.now()
    
    # Create Filtered Current Table
    # Filter for > 0 value or shares (shares > 1e-6)
    sec_table_current = sec_table[sec_table["shares"].abs() > 1e-6].copy()

    # Dynamic Sector Dataframe
    sector_df = _prepare_sector_df(sec_table_current)
    
    # Pre-fetch prices ONCE to avoid multiple API calls and ensure consistency
    all_tickers = sec_table[sec_table["ticker"] != "CASH"]["ticker"].unique().tolist()
    
    # Ensure S&P 500 is always available for benchmarks and proxies
    if "SPY" not in all_tickers: all_tickers.append("SPY")
    
    # Ensure all Proxies are available
    for proxy in ASSET_CLASS_PROXIES.values():
        if proxy not in all_tickers: all_tickers.append(proxy)
        
    prices_cached = fetch_price_history(all_tickers) if all_tickers else pd.DataFrame()
    
    # Robustly extract errors from dataframe metadata
    errors = getattr(prices_cached, "attrs", {}).get("errors", [])
    if errors: print(f"DEBUG: dash_wrappers found errors: {errors}")

    # Dynamic Risk Profile (Vol, Return, Correlation)
    dynamic_risk_return, dynamic_corr_matrix = _calculate_dynamic_risk_profile(
        prices_cached, sec_table_current, holdings, end_date
    )
    
    return {
        "twr_df": twr_df,
        "sec_table": sec_table,
        "sec_table_current": sec_table_current,
        "class_df": class_df,
        "pv": pv,
        "twr_si": twr_si,
        "twr_si_ann": twr_si_annualized,
        "pl_si": pl_si,
        "cf_ext": cf_ext,
        "tx_raw": tx_raw,
        "holdings": holdings,
        "dividends": dividends,
        "inception_date": inception_date,
        "sector_df": sector_df,
        "prices": prices_cached,
        "errors": errors,
        "risk_return": dynamic_risk_return,
        "correlation_matrix": dynamic_corr_matrix
    }

def _prepare_sector_df(sec_table):
    """Internal helper to build sector allocation dataframe from Dynamic Fetcher."""
    if sec_table.empty: return pd.DataFrame()
    
    sector_exposure = defaultdict(float)
    SECTOR_NORMALIZATION = {
        "Comm Services": "Communication Services",
        "Consumer Disc.": "Consumer Discretionary",
        "Information Technology": "Tech",
        "Other": None,
    }
    
    # Use raw market values, EXCLUDING CASH for the denominator
    # to show allocation of INVESTED capital
    sector_universe = sec_table[sec_table["ticker"] != "CASH"].copy()
    sector_universe = sector_universe[sector_universe["market_value"] > 0]
    
    total_invested = sector_universe["market_value"].sum()
    
    if total_invested > 0:
        for _, row in sector_universe.iterrows():
            ticker = row["ticker"]
            weight_pct = (row["market_value"] / total_invested) * 100.0
            
            # Dynamic Fetch
            etf_sectors = fetch_etf_sectors(ticker)
            
            for sector, pct in etf_sectors.items():
                norm_sector = SECTOR_NORMALIZATION.get(sector, sector)
                if norm_sector is None: continue
                sector_exposure[norm_sector] += weight_pct * pct / 100.0
                
    sector_df = pd.DataFrame(
        list(sector_exposure.items()),
        columns=["Sector", "Exposure"]
    ).sort_values("Exposure", ascending=True)
    
    return sector_df

def _calculate_dynamic_risk_profile(prices, sec_table, holdings, end_date=None):
    """
    Calculates Realized Volatility, TTM Return, and Asset Class Correlation Matrix
    based on the portfolio's actual 10-year history (Pro-Forma).
    """
    if prices.empty or sec_table.empty:
        return {}, {}
        
    # 1. Clip Prices to End Date (Time Machine)
    if end_date:
        prices = prices[prices.index <= pd.Timestamp(end_date)]
        
    daily_rets = prices.pct_change()
    
    # 2. Identify Asset Classes and Tickers
    # Use sec_table for weights (current composition)
    ac_weights = defaultdict(list) # {AC: [(Ticker, Weight), ...]}
    
    total_value = sec_table[sec_table["ticker"] != "CASH"]["market_value"].sum()
    if total_value <= 0: total_value = 1.0
    
    for _, row in sec_table.iterrows():
        t = row["ticker"]
        if t == "CASH": continue
        ac = row["asset_class"]
        w = row["market_value"] / total_value
        ac_weights[ac].append((t, w))
        
    # 3. Construct Asset Class Daily Return Series
    ac_daily_series = pd.DataFrame(index=daily_rets.index)
    
    for ac, items in ac_weights.items():
        tickers = [x[0] for x in items if x[0] in daily_rets.columns]
        weights = pd.Series({x[0]: x[1] for x in items if x[0] in daily_rets.columns})
        
        if not tickers:
            # Entire AC missing from prices? Use Proxy
            proxy = ASSET_CLASS_PROXIES.get(ac, "SPY")
            if proxy in daily_rets.columns:
                ac_daily_series[ac] = daily_rets[proxy]
            continue
            
        # Weighted Average Return (Renormalizing for missing data on specific days)
        # Select returns for these tickers
        t_rets = daily_rets[tickers]
        
        # Calculate weighted sum of available returns
        # Numerator: sum(w_i * r_i)
        numer = t_rets.multiply(weights).sum(axis=1)
        
        # Denominator: sum(w_i) for available r_i
        # Mask NaNs in returns
        valid_mask = t_rets.notna()
        denom = valid_mask.multiply(weights).sum(axis=1)
        
        # Handle days where NO tickers have data (denom=0)
        ac_series = numer.div(denom)
        
        # Gap Filling: If day is NaN, fill with Proxy return
        if ac_series.isna().any():
            proxy = ASSET_CLASS_PROXIES.get(ac, "SPY")
            if proxy in daily_rets.columns:
                ac_series = ac_series.fillna(daily_rets[proxy])
            else:
                ac_series = ac_series.fillna(0.0) # Last resort
                
        ac_daily_series[ac] = ac_series
        
    # 4. Calculate Correlation Matrix (Asset Class Level)
    # Pairwise deletion is automatic in pandas corr()
    corr_matrix_df = ac_daily_series.corr()
    
    # Convert to nested dict for consumption
    # Use to_dict() directly? Need {AC: {AC: val}}
    dynamic_corr_matrix = corr_matrix_df.to_dict()
    
    # 5. Calculate Risk/Return Metrics (Realized)
    dynamic_risk_return = {}
    
    for ac in ac_daily_series.columns:
        series = ac_daily_series[ac].dropna()
        if series.empty: continue
        
        # Volatility (Annualized Std Dev)
        vol = series.std() * np.sqrt(252) * 100.0
        
        # TTM Return (Last 252 trading days)
        # Or full history CAGR? Prompt says "TTM performance"
        if len(series) >= 252:
            recent = series.tail(252)
            ttm_ret = ((1 + recent).prod() - 1.0) * 100.0
        else:
            # Annualize available history
            days = len(series)
            total_ret = (1 + series).prod() - 1.0
            if days > 20:
                ttm_ret = ((1 + total_ret) ** (252/days) - 1.0) * 100.0
            else:
                ttm_ret = total_ret * 100.0 # Too short to annualize safely
                
        dynamic_risk_return[ac] = {
            "return": ttm_ret,
            "vol": vol
        }
        
    # Add Fixed Benchmarks if missing (for gauge stability)
    if "Fixed Income" not in dynamic_risk_return:
        dynamic_risk_return["Fixed Income"] = {"return": 4.0, "vol": 5.0}
    if "US Large Cap" not in dynamic_risk_return:
         dynamic_risk_return["US Large Cap"] = {"return": 10.0, "vol": 15.0}

    return dynamic_risk_return, dynamic_corr_matrix

def _get_daily_twr_curve(data):
    """
    Helper to generate the Daily TWR Curve (Growth of $1) for the portfolio.
    Used by charts and risk metrics to ensure geometric consistency.
    """
    pv = data["pv"]
    cf_ext = data.get("cf_ext")
    
    if pv.empty: return pd.Series(dtype=float)
    
    # 1. Align Flows
    start_date = pv.index.min()
    flows_daily = cf_ext.groupby("date")["amount"].sum() if cf_ext is not None else pd.Series(dtype=float)
    
    curve_data = {}
    
    # 2. Handle Day 1 (Funding)
    flows_on_start = flows_daily.get(start_date, 0.0)
    if flows_on_start > 0:
        pv_day1 = pv.iloc[0]
        # Return = (End - Funding) / Funding
        r_0 = (pv_day1 - flows_on_start) / flows_on_start
        curve_data[start_date] = 1.0 + r_0
        running = 1.0 + r_0
    else:
        curve_data[start_date] = 1.0
        running = 1.0
        
    # 3. Chain Daily Returns
    pv_dates = pv.index
    for i in range(1, len(pv_dates)):
        d0 = pv_dates[i-1]
        d1 = pv_dates[i]
        
        flow = flows_daily.get(d1, 0.0)
        denom = pv.loc[d0] + flow
        
        if denom <= 0:
            R = 0.0
        else:
            R = (pv.loc[d1] - denom) / denom
            
        running *= (1 + R)
        curve_data[d1] = running
        
    return pd.Series(curve_data).sort_index()

def calculate_efficiency_metrics(twr_series):
    """
    Calculates Sharpe and Sortino Ratios based on daily TWR series.
    Uses RISK_FREE_RATE from config.
    """
    if twr_series.empty or len(twr_series) < 2:
        return {"sharpe": "N/A", "sortino": "N/A"}
        
    # Calculate Daily Returns from the Curve
    daily_rets = twr_series.pct_change().dropna()
    
    if daily_rets.empty:
        return {"sharpe": "N/A", "sortino": "N/A"}
    
    # Annualize Risk Free Rate for daily subtraction
    # (1 + r_annual)^(1/252) - 1
    rf_daily = (1 + RISK_FREE_RATE) ** (1/252) - 1
    
    # Excess Returns
    excess_rets = daily_rets - rf_daily
    
    # Annualized Mean Excess Return
    # Geometric mean is more accurate for long horizons, but arithmetic is standard for Sharpe
    mean_excess = excess_rets.mean() * 252
    
    # 1. Sharpe Ratio
    std_dev = daily_rets.std() * np.sqrt(252)
    sharpe = mean_excess / std_dev if std_dev > 0 else 0.0
    
    # 2. Sortino Ratio
    # Downside Deviation: Std Dev of NEGATIVE returns only (relative to MAR=0 or MAR=RiskFree?)
    # Standard Sortino uses MAR = Risk Free Rate.
    # So we look at variability of (R - Rf) where (R - Rf) < 0.
    downside_rets = excess_rets[excess_rets < 0]
    
    if downside_rets.empty:
        sortino = 100.0 # Infinite/High
    else:
        # Calculate downside deviation (root mean squared downside)
        downside_dev = np.sqrt((downside_rets ** 2).mean()) * np.sqrt(252)
        sortino = mean_excess / downside_dev if downside_dev > 0 else 0.0
        
    return {
        "sharpe": sharpe,
        "sortino": sortino
    }

def calculate_active_metrics(data, benchmark_ticker="SPY"):
    """
    Calculates Beta and Tracking Error vs Benchmark.
    """
    twr_curve = _get_daily_twr_curve(data)
    if twr_curve.empty: return {"beta": "N/A", "te": "N/A"}
    
    # Get Benchmark Prices
    prices = fetch_price_history([benchmark_ticker], use_adj_close=True)
    if benchmark_ticker not in prices.columns:
        return {"beta": "N/A", "te": "N/A"}
        
    bm_series = prices[benchmark_ticker].dropna()
    
    # Align Dates
    common_idx = twr_curve.index.intersection(bm_series.index)
    if len(common_idx) < 30: # Need some history
        return {"beta": "N/A", "te": "N/A"}
        
    port_rets = twr_curve.loc[common_idx].pct_change().dropna()
    bm_rets = bm_series.loc[common_idx].pct_change().dropna()
    
    # Re-align after pct_change (drops first)
    valid_idx = port_rets.index.intersection(bm_rets.index)
    
    y = port_rets.loc[valid_idx]
    x = bm_rets.loc[valid_idx]
    
    if len(y) < 20: return {"beta": "N/A", "te": "N/A"}
    
    # 1. Beta = Cov(P, B) / Var(B)
    covariance = np.cov(y, x)[0][1]
    variance = np.var(x)
    beta = covariance / variance if variance > 0 else 1.0
    
    # 2. Tracking Error = StdDev(P - B) * sqrt(252)
    active_rets = y - x
    te = active_rets.std() * np.sqrt(252) * 100.0 # Percentage
    
    return {
        "beta": beta,
        "te": te
    }

# ============================================================
# DATA HELPERS (METRICS & TABLES)
# ============================================================

def get_snapshot_metrics(data):
    """Returns top-level KPI metrics including Risk Efficiency."""
    pv = data["pv"]
    current_mv = pv.iloc[-1] if not pv.empty else 0.0
    
    twr_si = data["twr_si_ann"] if pd.notna(data["twr_si_ann"]) else data["twr_si"]
    
    # MTD Return
    twr_df = data["twr_df"]
    mtd_row = twr_df[twr_df["Horizon"] == "MTD"]
    mtd_ret = mtd_row["Return"].iloc[0] if not mtd_row.empty else 0.0
    
    # Calculate Efficiency Scores (Sharpe/Sortino)
    twr_curve = _get_daily_twr_curve(data)
    eff = calculate_efficiency_metrics(twr_curve)
    
    return {
        "current_mv": current_mv,
        "twr_si": twr_si,
        "pl_si": data["pl_si"],
        "mtd_ret": mtd_ret,
        "sharpe": eff["sharpe"],
        "sortino": eff["sortino"]
    }

def get_horizon_analysis(data):
    """
    Returns DataFrame for Horizon Analysis (Return %, P/L $, Sharpe, Sortino).
    Replicates the 'Portfolio Snapshot' logic.
    """
    twr_df = data["twr_df"]
    pv = data["pv"]
    inception_date = data["inception_date"]
    cf_ext = data["cf_ext"]
    pl_si = data["pl_si"]
    twr_si = data["twr_si"]
    twr_si_ann = data["twr_si_ann"]
    
    # Get Full TWR Curve for slicing
    twr_curve_full = _get_daily_twr_curve(data)
    
    horizons = ["1D", "1W", "MTD", "1M", "3M", "6M", "1Y"]
    
    snap_map = {row["Horizon"]: row["Return"] for _, row in twr_df.iterrows()}
    
    as_of = pv.index.max()
    
    rows = []
    for h in horizons:
        ret = snap_map.get(h, np.nan)
        
        # Calculate PL components locally to expose meta data
        # Logic matches calculate_horizon_pl in portfolio_engine.py
        start = get_portfolio_horizon_start(pv, inception_date, h)
        
        pl_val = np.nan
        mv_start = 0.0
        mv_end = 0.0
        net_flows = 0.0
        sharpe = "N/A"
        sortino = "N/A"
        
        if start is not None and start < as_of:
            # Map start to pv index
            if start not in pv.index:
                pv_idx = pv.index.sort_values()
                pos = pv_idx.searchsorted(start)
                if pos < len(pv_idx):
                    start = pv_idx[pos]
            
            if start in pv.index:
                mv_start = float(pv.loc[start])
                mv_end = float(pv.loc[as_of])
                
                # Flows
                if cf_ext is not None and not cf_ext.empty:
                    mask = (cf_ext["date"] > start) & (cf_ext["date"] <= as_of)
                    net_flows = float(cf_ext.loc[mask, "amount"].sum())
                
                pl_val = mv_end - mv_start - net_flows
                
            # Calculate Risk Metrics for this window
            # Slice TWR Curve: start to as_of
            if not twr_curve_full.empty:
                curve_slice = twr_curve_full[
                    (twr_curve_full.index >= start) & 
                    (twr_curve_full.index <= as_of)
                ]
                # Only calculate if we have sufficient data points (>10 days to be meaningful)
                if len(curve_slice) > 10:
                    eff = calculate_efficiency_metrics(curve_slice)
                    sharpe = eff["sharpe"]
                    sortino = eff["sortino"]

        rows.append({
            "Horizon": h,
            "Return": ret,
            "P/L": pl_val,
            "Sharpe": sharpe,
            "Sortino": sortino,
            # Audit Meta Columns
            f"meta_Return_start": mv_start,
            f"meta_Return_end": mv_end,
            f"meta_Return_flow": net_flows,
            f"meta_Return_inc": 0.0, # Portfolio level income tricky to separate here
            f"meta_Return_denom": mv_start + net_flows, # Approximation for display
            
            f"meta_P/L_start": mv_start,
            f"meta_P/L_end": mv_end,
            f"meta_P/L_flow": net_flows,
            f"meta_P/L_inc": 0.0 # Included in PL but not separated
        })
        
    # SI Row
    # Recalculate SI components
    si_start = inception_date
    if si_start not in pv.index:
        pv_idx = pv.index.sort_values()
        pos = pv_idx.searchsorted(si_start)
        if pos < len(pv_idx): si_start = pv_idx[pos]
        
    # GIPS COMPLIANCE FIX: Handle Day 1 Logic
    if si_start == inception_date:
        si_mv_start = 0.0
        # Use >= to CAPTURE the Day 1 funding flow
        if cf_ext is not None and not cf_ext.empty:
            mask = (cf_ext["date"] >= si_start) & (cf_ext["date"] <= as_of)
            si_flows = float(cf_ext.loc[mask, "amount"].sum())
        else:
            si_flows = 0.0
    else:
        # Standard Horizon (Start Value is previous close)
        si_mv_start = float(pv.loc[si_start]) if si_start in pv.index else 0.0
        # Use > to exclude the capital that established the start value
        if cf_ext is not None and not cf_ext.empty:
            mask = (cf_ext["date"] > si_start) & (cf_ext["date"] <= as_of)
            si_flows = float(cf_ext.loc[mask, "amount"].sum())
        else:
            si_flows = 0.0
    
    si_mv_end = float(pv.loc[as_of])
        
    si_ret = twr_si_ann if pd.notna(twr_si_ann) else twr_si
    
    # SI Risk Metrics
    eff_si = calculate_efficiency_metrics(twr_curve_full)
    
    rows.append({
        "Horizon": "Since Inception",
        "Return": si_ret,
        "P/L": pl_si,
        "Sharpe": eff_si["sharpe"],
        "Sortino": eff_si["sortino"],
        
        f"meta_Return_start": si_mv_start,
        f"meta_Return_end": si_mv_end,
        f"meta_Return_flow": si_flows,
        f"meta_Return_inc": 0.0,
        f"meta_Return_denom": si_mv_start + si_flows,
        
        f"meta_P/L_start": si_mv_start,
        f"meta_P/L_end": si_mv_end,
        f"meta_P/L_flow": si_flows,
        f"meta_P/L_inc": 0.0
    })
    
    return pd.DataFrame(rows)

def get_ticker_pl_df(data, horizon="SI"):
    """
    Computes ticker-level P/L for a specific horizon.
    Used for 'Performance Highlights' and detailed tables.
    Now includes meta columns for Audit Trail.
    """
    pv = data["pv"]
    inception_date = data["inception_date"]
    sec_table = data["sec_table"]
    tx_raw = data["tx_raw"]
    dividends = data["dividends"]
    prices = data["prices"]  # Use cached prices for consistency
    
    if prices.empty:
        return pd.DataFrame()
    
    # Merge target_pct if not in sec_table
    # (Engine usually puts it there, but let's be safe)
    
    as_of = pv.index.max()
    if horizon == "SI":
        raw_start = None
    else:
        raw_start = get_portfolio_horizon_start(pv, inception_date, horizon)
        
    results = []
    # FIX: Use pv.index.min() (first trading day) instead of inception_date for SI
    # This ensures ticker P/L boundaries match portfolio P/L boundaries exactly
    pv_start_date = pv.index.min()

    for _, row in sec_table.iterrows():
        t = row["ticker"]
        
        # Call calculate_ticker_pl with return_components=True
        # Pass portfolio_inception to align SI calculation
        res = calculate_ticker_pl(
            t, horizon, prices, as_of, tx_raw, sec_table, raw_start, 
            dividends=dividends,
            portfolio_inception=pv_start_date if horizon == "SI" else None,
            return_components=True
        )
        
        if isinstance(res, dict):
            item = {"ticker": t, "pl": res["pl"]}
            # Populate Meta Columns directly from calculation components
            item[f"meta_{horizon}_start"] = res["start"]
            item[f"meta_{horizon}_end"] = res["end"]
            item[f"meta_{horizon}_flow"] = res["flow"]
            item[f"meta_{horizon}_inc"] = res["inc"]
            item[f"meta_{horizon}_denom"] = res["denom"]
        else:
            item = {"ticker": t, "pl": res}
            
        results.append(item)
        
    return pd.DataFrame(results).set_index("ticker")

def get_asset_class_pl(data, asset_class, horizon, return_components=False):
    """
    Computes DIRECT asset class P/L using centralized engine logic.
    """
    return calculate_asset_class_pl(
        asset_class,
        horizon,
        data["prices"],
        data["pv"],
        data["inception_date"],
        data["tx_raw"],
        data["sec_table"],
        data["dividends"],
        return_components=return_components
    )

def get_projections_data(data):
    """
    Calculates projection scenarios.
    """
    pv = data["pv"]
    if pv.empty: return pd.DataFrame()
    
    initial_value = float(pv.iloc[-1])
    monthly_contrib = TARGET_MONTHLY_CONTRIBUTION
    rates = [0.05, 0.07, 0.09]
    years = list(range(21))
    
    results = []
    for yr in years:
        row = {"Year": yr}
        for r in rates:
            # Lump sum only
            lump_val = fv_lump(initial_value, r, yr)
            row[f"Lump {int(r*100)}%"] = lump_val
            
            # With Contributions
            contrib_val = lump_val + fv_contrib(monthly_contrib, r, yr)
            row[f"Contrib {int(r*100)}%"] = contrib_val
        results.append(row)
        
    return pd.DataFrame(results)

def get_rolling_correlations(data, window=90):
    """
    Calculates 90-day rolling correlation matrix for top 10 holdings.
    """
    sec_current = data["sec_table_current"]
    prices = data["prices"]
    
    if sec_current.empty or prices.empty:
        return pd.DataFrame()
        
    # Top 10 by weight (excluding CASH)
    top_tickers = sec_current[sec_current["ticker"] != "CASH"] \
        .nlargest(10, "weight")["ticker"].tolist()
        
    if not top_tickers:
        return pd.DataFrame()
        
    # Extract prices
    subset = prices[top_tickers].dropna()
    if subset.empty:
        return pd.DataFrame()
        
    # Returns
    rets = subset.pct_change().dropna()
    
    # Rolling Correlation
    # We want the *latest* snapshot of the 90-day correlation
    # If history < 90 days, use full history
    if len(rets) < window:
        corr_matrix = rets.corr()
    else:
        # Get the correlation matrix at the last timestamp
        # rolling().corr() returns a MultiIndex series (Date, Ticker) -> Ticker
        # Easier: Just take last 'window' days and corr()
        last_window = rets.iloc[-window:]
        corr_matrix = last_window.corr()
        
    return corr_matrix

def get_correlation_heatmap(data, theme="light"):
    """Generates Heatmap for Rolling Correlations."""
    corr = get_rolling_correlations(data)
    if corr.empty: return go.Figure()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r", # Red = High Corr, Blue = Inverse
        zmin=-1, zmax=1
    )
    
    fig.update_traces(hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>")

    fig.update_layout(
        title="90-Day Rolling Correlation (Top Holdings)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40),
        height=500
    )
    return fig

def get_daily_attribution_breakdown(data, date_str):
    """
    Decomposes a specific day's Market Effect into Asset Class components.
    
    Market Effect = (PV_end - PV_start) - Net External Flows
    Asset Effect = (AC_PV_end - AC_PV_start) - AC_Net_Internal_Flows + Dividends
    """
    pv = data["pv"]
    tx_raw = data["tx_raw"]
    holdings = data["holdings"]
    prices = data["prices"]
    dividends = data["dividends"]
    
    if pv.empty: return pd.DataFrame()
    
    target_date = pd.Timestamp(date_str)
    prev_date = target_date - pd.Timedelta(days=1)
    
    pv_daily = pv.sort_index()
    if target_date not in pv_daily.index:
        return pd.DataFrame()
        
    # Find previous available PV date
    idx_loc = pv_daily.index.searchsorted(target_date)
    if idx_loc == 0:
        return pd.DataFrame()
    prev_date = pv_daily.index[idx_loc - 1]
    
    ac_map = holdings.set_index("ticker")["asset_class"].to_dict()
    
    # Identify all active tickers
    available_tickers = [c for c in prices.columns if c in ac_map]
    
    # Pre-fetch prices
    window_prices = prices.loc[:target_date].ffill().iloc[-5:]
    try:
        p_curr = window_prices.loc[target_date]
        p_prev = window_prices.loc[prev_date]
    except KeyError:
        return pd.DataFrame()
        
    # Shares
    tx_sub = tx_raw[tx_raw["date"] <= target_date]
    shares_curr = tx_sub.groupby("ticker")["shares"].sum()
    
    tx_prev = tx_sub[tx_sub["date"] <= prev_date]
    shares_prev = tx_prev.groupby("ticker")["shares"].sum()
    
    # Flows on date
    tx_on_date = tx_raw[tx_raw["date"] == target_date]
    flows_by_ticker = defaultdict(float)
    if not tx_on_date.empty:
        grp = tx_on_date.groupby("ticker")["amount"].sum()
        for t, amt in grp.items():
            flows_by_ticker[t] = -amt

    # External Flows (Deposits/Withdrawals) from cf_ext
    # These are usually CASH flows that are NOT in tx_raw (if configured that way)
    # We must add them to flows_by_ticker["CASH"] to ensure Cash Effect is not distorted.
    cf_ext = data.get("cf_ext")
    ext_flow_today = 0.0
    if cf_ext is not None and not cf_ext.empty:
        ext_flow_today = cf_ext.loc[cf_ext["date"] == target_date, "amount"].sum()
        # Add to CASH flow (Deposit > 0 implies Flow In > 0)
        flows_by_ticker["CASH"] += ext_flow_today

    # Dividends on date (or between prev and curr?)
    # Daily attribution usually implies "on this day".
    divs_by_ticker = defaultdict(float)
    total_dividends = 0.0
    if not dividends.empty:
        # Strictly speaking, if we step from prev_date to target_date, we catch divs in (prev, target].
        # If daily, that's just target_date.
        mask_div = (dividends["date"] > prev_date) & (dividends["date"] <= target_date)
        div_grp = dividends.loc[mask_div].groupby("ticker")["amount"].sum()
        for t, amt in div_grp.items():
            divs_by_ticker[t] = amt
            total_dividends += amt
            
    # FIX: Treat Dividends as Flow INTO Cash
    # If dividend is paid, Cash Balance increases by 'total_dividends'.
    # This increase is NOT performance of Cash (unless interest).
    # So we must treat it as a Flow (Investment) into Cash asset class.
    # Flow = +Amount (Cash In).
    # flows_by_ticker["CASH"] stores -Amount. (If Buy, amt<0, Flow>0).
    # Here dividend is Cash In. So Flow should be positive.
    # We add total_dividends to the CASH flow.
    flows_by_ticker["CASH"] += total_dividends

    # Iterate all tickers involved
    all_tickers = set(shares_curr.index) | set(shares_prev.index) | set(flows_by_ticker.keys()) | set(divs_by_ticker.keys())
    
    # Calculate Denominator for Contribution %
    # GIPS Standard for Daily TWR: Denom = PV_start + External_Flows (Start-of-Day)
    # pv_daily has prev_date
    pv_start = float(pv_daily.loc[prev_date])
    denominator = pv_start + ext_flow_today
    if abs(denominator) < 1e-6: denominator = 1.0 # Avoid div/0

    ac_effects = defaultdict(float)
    ac_details = defaultdict(lambda: {"start": 0.0, "end": 0.0, "flow": 0.0, "inc": 0.0})
    
    for t in all_tickers:
        if t == "CASH": continue
        
        ac = ac_map.get(t, "Other")
        
        s_c = shares_curr.get(t, 0)
        s_p = shares_prev.get(t, 0)
        
        px_c = p_curr.get(t, 0)
        px_p = p_prev.get(t, 0)
        
        val_c = s_c * px_c
        val_p = s_p * px_p
        
        flow = flows_by_ticker.get(t, 0)
        div = divs_by_ticker.get(t, 0)
        
        # Effect = Change in Value - Net Investment + Income
        eff = (val_c - val_p) - flow + div
        ac_effects[ac] += eff
        
        # Aggregate Details for Meta
        ac_details[ac]["start"] += val_p
        ac_details[ac]["end"] += val_c
        ac_details[ac]["flow"] += flow
        ac_details[ac]["inc"] += div
        
    # Convert to DF
    df = pd.DataFrame(list(ac_effects.items()), columns=["Asset Class", "Effect"])
    
    # Calculate Contribution %
    df["Contribution (%)"] = (df["Effect"] / denominator) * 100.0
    
    # Add Audit Meta Columns
    df["meta_denominator"] = denominator
    df["meta_Return_denom"] = denominator # Alias for consistency
    df["meta_Return_start"] = pv_start
    df["meta_Return_flow"] = ext_flow_today
    
    # Add Asset Class specific meta
    df["meta_ac_start"] = df["Asset Class"].map(lambda x: ac_details[x]["start"])
    df["meta_ac_end"] = df["Asset Class"].map(lambda x: ac_details[x]["end"])
    df["meta_ac_flow"] = df["Asset Class"].map(lambda x: ac_details[x]["flow"])
    df["meta_ac_inc"] = df["Asset Class"].map(lambda x: ac_details[x]["inc"])
    
    df = df.sort_values("Effect", ascending=False)
    
    return df

# ============================================================
# CHART GENERATORS (PLOTLY)
# ============================================================

def _hex_to_rgba(hex_code, alpha=0.2):
    """Helper to convert hex to rgba string."""
    hex_code = hex_code.lstrip('#')
    return f"rgba({int(hex_code[0:2], 16)}, {int(hex_code[2:4], 16)}, {int(hex_code[4:6], 16)}, {alpha})"

def get_pv_mountain_chart(data, theme="light"):
    """Generates interactive PV Mountain chart using GIPS-compliant TWR."""
    pv = data["pv"]
    if pv.empty: return go.Figure()
    
    # Build daily PV for hover display
    pv_daily = pv.sort_index().reindex(
        pd.date_range(pv.index.min(), pv.index.max(), freq="D")
    ).ffill()
    
    # Fetch TWR Curve from centralized helper
    twr_curve = _get_daily_twr_curve(data)
    twr_curve_daily = twr_curve.reindex(pv_daily.index, method='ffill')
    
    # Convert to percentage return (NO REBASING - curve already contains Day 1 return)
    twr_ret_pct = (twr_curve_daily - 1.0) * 100.0
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=twr_ret_pct.index,
        y=twr_ret_pct.values,
        mode='lines',
        fill='tozeroy',
        name='Portfolio Return (TWR)',
        line=dict(color=GLOBAL_PALETTE[0], width=2),
        fillcolor=_hex_to_rgba(GLOBAL_PALETTE[0], 0.2),
        customdata=pv_daily.values,
        hovertemplate="<b>TWR</b>: %{y:.2f}%<br><b>Value</b>: %{customdata:$,.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        
        yaxis_title="Return (%)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified"
    )
    return fig

def get_cumulative_return_chart(data, start_date=None, benchmark_tickers=None, theme="light"):
    """Generates Cumulative Return chart (TWR) vs Benchmarks."""
    pv = data["pv"]
    
    # 1. FILTER PV: Ignore pre-inception zeros (This forces the 11/4 start)
    pv = pv[pv > 0].sort_index()
    if pv.empty: return go.Figure()

    # 2. DETERMINE START: Default to first non-zero PV date
    market_start = pv.index[0]
    
    if not start_date:
        start_date = market_start
    else:
        start_date = pd.to_datetime(start_date)
        # Prevent start_date from going back before data exists
        if start_date < market_start: start_date = market_start
    
    # 1. Fetch TWR Curve from centralized helper
    twr_curve = _get_daily_twr_curve(data)
    
    # Filter Window
    twr_window = twr_curve[twr_curve.index >= start_date]
    if twr_window.empty: return go.Figure()
    
    # Convert to percentage return (NO REBASING - curve already contains Day 1 return)
    twr_plot = (twr_window - 1.0) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=twr_plot.index,
        y=twr_plot.values,
        mode='lines',
        name='Portfolio',
        line=dict(color=GLOBAL_PALETTE[0], width=3),
        hovertemplate="<b>Portfolio</b>: %{y:.2f}%<extra></extra>"
    ))
    
    # 2. Benchmarks (unchanged logic, just context)
    colors = [GLOBAL_PALETTE[2], GLOBAL_PALETTE[4], GLOBAL_PALETTE[6], GLOBAL_PALETTE[10]]
    if benchmark_tickers:
        for i, (name, ticker) in enumerate(benchmark_tickers.items()):
            try:
                hist = fetch_price_history([ticker], use_adj_close=True)
                ser = hist[ticker]
                
                # GIPS COMPLIANCE FIX: Benchmark Normalization
                # If starting at inception, we need the benchmark's return for Day 1.
                # Standard normalization (P / P[0] - 1) sets Day 1 return to 0%.
                # We attempt to fetch the previous day's close to use as the base.
                
                base_price = None
                
                if start_date == market_start:
                    # Look for price strictly before start_date
                    history_before = ser[ser.index < start_date]
                    if not history_before.empty:
                        base_price = float(history_before.iloc[-1])
                
                # Filter strictly >= start_date for plotting X-axis
                ser_plot = ser[ser.index >= start_date]
                ser_plot = ser_plot[ser_plot.index <= pv.index.max()]
                
                if not ser_plot.empty:
                    # If we found a prior close, use it as base. Otherwise default to Day 1 Open/Close (0% start)
                    if base_price is None:
                        base_price = float(ser_plot.iloc[0])
                        
                    ser_norm = (ser_plot / base_price - 1.0) * 100.0
                    fig.add_trace(go.Scatter(
                        x=ser_norm.index,
                        y=ser_norm.values,
                        mode='lines',
                        name=name,
                        line=dict(color=colors[i % len(colors)], width=1.5),
                        hovertemplate=f"<b>{name}</b>: %{{y:.2f}}%<extra></extra>"
                    ))
            except:
                pass
                
    fig.update_layout(
        
        yaxis_title="Return (%)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def get_asset_allocation_charts(data, theme="light"):
    """Generates Pie and Bar charts for Asset Allocation."""
    sec_table = data["sec_table_current"]
    holdings = data["holdings"]
    
    if sec_table.empty: return go.Figure(), go.Figure()
    
    # Prepare Data
    sec_grouped = sec_table.groupby("asset_class").agg(value=("market_value", "sum")).reset_index()
    
    asset_class_map = {
        "US Large Cap": "US LC", "US Growth": "US Growth", "US Small Cap": "US SC",
        "International Equity": "INTL EQTY", "Gold / Precious Metals": "GOLD",
        "Digital Assets": "DIGITAL", "US Bonds": "US Bonds", "CASH": "CASH", "Fixed Income": "FI"
    }
    sec_grouped["short_name"] = sec_grouped["asset_class"].map(lambda x: asset_class_map.get(x, x))
    
    targets = holdings.groupby("asset_class")["target_pct"].sum().reset_index()
    targets["short_name"] = targets["asset_class"].map(lambda x: asset_class_map.get(x, x))
    
    merged = pd.merge(sec_grouped, targets[["short_name", "target_pct"]], on="short_name", how="outer").fillna(0)
    merged = merged.sort_values("value", ascending=False)
    
    total_val = merged["value"].sum()
    merged["actual_pct"] = merged["value"] / total_val * 100
    merged["delta"] = merged["actual_pct"] - merged["target_pct"]
    
    # Generate Custom Text Labels (Hide < 5%)
    display_text = []
    for _, row in merged.iterrows():
        if row["actual_pct"] < 5.0:
            display_text.append("")
        else:
            display_text.append(f"{row['short_name']}<br>{row['actual_pct']:.1f}%")
            
    # Pie Chart
    pie_fig = go.Figure(go.Pie(
        labels=merged["short_name"],
        values=merged["value"],
        text=display_text,
        hole=0.4,
        textinfo='text',
        marker=dict(colors=GLOBAL_PALETTE),
        sort=False,
        direction='clockwise',
        rotation=-90,
        textfont=dict(color='black' if theme == 'light' else 'white'),
        hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Share: %{percent:.2%}<extra></extra>"
    ))
    pie_fig.update_layout(
        
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05,
            title_text="&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Legend",
            bordercolor="Grey",
            borderwidth=1
        )
    )
    
    # Bar Chart (Actual vs Target)
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=merged["short_name"],
        y=merged["actual_pct"],
        name="Actual %",
        marker_color=GLOBAL_PALETTE[0],
        customdata=merged["value"],
        hovertemplate="<b>Actual</b>: %{y:.2f}%<br>Value: %{customdata:$,.2f}<extra></extra>"
    ))
    bar_fig.add_trace(go.Bar(
        x=merged["short_name"],
        y=merged["target_pct"],
        name="Target %",
        marker_color=GLOBAL_PALETTE[1],
        hovertemplate="<b>Target</b>: %{y:.2f}%<extra></extra>"
    ))
    
    bar_fig.update_layout(
        
        barmode='group',
        yaxis_title="Percentage (%)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40)
    )
    
    return pie_fig, bar_fig

def get_asset_drilldown_chart(data, asset_class, theme="light"):
    """Generates Ticker Pie Chart for a specific Asset Class."""
    sec_table = data["sec_table_current"]
    
    if sec_table.empty: return go.Figure()

    # Reverse Map for Drilldown (Short Name -> Full Name)
    # Must match get_asset_allocation_charts mapping
    asset_class_map = {
        "US Large Cap": "US LC", "US Growth": "US Growth", "US Small Cap": "US SC",
        "International Equity": "INTL EQTY", "Gold / Precious Metals": "GOLD",
        "Digital Assets": "DIGITAL", "US Bonds": "US Bonds", "CASH": "CASH", "Fixed Income": "FI"
    }
    reverse_map = {v: k for k, v in asset_class_map.items()}
    full_name = reverse_map.get(asset_class, asset_class)
    
    # Filter by Asset Class
    filtered = sec_table[sec_table["asset_class"] == full_name].copy()
    
    # Filter for > 0 value
    filtered = filtered[filtered["market_value"] > 0].copy()
    filtered = filtered.sort_values("market_value", ascending=False)
    
    if filtered.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white" if theme == "light" else "plotly_dark",
            title=f"No holdings in {full_name}"
        )
        return fig
    
    # Calculate percentages
    total_val = filtered["market_value"].sum()
    filtered["actual_pct"] = filtered["market_value"] / total_val * 100
    
    display_text = []
    for _, row in filtered.iterrows():
        if row["actual_pct"] < 5.0:
            display_text.append("")
        else:
            display_text.append(f"{row['ticker']}<br>{row['actual_pct']:.1f}%")
            
    # Pie Chart
    fig = go.Figure(go.Pie(
        labels=filtered["ticker"],
        values=filtered["market_value"],
        text=display_text,
        hole=0.6,
        textinfo='text',
        textposition='outside',
        marker=dict(colors=GLOBAL_PALETTE),
        sort=False,
        direction='clockwise',
        rotation=-90,
        textfont=dict(color='black' if theme == 'light' else 'white'),
        hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Share: %{percent:.2%}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=f"{full_name}", x=0.5,y=0.5, xanchor='center', yanchor='middle', font=dict(size=14, color='black' if theme == 'light' else 'white')),
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05,
            title_text="Holdings",
            bordercolor="Grey",
            borderwidth=1
        )
    )
    
    return fig

def get_sector_allocation_chart(data, theme="light"):
    """Generates Horizontal Bar for Sector Allocation."""
    sector_df = data["sector_df"]
    if sector_df.empty: return go.Figure()
    
    fig = go.Figure(go.Bar(
        y=sector_df["Sector"],
        x=sector_df["Exposure"],
        orientation='h',
        marker_color=GLOBAL_PALETTE[0],
        text=sector_df["Exposure"].apply(lambda x: f"{x:.2f}%"),
        textposition='auto',
        hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(
        
        xaxis_title="Exposure (%)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def get_allocation_history_chart(data, theme="light"):
    """Generates Stacked Area for Allocation History."""
    # Similar logic to dash_wrappers (old) but returning Plotly
    # Reimplementing simplified version to avoid huge code block
    # Need pos_daily logic.
    
    tx_hist = data["tx_raw"]
    if tx_hist.empty: return go.Figure()
    
    # Simplified: Just grab holdings history from transactions is hard without full engine rebuild
    # For now, return a placeholder or implement the full logic if critical.
    # The user mandated "Replicate ALL report components".
    # So I must implement the full logic.
    
    pv = data["pv"]
    prices = fetch_price_history(list(set(tx_hist["ticker"].unique()) - {"CASH"}))
    sec_table = data["sec_table"]
    holdings = data["holdings"]
    
    full_index = pd.date_range(start=pv.index.min(), end=pv.index.max(), freq="D")
    
    # Process shares
    tx_hist["date"] = pd.to_datetime(tx_hist["date"])
    pos_changes = tx_hist.pivot_table(index="date", columns="ticker", values="shares", aggfunc="sum").sort_index()
    pos_changes = pos_changes.reindex(full_index, fill_value=0.0)
    pos_daily = pos_changes.cumsum().ffill().bfill()
    
    # Reconcile to current
    current_shares = sec_table.set_index("ticker")["shares"]
    for t, shares in current_shares.items():
        if t == "CASH": continue
        if t in pos_daily.columns:
            diff = shares - pos_daily[t].iloc[-1]
            if abs(diff) > 1e-6: pos_daily[t] += diff
        elif t in prices.columns:
            pos_daily[t] = shares
            
    common = [t for t in pos_daily.columns if t in prices.columns]
    pos_daily = pos_daily[common]
    px_aligned = prices[common].reindex(full_index).ffill().bfill()
    
    mv_daily = pos_daily * px_aligned
    
    # Map to Asset Class
    ac_map = holdings.set_index("ticker")["asset_class"].to_dict()
    mv_daily.columns = [ac_map.get(t, "Unknown") for t in mv_daily.columns]
    mv_by_class = mv_daily.T.groupby(level=0).sum().T
    
    # Cash residual
    invested = mv_by_class.sum(axis=1)
    pv_aligned = pv.reindex(full_index).ffill().bfill()
    mv_by_class["Cash"] = pv_aligned - invested
    
    # Percentages
    total = mv_by_class.sum(axis=1).replace(0, np.nan)
    pct = mv_by_class.div(total, axis=0) * 100
    
    # Smoothing (Removed to ensure endpoint matches Donut chart)
    # pct_smooth = pct.rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    fig = go.Figure()
    for i, col in enumerate(pct.columns):
        fig.add_trace(go.Scatter(
            x=pct.index,
            y=pct[col],
            mode='lines',
            stackgroup='one',
            name=col,
            line=dict(color=GLOBAL_PALETTE[i % len(GLOBAL_PALETTE)]),
            hovertemplate=f"<b>{col}</b>: %{{y:.2f}}%<extra></extra>"
        ))
        
    fig.update_layout(
        
        yaxis_title="Allocation (%)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            xanchor="center",
            x=0.5,
            title_text="Legend",
            bordercolor="Grey",
            borderwidth=1
        )
    )
    return fig

def get_monthly_attribution_breakdown(data, year_month_str):
    """
    Decomposes a specific month's Market Effect into Asset Class components
    using Frongello Linking.
    """
    year, month = map(int, year_month_str.split('-'))
    start_date = pd.Timestamp(year, month, 1)
    end_date = start_date + pd.offsets.MonthEnd(1)

    # Use the robust Frongello engine
    df = _calculate_frongello_linking(data, start_date=start_date, end_date=end_date)
    
    if df.empty: return pd.DataFrame()
    
    # For Monthly/Weekly, we need to manually adjust for Residual/Recon 
    # since _calculate_frongello_linking doesn't include it by default (it's for SI).
    # However, for a single month, it's safer to just sort and return.
    return df.sort_values("Contribution (%)", ascending=False)


def get_weekly_attribution_breakdown(data, date_str):
    """
    Decomposes a specific week's Market Effect into Asset Class components
    using Frongello Linking. 
    
    Aligns strictly with the 'W-FRI' resampling used in the main chart.
    """
    end_date = pd.Timestamp(date_str)
    
    # Calculate Start Date: The day after the previous Friday.
    # This ensures alignment with resample('W-FRI') bins.
    
    # 1. Find the Friday strictly before end_date
    # weekday: Mon=0 ... Fri=4 ... Sun=6
    days_since_fri = (end_date.weekday() - 4) % 7
    
    # If end_date is Fri (4), offset is 0. But we want the PREVIOUS Friday (start of bin), 
    # so we subtract 7. If offset > 0, it means we are mid-week, so we subtract offset 
    # to get to the immediately preceding Friday.
    offset = days_since_fri if days_since_fri > 0 else 7
    
    last_friday = end_date - pd.Timedelta(days=offset)
    start_date = last_friday + pd.Timedelta(days=1)

    # Use the robust Frongello engine
    df = _calculate_frongello_linking(data, start_date=start_date, end_date=end_date)
    
    if df.empty: return pd.DataFrame()
    
    return df.sort_values("Contribution (%)", ascending=False)


def get_smart_attribution_chart(data, theme="light"):
    """
    Generates Daily or Monthly Delta PV Attribution chart based on portfolio history.
    """
    pv = data["pv"]
    cf_ext = data["cf_ext"]
    if pv.empty:
        return go.Figure()

    pv_daily = pv.sort_index().reindex(pd.date_range(pv.index.min(), pv.index.max(), freq="D")).ffill()
    
    # GIPS COMPLIANCE: Calculate Market Effect (Delta PV) including Day 1
    # Market Effect = (End Value - Start Value) - Net External Flows
    # For Day 1: Market Effect = First Close Value - Initial Funding Flow
    
    # 1. External Flows
    if not cf_ext.empty:
        ext = cf_ext.groupby("date")["amount"].sum().reindex(pv_daily.index, fill_value=0)
    else:
        ext = pd.Series(0, index=pv_daily.index)

    # 2. Daily Changes in Value
    # Using a shifted series to correctly calculate Day 1 gain: (PV_1 - 0) - Flow_1
    pv_shifted = pv_daily.shift(1).fillna(0)
    mkt = (pv_daily - pv_shifted) - ext
    
    # Decide on aggregation
    history_days = (pv_daily.index.max() - pv_daily.index.min()).days
    
    if history_days > 90:
        # Long history: Monthly
        freq = 'ME' 
        p_label = 'monthly'
        fmt = '%Y-%m'
    elif history_days > 30:
        # Medium history: Weekly (Fixes jaggedness)
        freq = 'W-FRI'
        p_label = 'weekly'
        fmt = '%Y-%m-%d'
    else:
        # Short history: Daily
        freq = None
        p_label = 'daily'
        fmt = '%Y-%m-%d'

    # Apply Resampling
    if freq:
        mkt_agg = mkt.resample(freq).sum()
        ext_agg = ext.resample(freq).sum()
        
        # FIX: Clamp last date to actual end date (prevent future labels)
        true_end = pv.index.max()
        if not mkt_agg.empty and mkt_agg.index[-1] > true_end:
            new_idx = mkt_agg.index.tolist()
            new_idx[-1] = true_end
            mkt_agg.index = pd.Index(new_idx)
            ext_agg.index = pd.Index(new_idx)
    else:
        mkt_agg = mkt
        ext_agg = ext

    # GIPS COMPLIANCE: Include all periods including inception for correct Cumulative ΔPV reconciliation
    # (Previously dropped the first period to hide initial deposit, but this distorted the total P/L)

    # Set vars for plotting
    period = 'M' if freq == 'ME' else 'D'
    x_labels = mkt_agg.index.strftime(fmt)
    custom_data = [{'period': p_label, 'date': label} for label in x_labels]


    fig = go.Figure()

    # External Flows Bar
    fig.add_trace(go.Bar(
        x=x_labels,
        y=ext_agg,
        name="External Flows",
        marker_color=GLOBAL_PALETTE[1],
        customdata=[d['date'] for d in custom_data]
    ))

    # Market Effect Bar
    fig.add_trace(go.Bar(
        x=x_labels,
        y=mkt_agg,
        name="Market Effect",
        marker_color=GLOBAL_PALETTE[0],
        customdata=[d['date'] for d in custom_data]
    ))
    
    # Add customdata to traces for drilldown
    fig.update_traces(selector=dict(type='bar'), customdata=[d['date'] for d in custom_data], hovertemplate='%{y:$,.2f}<extra>%{customdata}</extra>')


    # Cumulative Line: Sum of Market Effects only (Represents Since Inception Gain)
    cum = mkt_agg.cumsum()
    
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=cum.values,
        mode='lines+markers',
        name="Cumulative ΔPV",
        line=dict(color=GLOBAL_PALETTE[2], width=2, dash='dot'),
        yaxis='y2',
        hovertemplate="<b>Cumulative Gain</b>: %{y:$,.2f}<extra></extra>"
    ))

    # Calculate a y-axis buffer to prevent label clipping on iPad
    # VISUAL FIX: Focus the Y-axis on Market Effect (Performance) rather than giant inflows.
    # We clip the primary axis to the range of market effects to keep the chart readable.
    bar_max = mkt_agg.max()
    bar_min = mkt_agg.min()
    y_max = max(bar_max * 1.5, 500) # Minimum $500 window for visibility
    y_min = min(bar_min * 1.2, -500)

    fig.update_layout(
        yaxis_title="Change ($)",
        yaxis=dict(range=[y_min, y_max]), # Focus scale on Performance Gain/Loss
        yaxis2=dict(title="Cumulative Gain ($)", overlaying='y', side='right'),
        barmode='relative',
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=40, r=40, t=80, b=40), # Increased top margin (t) to 80
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Period"
    )
    # Ensure text labels don't clip at the axis edge
    fig.update_traces(selector=dict(type='bar'), cliponaxis=False)
    
    # Store period type in figure's metadata for the callback
    fig.update_layout(meta={'period_type': p_label})

    return fig



def get_risk_return_chart(data, theme="light"):
    """Generates Risk vs Return scatter from Dynamic Risk Profile."""
    risk_return = data.get("risk_return", {})
    if not risk_return: return go.Figure()
    
    plot_data = []
    for cls, metrics in risk_return.items():
        plot_data.append({
            "Asset Class": cls,
            "Return": metrics["return"],
            "Volatility": metrics["vol"]
        })
    df = pd.DataFrame(plot_data)
    
    if df.empty: return go.Figure()

    fig = px.scatter(
        df, x="Volatility", y="Return", hover_name="Asset Class",
        size=[1]*len(df), size_max=10, # Uniform size markers
        color="Asset Class",
        color_discrete_sequence=GLOBAL_PALETTE
    )
    
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>")
    
    fig.update_layout(
        
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        showlegend=True,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
            title_text="Legend",
            bordercolor="Grey",
            borderwidth=1
        )
    )
    return fig

def get_drawdown_chart(data, theme="light"):
    """
    Generates Underwater Chart (Drawdown) from TWR Curve.
    """
    twr_curve = _get_daily_twr_curve(data)
    if twr_curve.empty: return go.Figure()
    
    drawdown_series, max_dd, recovery_days = compute_drawdown_series(twr_curve)
    
    fig = go.Figure()
    
    # Area Chart for Drawdown
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values,
        mode='lines',
        fill='tozeroy',
        name='Drawdown',
        line=dict(color=GLOBAL_PALETTE[2], width=1), # Red
        fillcolor=_hex_to_rgba(GLOBAL_PALETTE[2], 0.3),
        hovertemplate="<b>Drawdown</b>: %{y:.2f}%<extra></extra>"
    ))
    
    # Annotate Max Drawdown
    if max_dd < 0:
        min_date = drawdown_series.idxmin()
        fig.add_annotation(
            x=min_date, y=max_dd,
            text=f"Max Drawdown: {max_dd:.2f}%",
            showarrow=True,
            arrowhead=1,
            yshift=-10
        )
        
    fig.update_layout(
        yaxis_title="Drawdown (%)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        yaxis=dict(autorange="reversed") # Invert axis so 0 is at top
    )
    return fig

def get_projections_chart(data, theme="light", rate_pct=None, monthly_contrib=None):
    """
    Generates 20-Year Projection Chart.
    If rate_pct/monthly_contrib are provided, uses those.
    Otherwise defaults to 5/7/9% splits.
    """
    pv = data["pv"]
    if pv.empty: return go.Figure()
    
    initial_value = float(pv.iloc[-1])
    years = list(range(21))
    
    # Palette matching report_charts.py order (using GLOBAL_PALETTE indices)
    colors = [
        GLOBAL_PALETTE[0], GLOBAL_PALETTE[1], GLOBAL_PALETTE[2],  # Lump Sums: Low, Mid, High
        GLOBAL_PALETTE[3], GLOBAL_PALETTE[4], GLOBAL_PALETTE[5]   # Contribs: Low, Mid, High
    ]
    
    fig = go.Figure()
    
    if rate_pct is not None and monthly_contrib is not None:
        # Dynamic Mode
        rates = [rate_pct - 2, rate_pct, rate_pct + 2]
        
    else:
        # Static Mode (Default)
        rates = [5, 7, 9]
        monthly_contrib = TARGET_MONTHLY_CONTRIBUTION
        
        
    # 1. Plot Lump Sum Lines (Solid)
    for i, r in enumerate(rates):
        vals = []
        r_dec = r / 100.0
        for yr in years:
            vals.append(fv_lump(initial_value, r_dec, yr))
            
        fig.add_trace(go.Scatter(
            x=years, y=vals,
            mode='lines',
            name=f"{r:.1f}% Lump Sum",
            line=dict(color=colors[i], width=2),
            hovertemplate=f"<b>{r:.1f}% Lump Sum</b>: %{{y:$,.2f}}<extra></extra>"
        ))
        
    # 2. Plot Contribution Lines (Dashed)
    for i, r in enumerate(rates):
        vals = []
        r_dec = r / 100.0
        for yr in years:
            lump = fv_lump(initial_value, r_dec, yr)
            contrib = fv_contrib(monthly_contrib, r_dec, yr)
            vals.append(lump + contrib)
            
        fig.add_trace(go.Scatter(
            x=years, y=vals,
            mode='lines',
            name=f"{r:.1f}% + ${monthly_contrib:,.0f}/mo",
            line=dict(color=colors[i+3], width=2, dash='dash'),
            hovertemplate=f"<b>{r:.1f}% + ${monthly_contrib:,.0f}/mo</b>: %{{y:$,.2f}}<extra></extra>"
        ))
        
    fig.update_layout(
        
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        hovermode="x unified"
    )
    return fig

def get_flows_chart(data, theme="light"):
    """Generates Internal Flows by Asset Class chart."""
    tx_raw = data["tx_raw"]
    holdings = data["holdings"]
    
    if tx_raw.empty: return go.Figure()
    
    ac_map = holdings.set_index("ticker")["asset_class"].to_dict()
    tx_raw["asset_class"] = tx_raw["ticker"].map(ac_map).fillna("Other")
    
    net_flows = tx_raw.groupby("asset_class")["amount"].sum().sort_values()
    
    fig = go.Figure(go.Bar(
        y=net_flows.index,
        x=net_flows.values,
        orientation='h',
        marker_color=np.where(net_flows > 0, GLOBAL_PALETTE[4], GLOBAL_PALETTE[2]),
        hovertemplate="<b>%{y}</b>: %{x:$,.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        
        xaxis_title="Net Flow ($)",
        template="plotly_white" if theme == "light" else "plotly_dark"
    )
    return fig

def get_excess_return_chart(data, benchmark_tickers, theme="light"):
    """Generates Excess Return Bar Chart."""
    twr_df = data["twr_df"]
    pv = data["pv"]
    if twr_df.empty: return go.Figure()
    
    horizons = ["1D", "1W", "MTD", "1M", "3M", "6M", "1Y", "SI"]
    port_rets = twr_df.set_index("Horizon")["Return"]
    
    # Add SI if missing
    if "SI" not in port_rets:
        port_rets["SI"] = data["twr_si_ann"] if pd.notna(data["twr_si_ann"]) else data["twr_si"]
        
    fig = go.Figure()
    
    for i, (bm_name, bm_ticker) in enumerate(benchmark_tickers.items()):
        excess_vals = []
        tooltip_data = [] # Stores [Port Ret, BM Ret, Excess]
        
        for h in horizons:
            p_val = port_rets.get(h, 0.0) # Default to 0.0 if missing to avoid NaN issues in calc
            if pd.isna(p_val): p_val = 0.0
            
            # Fetch BM ret
            start = data["inception_date"] if h == "SI" else get_portfolio_horizon_start(pv, data["inception_date"], h)
            
            b_ret = 0.0
            diff = 0.0
            
            if start is not None: 
                try:
                    hist = fetch_price_history([bm_ticker], use_adj_close=True)
                    ser = hist[bm_ticker]
                    
                    # Logic to find base price (Handle SI / Day 1 Return)
                    base_price = None
                    
                    # If start aligns with earliest portfolio date (SI case), try to get previous close
                    # market_start is pv.index.min()
                    market_start = pv.index.min()
                    
                    if start <= market_start:
                         # Look for price strictly before start date
                         history_before = ser[ser.index < start]
                         if not history_before.empty:
                             base_price = float(history_before.iloc[-1])
                    
                    # Filter for window
                    ser_window = ser[ser.index >= start]
                    # Clip to end date
                    ser_window = ser_window[ser_window.index <= pv.index.max()]
                    
                    if not ser_window.empty:
                        if base_price is None:
                            base_price = float(ser_window.iloc[0])
                            
                        end_price = float(ser_window.iloc[-1])
                        b_ret = end_price / base_price - 1.0
                    
                    diff = (p_val - b_ret) * 100
                except:
                    pass
            
            excess_vals.append(diff)
            tooltip_data.append([p_val * 100, b_ret * 100, diff])
                
        fig.add_trace(go.Bar(
            x=horizons,
            y=excess_vals,
            name=bm_name,
            marker_color=GLOBAL_PALETTE[i % len(GLOBAL_PALETTE)] if len(benchmark_tickers) > 1 else GLOBAL_PALETTE[0],
            customdata=tooltip_data,
            hovertemplate=(
                f"<b>{bm_name}</b><br>"
                "Portfolio: %{customdata[0]:.2f}%<br>"
                "Benchmark: %{customdata[1]:.2f}%<br>"
                "Excess: %{customdata[2]:.2f}%<extra></extra>"
            )
        ))
        
    fig.update_layout(
        
        yaxis_title="Excess Return (%)",
        barmode='group',
        template="plotly_white" if theme == "light" else "plotly_dark"
    )
    return fig

def get_ticker_allocation_charts(data, theme="light"):
    """Generates Pie and Bar charts for Ticker Allocation."""
    sec_table = data["sec_table_current"]
    holdings = data["holdings"]
    
    if sec_table.empty: return go.Figure(), go.Figure()
    
    # Filter for > 0 value
    ticker_group = sec_table[sec_table["market_value"] > 0].copy()
    ticker_group = ticker_group.sort_values("market_value", ascending=False)
    
    # Calculate percentages for custom labels
    total_val_pie = ticker_group["market_value"].sum()
    ticker_group["actual_pct"] = ticker_group["market_value"] / total_val_pie * 100
    
    display_text = []
    for _, row in ticker_group.iterrows():
        if row["actual_pct"] < 5.0:
            display_text.append("")
        else:
            display_text.append(f"{row['ticker']}<br>{row['actual_pct']:.1f}%")
    
    # Pie Chart
    pie_fig = go.Figure(go.Pie(
        labels=ticker_group["ticker"],
        values=ticker_group["market_value"],
        text=display_text,
        hole=0.4,
        textinfo='text',
        marker=dict(colors=GLOBAL_PALETTE),
        sort=False,
        direction='clockwise',
        rotation=-90,
        textfont=dict(color='black' if theme == 'light' else 'white'),
        hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Share: %{percent:.2%}<extra></extra>"
    ))
    pie_fig.update_layout(
        
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05,
            title_text="&nbsp;&nbsp;&nbsp;Legend",
            bordercolor="Grey",
            borderwidth=1
        )
    )
    
    # Bar Chart (Actual vs Target)
    # Merge target_pct from holdings
    ticker_merge = ticker_group[["ticker", "market_value"]].merge(
        holdings[["ticker", "target_pct"]],
        on="ticker",
        how="left"
    ).fillna(0)
    
    total_val = ticker_merge["market_value"].sum()
    ticker_merge["actual_pct"] = ticker_merge["market_value"] / total_val * 100
    
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=ticker_merge["ticker"],
        y=ticker_merge["actual_pct"],
        name="Actual %",
        marker_color=GLOBAL_PALETTE[0],
        customdata=ticker_merge["market_value"],
        hovertemplate="<b>Actual</b>: %{y:.2f}%<br>Value: %{customdata:$,.2f}<extra></extra>"
    ))
    bar_fig.add_trace(go.Bar(
        x=ticker_merge["ticker"],
        y=ticker_merge["target_pct"],
        name="Target %",
        marker_color=GLOBAL_PALETTE[1],
        hovertemplate="<b>Target</b>: %{y:.2f}%<extra></extra>"
    ))
    
    bar_fig.update_layout(
        
        barmode='group',
        yaxis_title="Percentage (%)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40)
    )
    
    return pie_fig, bar_fig

def get_flows_summary_ytd(data):
    """Returns YTD Flows Summary table data."""
    pv = data["pv"]
    cf_ext = data["cf_ext"]
    tx_raw = data["tx_raw"]
    dividends = data["dividends"]
    
    if pv.empty: return pd.DataFrame()
    
    as_of = pv.index.max()
    ytd_start = as_of.replace(month=1, day=1)
    
    # External
    flows_ext = cf_ext[cf_ext["date"] >= ytd_start]
    ytd_deposits = flows_ext.loc[flows_ext["amount"] > 0, "amount"].sum()
    ytd_withdrawals = flows_ext.loc[flows_ext["amount"] < 0, "amount"].sum()
    net_ytd_ext = flows_ext["amount"].sum()
    most_recent_ext = flows_ext["date"].max() if not flows_ext.empty else None
    
    # Internal
    tx_ytd = tx_raw[tx_raw["date"] >= ytd_start]
    ytd_buys = tx_ytd.loc[tx_ytd["amount"] < 0, "amount"].sum()
    ytd_sells = tx_ytd.loc[tx_ytd["amount"] > 0, "amount"].sum()
    most_recent_tx = tx_ytd["date"].max() if not tx_ytd.empty else None
    
    # Dividends
    div_ytd = dividends[dividends["date"] >= ytd_start]
    ytd_income = div_ytd["amount"].sum()
    most_recent_div = div_ytd["date"].max() if not div_ytd.empty else None
    
    net_ytd_internal = ytd_buys + ytd_sells + ytd_income
    
    # Most recent date
    dates = [d for d in [most_recent_ext, most_recent_tx, most_recent_div] if d is not pd.NaT and d is not None]
    most_recent_any = max(dates).strftime("%Y-%m-%d") if dates else "N/A"
    
    rows = [
        {"Metric": "YTD Net External Flows", "Value": fmt_dollar_clean(net_ytd_ext)},
        {"Metric": "• YTD Deposits", "Value": fmt_dollar_clean(ytd_deposits)},
        {"Metric": "• YTD Withdrawals", "Value": fmt_dollar_clean(ytd_withdrawals)},
        {"Metric": "YTD Net Internal Activity", "Value": fmt_dollar_clean(net_ytd_internal)},
        {"Metric": "• YTD Buys (Cash Out)", "Value": fmt_dollar_clean(ytd_buys)},
        {"Metric": "• YTD Sells (Cash In)", "Value": fmt_dollar_clean(ytd_sells)},
        {"Metric": "• YTD Income (Divs)", "Value": fmt_dollar_clean(ytd_income)},
        {"Metric": "Most Recent Flow", "Value": most_recent_any},
    ]
    return pd.DataFrame(rows)

def get_risk_diversification(data):
    """Returns Risk & Diversification table data."""
    sec_table = data["sec_table_current"]
    holdings = data["holdings"]
    
    if sec_table.empty: return pd.DataFrame()
    
    sec_no_cash = sec_table[sec_table["ticker"] != "CASH"]
    
    # Top 3
    top3_pct = sec_no_cash.nlargest(3, "weight")["weight"].sum() * 100 if not sec_no_cash.empty else 0
    
    # Largest Class
    ac_weights = sec_no_cash.groupby("asset_class")["weight"].sum() * 100
    largest_class = ac_weights.idxmax() if not ac_weights.empty else "N/A"
    largest_class_pct = ac_weights.max() if not ac_weights.empty else 0
    
    # Over/Underweight
    target_pct_map = holdings.groupby("asset_class")["target_pct"].sum().to_dict()
    
    largest_over = None
    largest_under = None
    max_diff = -np.inf
    min_diff = np.inf
    
    for ac, wt in ac_weights.items():
        target = target_pct_map.get(ac, 0)
        diff = wt - target
        if diff > max_diff:
            max_diff = diff
            largest_over = f"{ac} ({wt:.2f}% vs {target:.2f}%)"
        if diff < min_diff:
            min_diff = diff
            largest_under = f"{ac} ({wt:.2f}% vs {target:.2f}%)"
            
    rows = [
        {"Metric": "Top 3 holdings % of portfolio", "Value": f"{top3_pct:.2f}%"},
        {"Metric": "Largest asset class", "Value": f"{largest_class} ({largest_class_pct:.2f}%)"},
        {"Metric": "Largest overweight", "Value": largest_over if largest_over else "N/A"},
        {"Metric": "Largest underweight", "Value": largest_under if largest_under else "N/A"},
    ]
    return pd.DataFrame(rows)

def get_performance_highlights(data):
    """Returns Performance Highlights table data."""
    sec_table = data["sec_table_current"]
    if sec_table.empty: return pd.DataFrame()
    
    # Helper to get PL string
    def get_pl(t, h):
        df = get_ticker_pl_df(data, h)
        if df.empty or t not in df.index: return "N/A"
        return fmt_dollar_clean(df.loc[t, "pl"])
        
    rows = []
    
    # 1M
    if "1M" in sec_table.columns:
        valid = sec_table.dropna(subset=["1M"])
        if not valid.empty:
            top = valid.loc[valid["1M"].idxmax()]
            bot = valid.loc[valid["1M"].idxmin()]
            
            rows.append({
                "Metric": "Top 1M Performer",
                "Value": f"{top['ticker']} ({top['1M']*100:.2f}%, {get_pl(top['ticker'], '1M')})"
            })
            rows.append({
                "Metric": "Bottom 1M Performer",
                "Value": f"{bot['ticker']} ({bot['1M']*100:.2f}%, {get_pl(bot['ticker'], '1M')})"
            })
        else:
             rows.append({"Metric": "Top 1M Performer", "Value": "N/A"})
             rows.append({"Metric": "Bottom 1M Performer", "Value": "N/A"})
             
    # 1D
    if "1D" in sec_table.columns:
        valid = sec_table.dropna(subset=["1D"])
        if not valid.empty:
            top = valid.loc[valid["1D"].idxmax()]
            bot = valid.loc[valid["1D"].idxmin()]
            
            rows.append({
                "Metric": "Best 1D Performer",
                "Value": f"{top['ticker']} ({top['1D']*100:.2f}%, {get_pl(top['ticker'], '1D')})"
            })
            rows.append({
                "Metric": "Bottom 1D Performer",
                "Value": f"{bot['ticker']} ({bot['1D']*100:.2f}%, {get_pl(bot['ticker'], '1D')})"
            })
        else:
             rows.append({"Metric": "Best 1D Performer", "Value": "N/A"})
             rows.append({"Metric": "Bottom 1D Performer", "Value": "N/A"})
             
    return pd.DataFrame(rows)

# ============================================================
# GROWTH OF INVESTED CAPITAL
# ============================================================

def calculate_growth_of_capital_data(data, end_date=None):
    """
    GROWTH OF INVESTED CAPITAL CALCULATION
    
    1. Total Cash Invested = Cumulative External Flows (Deposits - Withdrawals).
    2. Asset Class Cash Invested = Net Internal Flows (Buys - Sells).
    3. Cash (Asset) Invested = Total Invested - Sum(Asset Class Invested).
       This balancing item ensures the columns sum to the Total.
       
    Returns:
        - time_series: DataFrame with Date, Asset Class, Cash Invested, Portfolio Value, Growth, Growth %
        - summary: Latest snapshot by asset class
    """
    tx_raw = data["tx_raw"]
    holdings = data["holdings"]
    sec_table = data["sec_table"]
    prices = data["prices"]
    pv = data["pv"]
    cf_ext = data.get("cf_ext")
    
    if tx_raw.empty or pv.empty:
        return {"time_series": pd.DataFrame(), "summary": pd.DataFrame()}

    # Date range covers full history
    start_date = data["inception_date"]
    
    # Handle end date
    if end_date:
        end_date = pd.Timestamp(end_date)
        max_avail = pv.index.max()
        if end_date > max_avail:
            end_date = max_avail
    else:
        end_date = pv.index.max()
        
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # ------------------------------------------------
    # 1. GLOBAL TOTALS (The Truth)
    # ------------------------------------------------
    
    # Total Portfolio Value (from Engine)
    pv_total = pv.reindex(date_range).ffill().fillna(0.0)
    
    # Total Cash Invested (External Flows)
    if cf_ext is not None and not cf_ext.empty:
        daily_ext = cf_ext.groupby("date")["amount"].sum().cumsum()
        invested_total = daily_ext.reindex(date_range, method='ffill').fillna(0.0)
    else:
        invested_total = pd.Series(0.0, index=date_range)
        
    # ------------------------------------------------
    # 2. ASSET CLASS COMPONENTS (Securities)
    # ------------------------------------------------
    
    ticker_to_ac = holdings.set_index("ticker")["asset_class"].to_dict()
    tx = tx_raw.copy()
    tx["asset_class"] = tx["ticker"].map(ticker_to_ac)
    # Filter for securities only (Exclude CASH transactions if any, and unmapped)
    tx = tx[tx["asset_class"].notna() & (tx["asset_class"] != "CASH")].copy()
    
    asset_classes = sorted(tx["asset_class"].unique())
    
    ac_invested_series = {}
    ac_pv_series = {}
    
    sum_ac_invested = pd.Series(0.0, index=date_range)
    sum_ac_pv = pd.Series(0.0, index=date_range)
    
    for ac in asset_classes:
        # A. Net Invested (Buys - Sells)
        ac_tx = tx[tx["asset_class"] == ac].copy()
        if ac_tx.empty:
            daily_net = pd.Series(0.0, index=date_range)
        else:
            # -Amount (Buy is neg, so -Buy is pos investment)
            daily_net = -ac_tx.groupby("date")["amount"].sum().cumsum()
        
        cum_inv = daily_net.reindex(date_range, method='ffill').fillna(0.0)
        ac_invested_series[ac] = cum_inv
        sum_ac_invested += cum_inv
        
        # B. Portfolio Value (Reconstructed)
        # Get all tickers in this asset class
        ac_tickers = holdings[holdings["asset_class"] == ac]["ticker"].unique()
        ac_value = pd.Series(0.0, index=date_range)
        
        for ticker in ac_tickers:
            if ticker == "CASH" or ticker not in prices.columns:
                continue
            
            ticker_prices = prices[ticker].reindex(date_range).ffill().bfill().fillna(0.0)
            ticker_tx = tx[tx["ticker"] == ticker].copy()
            if ticker_tx.empty:
                continue
                
            daily_shares = ticker_tx.groupby("date")["shares"].sum()
            shares = daily_shares.reindex(date_range, fill_value=0.0).cumsum()
            ac_value += (shares * ticker_prices)
            
        ac_pv_series[ac] = ac_value
        sum_ac_pv += ac_value

    # ------------------------------------------------
    # 3. CASH PV (Explicit Calculation)
    # ------------------------------------------------
    # Cash Invested = Total External - Sum(Security Net Invested)
    # This represents the Principal allocated to Cash.
    cash_invested = invested_total - sum_ac_invested
    
    # Cash PV = (Total External) - (Sum Net Invested in Secs) + (Cum Dividends)
    # Ideally should also include Realized P/L from trading, but since
    # sum_ac_invested accounts for the *cost* of buys and *proceeds* of sells,
    # the difference (invested_total - sum_ac_invested) CORRECTLY captures
    # the cash balance resulting from all capital flows (External + Trading).
    # We just need to add Income (Dividends).
    
    dividends = data.get("dividends")
    if dividends is not None and not dividends.empty:
        daily_divs = dividends.groupby("date")["amount"].sum().cumsum()
        cum_divs = daily_divs.reindex(date_range, method='ffill').fillna(0.0)
    else:
        cum_divs = pd.Series(0.0, index=date_range)
        
    cash_pv = cash_invested + cum_divs
    
    # Ensure non-negative (rounding protection)
    cash_pv = cash_pv.clip(lower=0.0)
    
    # Add CASH to our collections
    ac_invested_series["CASH"] = cash_invested
    ac_pv_series["CASH"] = cash_pv
    asset_classes.append("CASH") # Add to list for iteration
    
    # ------------------------------------------------
    # 4. RECALCULATE TOTAL PV (Consistency Check)
    # ------------------------------------------------
    # Ensure the chart's "Total" line matches the sum of the stacked components
    pv_total_recalc = sum_ac_pv + cash_pv
    
    # ------------------------------------------------
    # 5. BUILD DATAFRAME
    # ------------------------------------------------
    rows = []
    
    # A. Asset Classes (including CASH)
    for ac in asset_classes:
        inv = ac_invested_series[ac]
        val = ac_pv_series[ac]
        growth = val - inv
        
        # Build DataFrame directly from Series for speed
        df_ac = pd.DataFrame({
            "Date": date_range,
            "Asset Class": ac,
            "Cash Invested": inv,
            "Portfolio Value": val,
            "Growth": growth
        })
        # Handle division by zero for %
        df_ac["Growth %"] = np.where(
            df_ac["Cash Invested"] > 1.0, # Threshold to avoid noise
            ((df_ac["Portfolio Value"] / df_ac["Cash Invested"]) - 1) * 100,
            0.0
        )
        
        # Resample to Weekly but include Start Date AND End Date
        start_row = df_ac.iloc[[0]]
        end_row = df_ac.iloc[[-1]]
        
        weekly = df_ac.set_index("Date").resample("W").last().reset_index()
        
        # Concatenate Start, Weekly points, and End Date
        # Filter weekly points to ensure they don't exceed end_date (resample bin edge issue)
        weekly = weekly[weekly["Date"] <= end_date]
        
        final_df = pd.concat([start_row, weekly, end_row]).drop_duplicates("Date").sort_values("Date")
        final_df["Asset Class"] = ac # Restore col
        rows.append(final_df)
        
    # B. TOTAL Row (Using recalculated total for consistency)
    df_total = pd.DataFrame({
        "Date": date_range,
        "Asset Class": "Total",
        "Cash Invested": invested_total,
        "Portfolio Value": pv_total_recalc,
        "Growth": pv_total_recalc - invested_total
    })
    df_total["Growth %"] = np.where(
        df_total["Cash Invested"] > 1.0,
        ((df_total["Portfolio Value"] / df_total["Cash Invested"]) - 1) * 100,
        0.0
    )
    
    start_row_total = df_total.iloc[[0]]
    end_row_total = df_total.iloc[[-1]]
    
    weekly_total = df_total.set_index("Date").resample("W").last().reset_index()
    weekly_total = weekly_total[weekly_total["Date"] <= end_date]
    
    final_total = pd.concat([start_row_total, weekly_total, end_row_total]).drop_duplicates("Date").sort_values("Date")
    final_total["Asset Class"] = "Total"
    rows.append(final_total)
    
    # Concat all
    time_series_df = pd.concat(rows, ignore_index=True)
    
    # ------------------------------------------------
    # 5. SUMMARY SNAPSHOT
    # ------------------------------------------------
    if not time_series_df.empty:
        latest_date = time_series_df["Date"].max()
        summary_df = time_series_df[time_series_df["Date"] == latest_date].copy()
        
        # Sort: Total last, others by PV desc
        total_row = summary_df[summary_df["Asset Class"] == "Total"]
        other_rows = summary_df[summary_df["Asset Class"] != "Total"].sort_values("Portfolio Value", ascending=False)
        summary_df = pd.concat([other_rows, total_row], ignore_index=True)
    else:
        summary_df = pd.DataFrame()

    return {
        "time_series": time_series_df,
        "summary": summary_df
    }

def get_growth_of_capital_chart(data, filter_value="Total", theme="light", end_date=None):
    """
    Generate Growth of Invested Capital Stacked Area Chart.
    
    - Stacked Area: Portfolio Value by Asset Class (includes CASH)
    - Dashed Line: Total Cash Invested (Cumulative External Flows)
    """
    growth_data = calculate_growth_of_capital_data(data, end_date=end_date)
    ts_df = growth_data["time_series"]
    
    if ts_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Color mapping for asset classes (aligned with GLOBAL_PALETTE where possible)
    color_map = {
        "US Large Cap": GLOBAL_PALETTE[0],
        "US Growth": GLOBAL_PALETTE[6],
        "US Small Cap": GLOBAL_PALETTE[2],
        "International Equity": GLOBAL_PALETTE[8],
        "Gold / Precious Metals": GLOBAL_PALETTE[10],
        "Digital Assets": GLOBAL_PALETTE[4],
        "Fixed Income": GLOBAL_PALETTE[1],
        "US Bonds": GLOBAL_PALETTE[1],
        "CASH": "#D3D3D3", # Keep light gray for Cash
        "Total": "#000000" # Keep black for Total
    }
    
    # 1. Prepare Data for Stacking (Everything EXCEPT Total)
    # If filter_value == "Total" or "All", show all components stacked.
    
    if filter_value in ["Total", "All"]:
        stack_df = ts_df[ts_df["Asset Class"] != "Total"].copy()
        # Sort by Asset Class for consistent coloring order
        stack_df = stack_df.sort_values(["Asset Class", "Date"])
        
        # Get list of classes to plot
        plot_classes = stack_df["Asset Class"].unique()
        
        # Add Stacked Area Traces (Portfolio Value)
        for ac in plot_classes:
            ac_data = stack_df[stack_df["Asset Class"] == ac]
            color = color_map.get(ac, "#808080")
            
            fig.add_trace(go.Scatter(
                x=ac_data["Date"],
                y=ac_data["Portfolio Value"],
                mode='lines',
                name=ac,
                stackgroup='one', # Enable Stacking
                line=dict(width=0.5, color=color),
                fillcolor=color,
                hovertemplate=(
                    f"<b>{ac}</b>" + 
                    ": %{y:$,.2f} (Inv: %{customdata[0]:$,.2f})<extra></extra>"
                ),
                customdata=ac_data[["Cash Invested"]]
            ))
            
        # Add Total Cash Invested Line (Dashed Overlay)
        total_data = ts_df[ts_df["Asset Class"] == "Total"].sort_values("Date")
        if not total_data.empty:
            fig.add_trace(go.Scatter(
                x=total_data["Date"],
                y=total_data["Cash Invested"],
                mode='lines',
                name="Total Net Invested", 
                line=dict(color="#FFA500", width=3, dash='dash'),
                hovertemplate=(
                    "<b>Total Net Invested</b>: %{y:$,.2f}<extra></extra>"
                )
            ))
            
    else:
        # Specific Asset Class View
        # Area chart for Value, Dashed line for Invested
        ac_data = ts_df[ts_df["Asset Class"] == filter_value].sort_values("Date")
        if not ac_data.empty:
            color = color_map.get(filter_value, "#808080")
            
            # Area (Value)
            fig.add_trace(go.Scatter(
                x=ac_data["Date"],
                y=ac_data["Portfolio Value"],
                mode='lines',
                name=f"{filter_value} Value",
                fill='tozeroy',
                line=dict(color=color, width=2),
                hovertemplate="<b>Value</b>: %{y:$,.2f}<extra></extra>"
            ))
            
            # Line (Invested)
            fig.add_trace(go.Scatter(
                x=ac_data["Date"],
                y=ac_data["Cash Invested"],
                mode='lines',
                name="Net Invested", 
                line=dict(color="#FFA500", width=2, dash='dash'),
                hovertemplate="<b>Invested</b>: %{y:$,.2f}<extra></extra>"
            ))

    fig.update_layout( 
        
        yaxis_title="Value ($)",
        template="plotly_white" if theme == "light" else "plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            title_text="Legend",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="Grey",
            borderwidth=1
        )
    )
    
    return fig

def get_growth_of_capital_table_data(data):
    """
    Generate Growth of Invested Capital summary table data.
    Returns DataFrame with latest snapshot by asset class.
    """
    growth_data = calculate_growth_of_capital_data(data)
    summary_df = growth_data["summary"]
    
    if summary_df.empty:
        return pd.DataFrame()
    
    # Format for display
    display_df = summary_df.copy()
    display_df["Cash Invested"] = display_df["Cash Invested"].apply(fmt_dollar_clean)
    display_df["Portfolio Value"] = display_df["Portfolio Value"].apply(fmt_dollar_clean)
    display_df["Growth"] = display_df["Growth"].apply(fmt_dollar_clean)
    display_df["Growth %"] = display_df["Growth %"].apply(lambda x: f"{x:.2f}%")
    
    # Add Meta Columns for Audit (Growth % is effectively SI return with Start=0)
    # Start = 0 (assuming from inception)
    # End = Portfolio Value
    # Flow = Cash Invested (Net External)
    # Income = 0 (Included in PV/Growth)
    
    # Note: Using raw numeric values before formatting
    summary_df["meta_Growth %_start"] = 0.0
    summary_df["meta_Growth %_end"] = summary_df["Portfolio Value"]
    summary_df["meta_Growth %_flow"] = summary_df["Cash Invested"]
    summary_df["meta_Growth %_inc"] = 0.0
    summary_df["meta_Growth %_denom"] = summary_df["Cash Invested"]

    # Select and order columns (include meta)
    cols = ["Asset Class", "Cash Invested", "Portfolio Value", "Growth", "Growth %"]
    meta_cols = [c for c in summary_df.columns if c.startswith("meta_")]
    
    # Format for display
    display_df = summary_df.copy()
    display_df["Cash Invested"] = display_df["Cash Invested"].apply(fmt_dollar_clean)
    display_df["Portfolio Value"] = display_df["Portfolio Value"].apply(fmt_dollar_clean)
    display_df["Growth"] = display_df["Growth"].apply(fmt_dollar_clean)
    display_df["Growth %"] = display_df["Growth %"].apply(lambda x: f"{x:.2f}%")
    
    display_df = display_df[cols + meta_cols]
    
    # Ensure Total row is at the bottom
    total_row = display_df[display_df["Asset Class"] == "Total"]
    other_rows = display_df[display_df["Asset Class"] != "Total"]
    display_df = pd.concat([other_rows, total_row], ignore_index=True)
    
    return display_df

def get_cash_recon_pl(data, horizons):
    """
    Calculate Cash / Recon P/L for each horizon.
    
    Cash/Recon P/L = Portfolio Total P/L - Sum(All Ticker P/Ls)
    
    This captures cash interest, dividends in transit, fees, and other
    non-security-specific P/L components.
    
    Args:
        data: Data cache dict
        horizons: List of horizon strings (e.g. ["1D", "1W", "MTD", ...])
    
    Returns:
        Dict mapping horizon -> cash_recon_pl value (or None if N/A)
    """
    pv = data["pv"]
    inception_date = data["inception_date"]
    cf_ext = data["cf_ext"]
    pl_si = data["pl_si"]
    sec_table = data["sec_table"]  # Full table (not filtered)
    tx_raw = data["tx_raw"]
    dividends = data["dividends"]
    prices = data["prices"]
    
    cash_recon = {}
    
    for h in horizons:
        # 1. Get Portfolio Total P/L (External Flows)
        if h == "SI":
            port_pl = pl_si
        else:
            port_pl = calculate_horizon_pl(pv, inception_date, cf_ext, h)
        
        if port_pl is None:
            cash_recon[h] = None
            continue
        
        # 2. Sum Ticker P/Ls (Internal Flows)
        sum_ticker_pl = 0.0
        
        # Iterate all non-CASH tickers (use full table to match Portfolio PL)
        all_tickers = sec_table[sec_table["ticker"] != "CASH"]["ticker"].unique()
        
        as_of_dt = pv.index.max()
        if h == "SI":
            raw_start = None
        else:
            raw_start = get_portfolio_horizon_start(pv, inception_date, h)
        
        # FIX: Use pv.index.min() (first trading day) instead of inception_date for SI
        # This ensures ticker P/L boundaries match portfolio P/L boundaries exactly
        pv_start_date = pv.index.min()
        
        for t in all_tickers:
            val = calculate_ticker_pl(
                t, h, prices, as_of_dt, tx_raw, sec_table, raw_start, dividends=dividends,
                portfolio_inception=pv_start_date if h == "SI" else None
            )
            if val is not None:
                sum_ticker_pl += val
        
        # 3. Diff is Cash / Recon P/L
        diff = port_pl - sum_ticker_pl
        cash_recon[h] = diff
    
    return cash_recon

def get_monthly_contribution_schedule(data):
    """
    Generate Illustrative Monthly Contribution Schedule.
    
    EXACT REPLICATION of generate_report.py lines 477-507.
    Uses already-computed to_contrib values from sec_table_current.
    
    Returns:
        DataFrame with columns: Ticker, Asset Class, Gap to Target, Monthly Contrib, Share of Monthly
        Also returns footer text and whether schedule is empty
    """
    sec_current = data["sec_table_current"].copy()
    holdings = data["holdings"]
    
    if sec_current.empty:
        return pd.DataFrame(), "", True
    
    # Ensure to_contrib is calculated (matches PDF line 462)
    if "to_contrib" not in sec_current.columns:
        # Calculate if missing: amount needed to reach target weight
        if "allocation" not in sec_current.columns:
            sec_current["allocation"] = sec_current["weight"] * 100
        if "target_pct" not in sec_current.columns:
            # Merge from holdings
            sec_current = sec_current.merge(
                holdings[["ticker", "target_pct"]],
                on="ticker",
                how="left"
            )
            sec_current["target_pct"] = sec_current["target_pct"].fillna(0.0)
        
        total_value = sec_current["market_value"].sum()
        sec_current["to_contrib"] = np.maximum(
            ((sec_current["target_pct"] - sec_current["allocation"]) / 100) * total_value, 
            0
        )
    
    # Filter out holdings with zero to_contrib (EXACT match to PDF line 480)
    monthly_df = sec_current.copy()
    monthly_df["to_contrib_numeric"] = monthly_df["to_contrib"]
    monthly_df = monthly_df[monthly_df["to_contrib_numeric"] > 0].copy()
    
    if monthly_df.empty:
        return pd.DataFrame(), "", True
    
    # Use configurable monthly contribution from config (EXACT match to PDF line 484)
    total_monthly = TARGET_MONTHLY_CONTRIBUTION
    total_gap = monthly_df["to_contrib_numeric"].sum()
    
    # Calculate monthly contrib and share (EXACT match to PDF lines 485-486)
    monthly_df["monthly_contrib"] = monthly_df["to_contrib_numeric"] / total_gap * total_monthly
    monthly_df["share_of_monthly"] = monthly_df["monthly_contrib"] / total_monthly * 100
    
    # Build display dataframe
    display_df = pd.DataFrame({
        "Ticker": monthly_df["ticker"],
        "Asset Class": monthly_df["asset_class"],
        "Gap to Target": monthly_df["to_contrib_numeric"].apply(fmt_dollar_clean),
        "Monthly Contrib": monthly_df["monthly_contrib"].apply(fmt_dollar_clean),
        "Share of Monthly": monthly_df["share_of_monthly"].apply(lambda x: f"{x:.1f}%"),
        
        # Meta Columns for Audit
        "meta_Monthly Contrib_gap": monthly_df["to_contrib_numeric"],
        "meta_Monthly Contrib_total_gap": total_gap,
        "meta_Monthly Contrib_total_monthly": total_monthly
    })
    
    # Footer text (EXACT match to PDF lines 500-504)
    footer = (f"At approximately ${total_monthly:,.0f}/month, this schedule allocates contributions "
              "proportionally to each holding's gap. It would take about "
              f"{total_gap / total_monthly:.1f} months to close all gaps, assuming flat markets.")
    
    return display_df, footer, False

def get_asset_class_allocation_table(data):
    """
    Generate Asset Class Allocation Table.
    
    EXACT REPLICATION of generate_report.py lines 512-567.
    Shows Value, Actual %, Target %, and Delta % for each asset class.
    
    Returns:
        DataFrame with columns: Asset Class, Value ($), Actual %, Target %, Delta %
    """
    sec_current = data["sec_table_current"].copy()
    holdings = data["holdings"]
    
    if sec_current.empty:
        return pd.DataFrame()
    
    # Asset class short name mapping (matches PDF)
    asset_class_map = {
        "US Large Cap": "US LC",
        "US Growth": "US Growth",
        "US Small Cap": "US SC",
        "International Equity": "INTL EQTY",
        "Gold / Precious Metals": "GOLD",
        "Digital Assets": "DIGITAL",
        "US Bonds": "US Bonds",
        "CASH": "CASH",
        "Fixed Income": "FI"
    }
    
    # Use no_cash version for calculations (matches PDF line 515)
    sec_no_cash = sec_current.copy()
    total_value = sec_no_cash["market_value"].sum()
    
    # Merge short asset class and target_pct from holdings (EXACT match to PDF lines 517-522)
    sec_merge = sec_no_cash.merge(
        holdings[["ticker", "target_pct"]],
        on="ticker",
        how="left",
        suffixes=("", "_holdings")
    )
    
    # Compute actual allocations per asset class (EXACT match to PDF lines 525-533)
    asset_group = (
        sec_merge.groupby("asset_class")
        .agg(
            value=("market_value", "sum"),
            target_pct=("target_pct", "sum")
        )
        .reset_index()
    )
    
    # Compute actual percentage allocation (EXACT match to PDF line 536)
    asset_group["actual_pct"] = asset_group["value"] / total_value * 100
    
    # Compute delta (EXACT match to PDF line 539)
    asset_group["delta_pct"] = asset_group["actual_pct"] - asset_group["target_pct"]
    
    # Map to short names
    asset_group["asset_class_short"] = asset_group["asset_class"].map(lambda x: asset_class_map.get(x, x))
    
    # Format columns for display (EXACT match to PDF lines 542-544)
    asset_group["actual_pct_fmt"] = asset_group["actual_pct"].map(lambda x: f"{x:.2f}%")
    asset_group["target_pct_fmt"] = asset_group["target_pct"].map(lambda x: f"{x:.2f}%")
    asset_group["delta_pct_fmt"] = asset_group["delta_pct"].map(lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
    
    # Merge with class_df to get Meta Columns (for Audit)
    # class_df has Asset Class level meta data for SI, 1M, etc.
    # We'll attach SI meta data as the primary "explanation" for the position value
    class_df = data.get("class_df")
    if class_df is not None and not class_df.empty:
        # Merge on asset_class (not short name)
        asset_group = asset_group.merge(
            class_df[["asset_class", "meta_SI_start", "meta_SI_end", "meta_SI_flow", "meta_SI_inc", "meta_SI_denom"]],
            on="asset_class",
            how="left"
        )

    # Build table rows (EXACT match to PDF lines 547-555)
    table_rows = []
    for _, row in asset_group.iterrows():
        # Build breakdown for audit
        ac_name = row["asset_class"]
        subset = sec_merge[sec_merge["asset_class"] == ac_name][["ticker", "market_value"]]
        subset = subset.sort_values("market_value", ascending=False)
        
        breakdown_list = []
        for _, s_row in subset.iterrows():
            breakdown_list.append({
                "ticker": s_row["ticker"],
                "value": s_row["market_value"]
            })

        r_data = {
            "Asset Class": row["asset_class_short"],
            "Value ($)": fmt_dollar_clean(row["value"]),
            "Actual %": row["actual_pct_fmt"],
            "Target %": row["target_pct_fmt"],
            "Delta %": row["delta_pct_fmt"],
            # Meta Columns (using SI as default context)
            "meta_Value ($)_start": row.get("meta_SI_start", 0),
            "meta_Value ($)_end": row.get("meta_SI_end", 0),
            "meta_Value ($)_flow": row.get("meta_SI_flow", 0),
            "meta_Value ($)_inc": row.get("meta_SI_inc", 0),
            "meta_Value ($)_denom": row.get("meta_SI_denom", 0),
            # NEW: Breakdown meta
            "meta_Value ($)_breakdown": breakdown_list
        }
        table_rows.append(r_data)
    
    # Add TOTAL row (EXACT match to PDF lines 558-565)
    total_value_sum = asset_group["value"].sum()
    total_actual_pct = asset_group["actual_pct"].sum()
    total_target_pct = asset_group["target_pct"].sum()
    total_delta_pct = asset_group["delta_pct"].sum()
    
    table_rows.append({
        "Asset Class": "TOTAL",
        "Value ($)": fmt_dollar_clean(total_value_sum),
        "Actual %": f"{total_actual_pct:.2f}%",
        "Target %": f"{total_target_pct:.2f}%",
        "Delta %": f"{total_delta_pct:+.2f}%"
    })
    
    return pd.DataFrame(table_rows)

# ============================================================
# FRONGELLO ATTRIBUTION HELPERS
# ============================================================

def _get_daily_asset_class_series(data):
    """
    Generates a daily DataFrame of MV, Flow, and Income for each Asset Class.
    Used for Frongello Attribution Linking.
    
    Returns:
        pd.DataFrame with MultiIndex (Date, Asset Class) and columns:
        [Start_MV, End_MV, Net_Flow, Income]
    """
    pv = data["pv"]
    tx_raw = data["tx_raw"]
    holdings = data["holdings"]
    prices = data["prices"]
    dividends = data["dividends"]
    cf_ext = data.get("cf_ext")
    
    if cf_ext is None or cf_ext.empty: return pd.DataFrame()
    
    # 1. Timeline Setup
    market_start = data["inception_date"]
    end_date = pv.index.max()
    full_idx = pd.date_range(start=market_start, end=end_date, freq="D")
    
    # PV Safety Net
    cum_ext_flows = cf_ext.groupby("date")["amount"].sum().cumsum().reindex(full_idx, method='ffill').fillna(0.0)
    pv_aligned = pv.reindex(full_idx).ffill()
    
    # Use np.where: if pv is zero/missing AND cum_ext_flows is positive, overwrite with cum_ext_flows
    pv_aligned_values = np.where(
        (pv_aligned.isnull() | (pv_aligned == 0)) & (cum_ext_flows > 0),
        cum_ext_flows,
        pv_aligned
    )
    pv_aligned = pd.Series(pv_aligned_values, index=full_idx)

    # 2. Map Tickers to Asset Classes
    ac_map = holdings.set_index("ticker")["asset_class"].to_dict()
    
    # Normalize dates
    if not tx_raw.empty: tx_raw["date"] = pd.to_datetime(tx_raw["date"]).dt.normalize()
    if not dividends.empty: dividends["date"] = pd.to_datetime(dividends["date"]).dt.normalize()
    
    # 3. Daily Shares per Ticker
    if not tx_raw.empty:
        shares_delta = tx_raw.pivot_table(index="date", columns="ticker", values="shares", aggfunc="sum").fillna(0.0)
        
        pre_market_shares = shares_delta[shares_delta.index < market_start].sum()
        if not pre_market_shares.eq(0).all():
            if market_start in shares_delta.index:
                shares_delta.loc[market_start] += pre_market_shares
            else:
                shares_delta.loc[market_start] = pre_market_shares
                
        shares_delta = shares_delta.reindex(full_idx, fill_value=0.0)
        shares_daily = shares_delta.cumsum()
    else:
        shares_daily = pd.DataFrame(index=full_idx)
        
    # 4. Daily Prices per Ticker (ffill)
    px_daily = prices.reindex(full_idx).ffill()
    
    # 5. Calculate Daily MV per Asset Class
    common_tickers = list(set(shares_daily.columns) & set(px_daily.columns))
    
    unique_ac = list(set(ac_map.values()) | {"CASH"})
    mv_daily_ac = pd.DataFrame(0.0, index=full_idx, columns=unique_ac)
    
    if common_tickers:
        val_daily = shares_daily[common_tickers] * px_daily[common_tickers]
        for t in common_tickers:
            ac = ac_map.get(t, "Other")
            mv_daily_ac[ac] += val_daily[t].fillna(0.0)
            
    # 6. CASH Asset Class Handling
    sum_sec_mv = mv_daily_ac.drop(columns=["CASH"], errors="ignore").sum(axis=1)
    mv_daily_ac["CASH"] = pv_aligned - sum_sec_mv

    # 7. Daily Flows (Net Internal) per Asset Class
    flow_daily_ac = pd.DataFrame(0.0, index=full_idx, columns=mv_daily_ac.columns)
    
    def add_flow(date, col, amount):
        target_date = date if date >= market_start else market_start
        if target_date in flow_daily_ac.index:
            flow_daily_ac.loc[target_date, col] += amount

    if not tx_raw.empty:
        tx_mapped = tx_raw.copy()
        tx_mapped["asset_class"] = tx_mapped["ticker"].map(ac_map).fillna("Other")
        sec_tx = tx_mapped[tx_mapped["ticker"] != "CASH"]
        if not sec_tx.empty:
            grp = sec_tx.groupby(["date", "asset_class"])["amount"].sum()
            for (d, ac), amt in grp.items():
                add_flow(d, ac, -amt)
                add_flow(d, "CASH", amt)
                    
    ext_grp = cf_ext.groupby("date")["amount"].sum()
    for d, amt in ext_grp.items():
        add_flow(d, "CASH", amt)

    # 9. Daily Income (Dividends)
    inc_daily_ac = pd.DataFrame(0.0, index=full_idx, columns=mv_daily_ac.columns)
    if not dividends.empty:
        div_mapped = dividends.copy()
        div_mapped["asset_class"] = div_mapped["ticker"].map(ac_map).fillna("Other")
        grp = div_mapped.groupby(["date", "asset_class"])["amount"].sum()
        for (d, ac), amt in grp.items():
            if d in inc_daily_ac.index:
                inc_daily_ac.loc[d, ac] += amt
                # FIX: Record dividend as a flow INTO Cash so it doesn't look like a loss in Recon
                if d in flow_daily_ac.index:
                    flow_daily_ac.loc[d, "CASH"] += amt
                
    # 10. Construct Final DataFrame
    mv_stack = mv_daily_ac.stack().rename("End_MV")
    flow_stack = flow_daily_ac.stack().rename("Net_Flow")
    inc_stack = inc_daily_ac.stack().rename("Income")
    
    mv_shift = mv_daily_ac.shift(1).stack().rename("Start_MV")
    
    df_combined = pd.concat([mv_shift, mv_stack, flow_stack, inc_stack], axis=1).fillna(0.0)

    # Strict Day 0 Start
    first_date = df_combined.index.get_level_values(0).min()
    df_combined.loc[first_date, "Start_MV"] = 0.0
    
    return df_combined

def _calculate_frongello_linking(data, start_date=None, end_date=None):
    """
    Implements Frongello Attribution Linking Algorithm.
    
    Returns:
        pd.DataFrame with Asset Class breakdown:
        [Asset Class, Effect, Contribution (%), meta_...]
    """
    # 1. Get Daily Series
    daily_df = _get_daily_asset_class_series(data)
    if daily_df.empty: return pd.DataFrame()
    
    # 2. Filter Date Range
    if start_date is None: start_date = daily_df.index.get_level_values(0).min()
    if end_date is None: end_date = daily_df.index.get_level_values(0).max()
    
    # Slicing MultiIndex (Date, AC)
    # Using slice(start, end) on the first level
    mask = (daily_df.index.get_level_values(0) >= pd.Timestamp(start_date)) & \
           (daily_df.index.get_level_values(0) <= pd.Timestamp(end_date))
    window = daily_df[mask].copy()
    
    if window.empty: return pd.DataFrame()
    
    # 3. Pivot back to Wide Format for vectorized calculation
    # We need (Date x AC) matrices
    start_mv = window["Start_MV"].unstack(level=1).fillna(0.0)
    end_mv = window["End_MV"].unstack(level=1).fillna(0.0)
    flows = window["Net_Flow"].unstack(level=1).fillna(0.0)
    income = window["Income"].unstack(level=1).fillna(0.0)
    
    # 4. Calculate Portfolio Aggregates (Daily)
    port_start = start_mv.sum(axis=1)
    port_end = end_mv.sum(axis=1)
    # Note: Flows here include External (to CASH) and Internal (Sec <-> Cash cancellation)
    # Sum(Net_Flows) across all ACs should equal External Flows (since Internal sum to 0)
    port_flow = flows.sum(axis=1) 
    port_inc = income.sum(axis=1) # Total Dividends
    
    # 5. Calculate Daily Portfolio Returns (Rp)
    # Isolate External Flows for TWR denominator
    port_ext_flow = port_flow - port_inc
    
    # Correct Denominator for TWR
    denom_p = port_start + port_ext_flow
    denom_p_safe = denom_p.replace(0, np.nan).fillna(1.0)
    
    # Calculate Portfolio Return (Standard GIPS)
    # R = (End - (Start + Ext_Flow)) / (Start + Ext_Flow)
    r_portfolio = (port_end - denom_p) / denom_p_safe
    
    # 6. Calculate Linking Factors (Geometric)
    # Factor_t = Product_{j=1 to t-1} (1 + R_j)
    # Shift R_p by 1 to get Prev Returns, then Cumprod
    # Fill first day with 1.0
    
    prev_r = (1 + r_portfolio).shift(1).fillna(1.0)
    link_factors = prev_r.cumprod()
    
    # 7. Calculate Daily Asset Class Returns/Effects
    # Effect ($) = End - Start - Flow + Income?
    # Actually, in Frongello, we attribute the Portfolio Return.
    # r_i = (End_i - Start_i - Flow_i) / (Start_p + Flow_p) ?? No.
    #
    # 7. Calculate Daily Asset Class Returns/Effects
    # Logic: Capital at risk today = yesterday's value + today's flows
    capital_at_risk = start_mv + flows
    effect_daily = end_mv - capital_at_risk + income
    
    # Correct the Attribution Weighting
    contrib_daily = effect_daily.div(denom_p_safe, axis=0)
    
    # 8. Apply Linking
    # Linked_Contrib_it = c_it * LinkFactor_t
    linked_daily = contrib_daily.mul(link_factors, axis=0)
    
    # 9. Sum over the period
    total_linked_contrib = linked_daily.sum()
    total_effect = effect_daily.sum() # Simple Sum of P/L ($)
    
    # 10. Compile Results
    results = []
    for ac in total_linked_contrib.index:
        results.append({
            "Asset Class": ac,
            "Effect": total_effect[ac],
            "Contribution (%)": total_linked_contrib[ac] * 100.0,
            
            # Audit Meta
            "meta_frongello_sum_factors": link_factors.sum(),
            "meta_frongello_avg_denom": denom_p.mean()
        })
        
    df_res = pd.DataFrame(results).sort_values("Contribution (%)", ascending=False)
    
    return df_res


# ============================================================
# AUDIT TRAIL HELPERS (Server-Side On-Demand)
# ============================================================

def fetch_audit_details(request_data):
    """
    Server-side fetch for Audit Modal details.
    Enriches the request with detailed breakdown data on demand.
    """
    data = get_data()
    if not data or not request_data: return request_data
    
    grid_id = request_data.get("gridId", "")
    col_id = request_data.get("colId", "")
    row_data = request_data.get("rowData", {})
    
    # ------------------------------------------------
    # TYPE 5: TWR AUDIT (Snapshot Return Columns)
    # ------------------------------------------------
    if "snapshot-grid" in str(grid_id) and col_id == "Return":
        horizon = row_data.get("Horizon")
        if not horizon: return request_data
        
        # Map Display Label to Engine Code
        if horizon == "Since Inception": horizon = "SI"
        
        pv = data["pv"]
        cf_ext = data["cf_ext"]
        inception_date = data["inception_date"]
        
        # Calculate Start/End
        start = get_portfolio_horizon_start(pv, inception_date, horizon)
        end = pv.index.max()
        
        # Handle Insufficient Data (Enforce TWR View)
        if start is None: 
            request_data["twr_monthly_breakdown"] = []
            return request_data
        
        # Re-calculate Daily TWR Series using TRUSTED financial_math logic
        twr_val, daily_breakdown = compute_period_twr(pv, cf_ext, start, end, return_breakdown=True)
        
        if not daily_breakdown: 
            request_data["twr_monthly_breakdown"] = []
            return request_data
        
        # Aggregate to Monthly
        # Group by Year-Month
        monthly_map = defaultdict(list)
        for item in daily_breakdown:
            d = item["date"]
            # key = YYYY-MM
            key = f"{d.year}-{d.month:02d}"
            monthly_map[key].append(item)
            
        monthly_table = []
        
        sorted_keys = sorted(monthly_map.keys())
        for k in sorted_keys:
            items = monthly_map[k]
            # Link returns geometrically
            # R_mo = (1+r1)*(1+r2)... - 1
            mo_factor = 1.0
            for it in items:
                mo_factor *= (1.0 + it["return"])
            
            mo_ret = mo_factor - 1.0
            
            # Helper for display date (Last date in month)
            last_date = items[-1]["date"]
            
            monthly_table.append({
                "month_str": last_date.strftime("%Y-%m"), # Sortable
                "display_date": last_date.strftime("%b %Y"),
                "return": mo_ret,
                "factor": mo_factor
            })
            
        # Add to request_data (simulating it came from client)
        request_data["twr_monthly_breakdown"] = monthly_table
        request_data["meta_Return_start"] = row_data.get("meta_Return_start")
        request_data["meta_Return_end"] = row_data.get("meta_Return_end")
        request_data["meta_Return_flow"] = row_data.get("meta_Return_flow")
        
        return request_data
        
    return request_data

def _calculate_residual_return(data, df_explained):
    """
    Helper to calculate the 'residual' return (Cash/Recon).
    Ensures P/L Source of Truth is clipped to exactly the same date range as the Attribution.
    """
    # 1. TWR Residual
    port_twr = data["twr_si"]
    if pd.isna(port_twr): port_twr = 0.0
    
    explained_twr_pct = df_explained["Contribution (%)"].sum()
    residual_pct = (port_twr * 100.0) - explained_twr_pct
    
    # 2. P/L Residual
    pv = data["pv"]
    cf_ext = data.get("cf_ext")
    
    if not pv.empty:
        # Determine the EXACT end date of the attribution
        last_date = pv.index.max()
        current_val = float(pv.iloc[-1])
        
        # CRITICAL: Only sum flows that happened on or before the last price date
        # This matches the Attribution loop which stops at last_date
        if cf_ext is not None and not cf_ext.empty:
            relevant_flows = cf_ext[cf_ext["date"] <= last_date]
            total_invested = relevant_flows["amount"].sum()
        else:
            total_invested = 0.0
            
        pl_si_robust = current_val - total_invested
    else:
        pl_si_robust = 0.0
        
    explained_pl = df_explained["Effect"].sum()
    residual_pl = pl_si_robust - explained_pl
    
    return residual_pct, residual_pl

def get_si_attribution_summary(data):
    """
    Calculates the cumulative performance effect for every asset class from inception.
    Upgraded to use Frongello Attribution Linking for geometric accuracy.
    """
    # Use the new Frongello Engine
    df = _calculate_frongello_linking(data)
    
    if df.empty: return pd.DataFrame()
    
    # Calculate Residual / Recon
    residual_pct, residual_pl = _calculate_residual_return(data, df)
    
    # Add Residual Row if significant
    if abs(residual_pl) > 0.01 or abs(residual_pct) > 0.01:
        df = pd.concat([df, pd.DataFrame([{
            "Asset Class": "Recon/Residual",
            "Effect": residual_pl,
            "Contribution (%)": residual_pct,
            "meta_frongello_sum_factors": 0,
            "meta_frongello_avg_denom": 0
        }])], ignore_index=True)
        
    return df.sort_values("Contribution (%)", ascending=False)

def get_active_strategy_table(data):
    """
    Returns summary table of Active Risk metrics (Beta, Tracking Error)
    compared to major benchmarks.
    """
    benchmarks = {
        "S&P 500": "SPY",
        "Total World": "VT",
        "US Bonds": "BND"
    }
    
    rows = []
    
    for name, ticker in benchmarks.items():
        metrics = calculate_active_metrics(data, ticker)
        
        beta = metrics.get("beta")
        te = metrics.get("te")
        
        beta_str = f"{beta:.2f}" if isinstance(beta, (int, float)) else "N/A"
        te_str = f"{te:.2f}%" if isinstance(te, (int, float)) else "N/A"
        
        rows.append({
            "Benchmark": f"{name} ({ticker})",
            "Beta": beta_str,
            "Tracking Error": te_str
        })
        
    return pd.DataFrame(rows)
