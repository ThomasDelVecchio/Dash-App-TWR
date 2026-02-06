import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from collections import defaultdict
import io
import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

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
    fetch_etf_sectors,
    _METADATA_CACHE,
    PRICE_CACHE_EXPIRY_HOURS,
    clear_price_cache
)
from financial_math import (
    get_portfolio_horizon_start,
    compute_period_twr,
    fv_lump,
    fv_contrib,
    modified_dietz_for_ticker_window,
    annualize_return,
    is_annualized,
    get_effective_anchor_date,
    is_market_holiday
)
from tax_engine import calculate_tax_optimized_sales
from report_formatting import fmt_pct_clean, fmt_dollar_clean
import config
from config import TARGET_MONTHLY_CONTRIBUTION, GLOBAL_PALETTE, RISK_FREE_RATE, TAX_RATE_LT, TAX_RATE_ST
from components.backtest_engine import (
    get_strategy_backtest_results,
    get_strategy_backtest_growth_chart,
    get_strategy_backtest_drawdown_chart,
    get_strategy_backtest_risk_return_chart,
)

# ============================================================
# GOOGLE DRIVE CONFIG
# ============================================================
TOKEN_FILE = 'token.json'
PARENT_FOLDER_ID = '1deTQ3M6SIGcFfZM3vvzTjaq12kZZcWxj'

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

# ============================================================
# HELPER: Dynamic UI Labels
# ============================================================
def get_display_label_for_1d(report_date=None):
    """
    Returns '1D' if report_date is a trading day.
    Returns 'Last Close' (or 'Last Trading Day') if report_date is a Weekend/Holiday.
    Used to prevent misleading '1D Return' labels on Sundays showing Friday's return.
    """
    if report_date is None:
        report_date = datetime.now()
        
    ts = pd.Timestamp(report_date)
    
    if is_market_holiday(ts):
        return "Last Close"
    return "1D"

# --- UPDATE FUNCTION ---
def _get_drive_folder_id(service, folder_name, parent_id="root", create_if_missing=False):
    """Resolve (or create) a Google Drive folder by name under a parent."""
    query = (
        "mimeType='application/vnd.google-apps.folder' "
        f"and name='{folder_name}' "
        f"and '{parent_id}' in parents "
        "and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]

    if not create_if_missing:
        return None

    folder_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=folder_metadata, fields="id").execute()
    return folder.get("id")


def send_to_drive(content, filename, mimetype='text/csv', parent_folder_id=None, parent_folder_name=None, create_folder=False):
    """
    Uploads using the User's credentials (OAuth) to bypass Service Account limits.
    Supports both string content (auto-encoded) and binary/bytes content.
    """
    try:
        # Check if the token exists (created by authorize.py)
        if not os.path.exists(TOKEN_FILE):
            return "❌ Error: token.json missing. Run authorize.py first."

        # Authenticate as YOU (not the bot)
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, ['https://www.googleapis.com/auth/drive.file'])
        service = build('drive', 'v3', credentials=creds)

        target_parent_id = parent_folder_id or PARENT_FOLDER_ID
        if parent_folder_name:
            # IMPORTANT: resolve under the intended parent (not root),
            # so we find the existing folder instead of creating a duplicate.
            resolved_id = _get_drive_folder_id(
                service,
                parent_folder_name,
                parent_id=target_parent_id,
                create_if_missing=create_folder,
            )
            if resolved_id:
                target_parent_id = resolved_id

        file_metadata = {
            'name': os.path.basename(filename),
            'parents': [target_parent_id]
        }
        
        # dynamic buffer handling
        if isinstance(content, str):
            media_io = io.BytesIO(content.encode('utf-8'))
        elif isinstance(content, bytes):
            media_io = io.BytesIO(content)
        else:
            # Assume it's already a file-like object
            media_io = content
            
        media = MediaIoBaseUpload(
            media_io,
            mimetype=mimetype,
            resumable=True
        )

        service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        return "✅ Exported to Drive"

    except Exception as e:
        return f"❌ Export failed: {str(e)}"

def get_data():
    """Retrieve cached data, initializing if necessary."""
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = run_analytics_engine()
    return _DATA_CACHE

def refresh_data(end_date=None, force_price_refresh: bool = False):
    """Force refresh of the data cache."""
    global _DATA_CACHE
    if force_price_refresh:
        clear_price_cache()
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

    # Always include common benchmarks to avoid repeated fetches in callbacks
    benchmark_universe = ["SPY", "VTI", "VUG", "AOA", "AOR", "AOK", "QQQ"]
    for bm in benchmark_universe:
        if bm not in all_tickers: all_tickers.append(bm)
        
    prices_cached = fetch_price_history(all_tickers) if all_tickers else pd.DataFrame()
    benchmark_prices_adj = fetch_price_history(benchmark_universe, use_adj_close=True) if benchmark_universe else pd.DataFrame()
    
    # FIX: Align prices with end_date (Time Machine & Weekend Handling)
    # Matches portfolio_engine.py logic to ensure P/L calc sees the same end date as PV
    if end_date is not None:
        # Use Effective Anchor (Last Trading Day) for price cache to ensure
        # Volatility and P/L calculations don't drag on weekends/holidays.
        target_end_date = get_effective_anchor_date(pd.Timestamp(end_date))
        prices_cached = prices_cached[prices_cached.index <= target_end_date]
        
        if not prices_cached.empty and prices_cached.index.max() < target_end_date:
            last_row = prices_cached.iloc[[-1]].copy()
            last_row.index = [target_end_date]
            prices_cached = pd.concat([prices_cached, last_row])

        if not benchmark_prices_adj.empty:
            benchmark_prices_adj = benchmark_prices_adj[benchmark_prices_adj.index <= target_end_date]
            if benchmark_prices_adj.index.max() < target_end_date:
                last_row = benchmark_prices_adj.iloc[[-1]].copy()
                last_row.index = [target_end_date]
                benchmark_prices_adj = pd.concat([benchmark_prices_adj, last_row])
    
    # Robustly extract errors from dataframe metadata
    errors = getattr(prices_cached, "attrs", {}).get("errors", [])

    # Surface cash settlement bridge notice in the UI (if applied)
    bridge_info = getattr(pv, "attrs", {}).get("cash_settlement_bridge")
    if bridge_info and isinstance(errors, list):
        bridge_amount = bridge_info.get("amount")
        if bridge_amount is not None:
            bridge_msg = (
                "Cash is settling from a recent external sale. "
                f"Synthetic CASH adjustment of ${bridge_amount:,.2f} is applied until settlement posts."
            )
            if bridge_msg not in errors:
                errors.append(bridge_msg)
    if errors: print(f"DEBUG: dash_wrappers found errors: {errors}")

    # Dynamic Risk Profile (Vol, Return, Correlation)
    dynamic_risk_return, dynamic_corr_matrix = _calculate_dynamic_risk_profile(
        prices_cached, sec_table_current, holdings, end_date
    )
    
    # Calculate Effective As-Of Date explicitly for UI Helpers
    # This ensures P/L calculations in helpers match the TWR calculation end date
    as_of_pv = pv.index.max() if not pv.empty else pd.Timestamp.now()
    effective_as_of = get_effective_anchor_date(as_of_pv)

    # Price data timestamps (for diagnostics)
    price_as_of = None
    if prices_cached is not None and not prices_cached.empty:
        try:
            price_as_of = prices_cached.index.max()
        except Exception:
            price_as_of = None
    price_fetched_at = None
    price_cache_source = None
    if prices_cached is not None and hasattr(prices_cached, "attrs"):
        price_fetched_at = prices_cached.attrs.get("fetched_at")
        price_cache_source = prices_cached.attrs.get("cache_source")

    benchmark_as_of = None
    if benchmark_prices_adj is not None and not benchmark_prices_adj.empty:
        try:
            benchmark_as_of = benchmark_prices_adj.index.max()
        except Exception:
            benchmark_as_of = None
    benchmark_fetched_at = None
    benchmark_cache_source = None
    if benchmark_prices_adj is not None and hasattr(benchmark_prices_adj, "attrs"):
        benchmark_fetched_at = benchmark_prices_adj.attrs.get("fetched_at")
        benchmark_cache_source = benchmark_prices_adj.attrs.get("cache_source")
    
    base_data = {
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
        "benchmark_prices_adj": benchmark_prices_adj,
        "errors": errors,
        "risk_return": dynamic_risk_return,
        "correlation_matrix": dynamic_corr_matrix,
        "effective_as_of": effective_as_of,
        "selected_end_date": end_date,
        "price_as_of": price_as_of,
        "benchmark_as_of": benchmark_as_of,
        "price_fetched_at": price_fetched_at,
        "benchmark_fetched_at": benchmark_fetched_at,
        "price_cache_source": price_cache_source,
        "benchmark_cache_source": benchmark_cache_source,
        "price_cache_expiry_hours": PRICE_CACHE_EXPIRY_HOURS
    }

    # Precompute heavy P/L tables once per refresh
    horizons = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "SI"]
    ticker_pl_cache = {h: get_ticker_pl_df(base_data, h) for h in horizons}

    asset_class_pl_cache = {h: {} for h in horizons}
    for ac in class_df["asset_class"].unique():
        for h in horizons:
            asset_class_pl_cache[h][ac] = calculate_asset_class_pl(
                ac,
                h,
                base_data["prices"],
                base_data["pv"],
                base_data["inception_date"],
                base_data["tx_raw"],
                base_data["sec_table"],
                base_data["dividends"],
                return_components=True,
                effective_as_of=base_data.get("effective_as_of")
            )

    base_data["ticker_pl_cache"] = ticker_pl_cache
    base_data["asset_class_pl_cache"] = asset_class_pl_cache

    return base_data

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
    based on the portfolio's actual 20-year history (Pro-Forma).
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

        # ARITHMETIC MEAN RETURN (For Sharpe Ratio - GIPS Standard)
        mean_daily_ret = series.mean()
        arith_ret = mean_daily_ret * 252 * 100.0

        # GEOMETRIC RETURN (For Total Return Display)
        # TTM Return (Last 252 trading days)
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
        
        # Calculate Sharpe Ratio (Arithmetic Mean - Rf) / Vol
        # Use centralized Risk Free Rate
        rf_pct = RISK_FREE_RATE * 100.0 # Convert scalar 0.04 to 4.0
        sharpe = (arith_ret - rf_pct) / vol if vol > 0 else 0.0

        dynamic_risk_return[ac] = {
            "return": ttm_ret,       # Geometric (for plotting Return vs Vol)
            "arith_return": arith_ret, # Arithmetic (for Sharpe context)
            "vol": vol,
            "sharpe": sharpe
        }
        
    # Add Fixed Benchmarks if missing (for gauge stability)
    if "Fixed Income" not in dynamic_risk_return:
        dynamic_risk_return["Fixed Income"] = {"return": 4.0, "arith_return": 4.0, "vol": 5.0, "sharpe": 0.0}
    if "US Large Cap" not in dynamic_risk_return:
         dynamic_risk_return["US Large Cap"] = {"return": 10.0, "arith_return": 10.0, "vol": 15.0, "sharpe": 0.4}

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

def calculate_efficiency_metrics(twr_series, start_date=None, end_date=None):
    """
    Calculates Sharpe and Sortino Ratios based on daily TWR series.
    Uses RISK_FREE_RATE from config.
    Returns dictionary with ratios AND components for Audit Trail.
    
    Args:
        twr_series: Daily TWR curve (Growth of $1)
        start_date: Optional period start for duration-aware annualization
        end_date: Optional period end for duration-aware annualization
        
    GIPS Compliance:
        - Only annualizes if period > 365 days (matches annualize_return() logic)
        - For sub-annual periods, returns cumulative metrics to avoid misleading figures
    """
    default_res = {
        "sharpe": "N/A", "sortino": "N/A",
        "vol": 0.0, "ret": 0.0, "rf": RISK_FREE_RATE
    }
    
    if twr_series.empty or len(twr_series) < 2:
        return default_res
        
    # Calculate Daily Returns from the Curve
    daily_rets = twr_series.pct_change().dropna()
    
    if daily_rets.empty:
        return default_res
    
    # [GIPS FIX] Filter for TRADING DAYS ONLY
    # Removing 0.0 returns from weekends/holidays prevents artificial dampening of Volatility.
    # This aligns the Annualization Factor (252) with the actual data frequency.
    if isinstance(daily_rets.index, pd.DatetimeIndex):
        trading_day_mask = ~daily_rets.index.map(is_market_holiday)
        daily_rets = daily_rets[trading_day_mask]

        if daily_rets.empty:
            return default_res

    # Determine if we should annualize based on period duration
    should_annualize = True
    if start_date is not None and end_date is not None:
        days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        years = days / 365.25
        # GIPS Compliance: Only annualize if > 1 year (matches annualize_return() logic)
        should_annualize = (years > 1.0)

    # 1. Volatility Calculation
    std_dev_daily = daily_rets.std()
    
    # 2. Return Calculation (Arithmetic Mean)
    mean_daily_ret = daily_rets.mean()
    
    # 3. Risk Free Rate
    rf = RISK_FREE_RATE
    rf_daily = (1 + rf) ** (1/252) - 1
    
    if should_annualize:
        # Full annualization for periods > 1 year
        vol_result = std_dev_daily * np.sqrt(252)
        ret_result = mean_daily_ret * 252
        rf_result = rf
        
        # 4. Sharpe Ratio (Annualized)
        excess_ret = ret_result - rf_result
        sharpe = excess_ret / vol_result if vol_result > 0 else 0.0
        
        # 5. Sortino Ratio (Annualized)
        daily_excess = daily_rets - rf_daily
        downside_rets = daily_excess[daily_excess < 0]
        
        if downside_rets.empty:
            sortino = 10.0  # Capped high value
        else:
            downside_dev_daily = np.sqrt((downside_rets ** 2).mean())
            downside_dev_ann = downside_dev_daily * np.sqrt(252)
            sortino = excess_ret / downside_dev_ann if downside_dev_ann > 0 else 0.0
    else:
        # Cumulative metrics for sub-annual periods (GIPS-compliant)
        # Don't annualize to avoid misleading inflation of ratios
        vol_result = std_dev_daily  # Daily volatility
        ret_result = mean_daily_ret  # Daily return
        rf_result = rf_daily  # Daily risk-free rate
        
        # 4. Sharpe Ratio (Daily Basis)
        excess_ret = ret_result - rf_result
        sharpe = excess_ret / vol_result if vol_result > 0 else 0.0
        
        # 5. Sortino Ratio (Daily Basis)
        daily_excess = daily_rets - rf_daily
        downside_rets = daily_excess[daily_excess < 0]
        
        if downside_rets.empty:
            sortino = 10.0  # Capped high value
        else:
            downside_dev_daily = np.sqrt((downside_rets ** 2).mean())
            sortino = excess_ret / downside_dev_daily if downside_dev_daily > 0 else 0.0
        
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "vol": vol_result,    # For display/audit (decimal, e.g. 0.15 = 15% or daily)
        "ret": ret_result,    # For display/audit (annualized or daily)
        "rf": rf_result       # For display/audit (annualized or daily)
    }

def calculate_active_metrics(data, benchmark_ticker="SPY"):
    """
    Calculates Beta and Tracking Error vs Benchmark.
    """
    twr_curve = _get_daily_twr_curve(data)
    if twr_curve.empty: return {"beta": "N/A", "te": "N/A"}
    
    # Get Benchmark Prices (prefer cached adj-close)
    prices = data.get("benchmark_prices_adj")
    if prices is None or prices.empty or benchmark_ticker not in prices.columns:
        prices = fetch_price_history([benchmark_ticker], use_adj_close=True)

    if benchmark_ticker not in prices.columns:
        return {"beta": "N/A", "te": "N/A"}
        
    bm_series = prices[benchmark_ticker].dropna()
    
    # [NEW LOGIC START]
    # Check for Start Date Alignment (Gateway)
    # If benchmark starts > 7 days after portfolio, it's invalid for "active" metrics of this portfolio
    port_start = twr_curve.index[0]
    bm_start = bm_series.index[0]
    
    if (bm_start - port_start).days > 7:
        return {"beta": "N/A", "te": "N/A"}
    # [NEW LOGIC END]
    
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
    
    # YTD Return
    ytd_row = twr_df[twr_df["Horizon"] == "YTD"]
    ytd_ret = ytd_row["Return"].iloc[0] if not ytd_row.empty else 0.0
    
    # Calculate Efficiency Scores (Sharpe/Sortino)
    twr_curve = _get_daily_twr_curve(data)
    eff = calculate_efficiency_metrics(twr_curve)
    
    # Calculate Max Drawdown
    _, max_dd, _ = compute_drawdown_series(twr_curve)
    max_dd = max_dd / 100.0 # Convert back to decimal for formatting
    
    # Calculate Position Count (excluding CASH)
    sec_table = data.get("sec_table_current", pd.DataFrame())
    if not sec_table.empty:
        # Filter out CASH and ensure we only count real positions
        pos_df = sec_table[sec_table["ticker"] != "CASH"]
        position_count = len(pos_df)
    else:
        position_count = 0
        
    # Check if SI is annualized using same logic as engine
    # (Engine uses calculated inception_date and effective_as_of)
    si_is_ann = is_annualized(data["inception_date"], data.get("effective_as_of"))
    
    # ---- Alpha vs SPY (Since Inception) ----
    # Portfolio TWR (SI) minus SPY total return over the same window
    alpha_vs_spy = np.nan
    try:
        bench_prices_adj = data.get("benchmark_prices_adj")
        if bench_prices_adj is not None and "SPY" in bench_prices_adj.columns:
            spy_ser = bench_prices_adj["SPY"].dropna()
            pv = data["pv"]
            inception_date = data["inception_date"]
            effective_as_of = data.get("effective_as_of")
            market_start = pv.index.min() if not pv.empty else None

            if market_start is not None and not spy_ser.empty:
                # Base price: last close strictly before portfolio start (captures Day 1)
                history_before = spy_ser[spy_ser.index < market_start]
                if not history_before.empty:
                    base_price = float(history_before.iloc[-1])
                else:
                    base_price = float(spy_ser.asof(market_start)) if not spy_ser[spy_ser.index <= market_start].empty else None

                if base_price is not None and base_price > 0:
                    end_anchor = effective_as_of if effective_as_of is not None else pv.index.max()
                    end_price = float(spy_ser.asof(end_anchor))
                    if not pd.isna(end_price) and end_price > 0:
                        spy_cum = end_price / base_price - 1.0
                        spy_ret = annualize_return(spy_cum, market_start, end_anchor)
                        alpha_vs_spy = twr_si - spy_ret
    except Exception as e:
        print(f"Alpha vs SPY calculation error: {e}")
        alpha_vs_spy = np.nan

    # ---- Cash Drag % ----
    # CASH market value as a percentage of total portfolio value
    cash_drag_pct = 0.0
    if not sec_table.empty and current_mv > 0:
        cash_rows = sec_table[sec_table["ticker"] == "CASH"] if "ticker" in sec_table.columns else pd.DataFrame()
        # sec_table_current excludes CASH; use the full holdings data
        holdings = data.get("holdings", pd.DataFrame())
        if not holdings.empty:
            cash_rows_h = holdings[holdings["ticker"] == "CASH"]
            if not cash_rows_h.empty:
                cash_mv = float(cash_rows_h["shares"].iloc[0])  # CASH shares == dollar value
                cash_drag_pct = cash_mv / current_mv

    return {
        "current_mv": current_mv,
        "twr_si": twr_si,
        "pl_si": data["pl_si"],
        "mtd_ret": mtd_ret,
        "ytd_ret": ytd_ret,
        "sharpe": eff["sharpe"],
        "sortino": eff["sortino"],
        "max_dd": max_dd,
        "position_count": position_count,
        "is_annualized": si_is_ann,
        "alpha_vs_spy": alpha_vs_spy,
        "cash_drag_pct": cash_drag_pct,
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
    
    horizons = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y"]
    
    snap_map = {row["Horizon"]: row["Return"] for _, row in twr_df.iterrows()}
    
    as_of = data.get("effective_as_of")
    if as_of is None:
        as_of = pv.index.max()
    
    # Dynamic Label for 1D - use raw user-selected date (not snapped) to match Holdings page
    report_end_date = data.get("selected_end_date") or as_of
    label_1d = get_display_label_for_1d(report_end_date)

    rows = []
    
    # Pre-calculate mv_end robustly
    mv_as_of = 0.0
    if as_of in pv.index:
        mv_as_of = float(pv.loc[as_of])
    else:
        # Fallback to nearest prior
        idx = pv.index.searchsorted(as_of)
        if idx > 0:
            mv_as_of = float(pv.iloc[idx - 1])
        else:
             # Should practically never happen if as_of is from engine
             mv_as_of = float(pv.iloc[-1]) if not pv.empty else 0.0

    for h in horizons:
        # Calculate PL components locally to expose meta data
        # Logic matches calculate_horizon_pl in portfolio_engine.py
        start = get_portfolio_horizon_start(pv, inception_date, h)

        # Check annualization for label
        is_ann_h = False
        if start is not None:
             is_ann_h = is_annualized(start, as_of)

        # Use dynamic label for 1D, otherwise standard
        display_h = label_1d if h == "1D" else h
        
        if is_ann_h:
             display_h += " (Ann.)"

        ret = snap_map.get(h, np.nan)
        
        pl_val = np.nan
        mv_start = 0.0
        mv_end = mv_as_of
        net_flows = 0.0
        sharpe = "N/A"
        sortino = "N/A"
        
        if start is not None and start < as_of:
            # Map start to pv index using BACKWARD SNAP (GIPS-compliant)
            # Must match get_portfolio_horizon_start behavior to ensure displayed
            # start date matches the actual calculation start date
            if start not in pv.index:
                pv_idx = pv.index.sort_values()
                # Backward snap: find last PV date <= start (not forward snap)
                prev_dates = pv_idx[pv_idx <= start]
                if len(prev_dates) > 0:
                    start = prev_dates.max()
                else:
                    # Fallback to forward snap if no prior date exists
                    pos = pv_idx.searchsorted(start)
                    if pos < len(pv_idx):
                        start = pv_idx[pos]
            
            if start in pv.index:
                mv_start = float(pv.loc[start])
                
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
                    
                    # Capture components for Audit
                    sharpe_vol = eff.get("vol", 0.0)
                    sharpe_ret = eff.get("ret", 0.0)
                    sharpe_rf = eff.get("rf", 0.0)
                else:
                    sharpe_vol = 0.0
                    sharpe_ret = 0.0
                    sharpe_rf = 0.0
        else:
             sharpe_vol = 0.0
             sharpe_ret = 0.0
             sharpe_rf = 0.0

        rows.append({
            "Horizon": display_h,
            "Return": ret,
            "P/L": pl_val,
            "Sharpe": sharpe,
            "Sortino": sortino,
            
            # Sharpe Audit Meta
            "meta_Sharpe_vol": sharpe_vol,
            "meta_Sharpe_ret": sharpe_ret,
            "meta_Sharpe_rf": sharpe_rf,
            
            # Audit Meta Columns
            f"meta_Return_start": mv_start,
            f"meta_Return_end": mv_end,
            f"meta_Return_flow": net_flows,
            f"meta_Return_inc": 0.0, # Portfolio level income tricky to separate here
            f"meta_Return_denom": mv_start + net_flows, # Approximation for display
            f"meta_Return_is_annualized": is_annualized(start, as_of) if start is not None else False,
            f"meta_Return_days": (as_of - start).days if start is not None else 0,
            f"meta_Return_start_date": start,
            f"meta_Return_end_date": as_of,

            f"meta_P/L_start": mv_start,
            f"meta_P/L_end": mv_end,
            f"meta_P/L_start_date": start,
            f"meta_P/L_end_date": as_of,
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
    
    si_mv_end = mv_as_of
        
    si_ret = twr_si_ann if pd.notna(twr_si_ann) else twr_si
    
    # Check if SI is annualized
    si_label = "Since Inception"
    if is_annualized(si_start, as_of):
        si_label += " (Ann.)"
    
    # SI Risk Metrics
    eff_si = calculate_efficiency_metrics(twr_curve_full)
    
    rows.append({
        "Horizon": si_label,
        "Return": si_ret,
        "P/L": pl_si,
        "Sharpe": eff_si["sharpe"],
        "Sortino": eff_si["sortino"],
        
        # Sharpe Audit Meta
        "meta_Sharpe_vol": eff_si.get("vol", 0.0),
        "meta_Sharpe_ret": eff_si.get("ret", 0.0),
        "meta_Sharpe_rf": eff_si.get("rf", 0.0),
        
        f"meta_Return_start": si_mv_start,
        f"meta_Return_end": si_mv_end,
        f"meta_Return_flow": si_flows,
        f"meta_Return_inc": 0.0,
        f"meta_Return_denom": si_mv_start + si_flows,
        f"meta_Return_is_annualized": is_annualized(si_start, as_of),
        f"meta_Return_days": (as_of - si_start).days,
        f"meta_Return_start_date": si_start,
        f"meta_Return_end_date": as_of,

        f"meta_P/L_start": si_mv_start,
        f"meta_P/L_end": si_mv_end,
        f"meta_P/L_start_date": si_start,
        f"meta_P/L_end_date": as_of,
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
    # Prefer precomputed cache (if available)
    if data is not None:
        cache = data.get("ticker_pl_cache")
        if cache and horizon in cache:
            return cache[horizon]

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
    
    # GIPS FIX: Use effective_as_of from engine for consistency
    effective_as_of = data.get("effective_as_of")
    
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
            return_components=True,
            effective_as_of=effective_as_of
        )
        
        if isinstance(res, dict):
            item = {"ticker": t, "pl": res["pl"]}
            # Populate Meta Columns directly from calculation components
            item[f"meta_{horizon}_start"] = res["start"]
            item[f"meta_{horizon}_end"] = res["end"]
            item[f"meta_{horizon}_flow"] = res["flow"]
            item[f"meta_{horizon}_inc"] = res["inc"]
            item[f"meta_{horizon}_denom"] = res["denom"]
            item[f"meta_{horizon}_start_date"] = res.get("start_date")
            item[f"meta_{horizon}_end_date"] = res.get("end_date")
        else:
            item = {"ticker": t, "pl": res}
            
        results.append(item)
        
    return pd.DataFrame(results).set_index("ticker")

def get_asset_class_pl(data, asset_class, horizon, return_components=False):
    """
    Computes DIRECT asset class P/L using centralized engine logic.
    """
    # Prefer precomputed cache (if available)
    if data is not None:
        cache = data.get("asset_class_pl_cache", {})
        cached = cache.get(horizon, {}).get(asset_class)
        if cached is not None:
            if return_components:
                return cached
            if isinstance(cached, dict):
                return cached.get("pl")
            return cached

    return calculate_asset_class_pl(
        asset_class,
        horizon,
        data["prices"],
        data["pv"],
        data["inception_date"],
        data["tx_raw"],
        data["sec_table"],
        data["dividends"],
        return_components=return_components,
        effective_as_of=data.get("effective_as_of")
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
    
    # Mask diagonal (self-correlation is always 1, visually dominates)
    labels = corr.columns.tolist()
    corr_masked = corr.copy()
    np.fill_diagonal(corr_masked.values, np.nan)

    # Custom CYBORG-themed diverging colorscale: deep blue → dark gray (0) → neon accent
    cyborg_colorscale = [
        [0.0,  "#1a3a5c"],   # deep blue (strong negative)
        [0.25, "#2a5a8c"],   # mid blue
        [0.5,  "#2d2d2d"],   # dark gray (zero / neutral)
        [0.75, "#6a4f8a"],   # muted purple accent
        [1.0,  "#8064A2"],   # neon purple accent (strong positive)
    ]
    
    fig = px.imshow(
        corr_masked,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=cyborg_colorscale,
        zmin=-1, zmax=1
    )
    
    fig.update_traces(hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>")
    
    # Add gray "1.00" annotations on the diagonal (masked cells)
    for i, lbl in enumerate(labels):
        fig.add_annotation(
            x=lbl, y=lbl,
            text="1.00",
            showarrow=False,
            font=dict(size=11, color="rgba(180,180,180,0.6)"),
        )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=10, b=40),
        height=550,
        font=dict(size=11),
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
    df["meta_Return_start_date"] = prev_date
    df["meta_Return_end_date"] = target_date
    
    # Add Asset Class specific meta
    df["meta_ac_start"] = df["Asset Class"].map(lambda x: ac_details[x]["start"])
    df["meta_ac_end"] = df["Asset Class"].map(lambda x: ac_details[x]["end"])
    df["meta_ac_flow"] = df["Asset Class"].map(lambda x: ac_details[x]["flow"])
    df["meta_ac_inc"] = df["Asset Class"].map(lambda x: ac_details[x]["inc"])
    # Map dates to AC meta for consistency with Frongello outputs
    df["meta_ac_start_date"] = prev_date
    df["meta_ac_end_date"] = target_date
    
    # --- FIX: Calculate and Append Residual ---
    residual_pct, residual_pl = _calculate_residual_return(data, df, start_date=prev_date, end_date=target_date)
    
    if abs(residual_pl) > 0.01 or abs(residual_pct) > 0.01:
        df = pd.concat([df, pd.DataFrame([{
            "Asset Class": "Recon/Residual",
            "Effect": residual_pl,
            "Contribution (%)": residual_pct,
            "meta_denominator": denominator,
            "meta_Return_denom": denominator,
            "meta_Return_start": pv_start,
            "meta_Return_flow": ext_flow_today,
            "meta_Return_start_date": prev_date,
            "meta_Return_end_date": target_date
        }])], ignore_index=True)

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
    # Glow trace (wider, semi-transparent behind main line for neon effect)
    fig.add_trace(go.Scatter(
        x=twr_ret_pct.index,
        y=twr_ret_pct.values,
        mode='lines',
        line=dict(color=_hex_to_rgba(GLOBAL_PALETTE[0], 0.25), width=8, shape='spline'),
        hoverinfo='skip',
        showlegend=False,
    ))
    # Main trace with vertical gradient fill
    fig.add_trace(go.Scatter(
        x=twr_ret_pct.index,
        y=twr_ret_pct.values,
        mode='lines',
        fill='tozeroy',
        name='Portfolio Return (TWR)',
        line=dict(color=GLOBAL_PALETTE[0], width=2.5, shape='spline'),
        fillgradient=dict(
            type="vertical",
            colorscale=[
                [0.0, "rgba(0,0,0,0)"],
                [1.0, _hex_to_rgba(GLOBAL_PALETTE[0], 0.35)],
            ],
        ),
        customdata=pv_daily.values,
        hovertemplate="<b>TWR</b>: %{y:.2f}%<br><b>Value</b>: %{customdata:$,.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        
        yaxis_title="Return (%)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified",
        height=428,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
    # Glow trace (wider, semi-transparent for neon effect)
    fig.add_trace(go.Scatter(
        x=twr_plot.index,
        y=twr_plot.values,
        mode='lines',
        line=dict(color=_hex_to_rgba(GLOBAL_PALETTE[0], 0.2), width=10, shape='spline'),
        hoverinfo='skip',
        showlegend=False,
    ))
    # Main portfolio trace with vertical gradient fill
    fig.add_trace(go.Scatter(
        x=twr_plot.index,
        y=twr_plot.values,
        mode='lines',
        fill='tozeroy',
        name='Portfolio',
        line=dict(color=GLOBAL_PALETTE[0], width=3, shape='spline'),
        fillgradient=dict(
            type="vertical",
            colorscale=[
                [0.0, "rgba(0,0,0,0)"],
                [1.0, _hex_to_rgba(GLOBAL_PALETTE[0], 0.25)],
            ],
        ),
        hovertemplate="<b>Portfolio</b>: %{y:.2f}%<extra></extra>"
    ))
    
    # 2. Benchmarks (unchanged logic, just context)
    colors = [GLOBAL_PALETTE[2], GLOBAL_PALETTE[4], GLOBAL_PALETTE[6], GLOBAL_PALETTE[10]]
    bench_prices_adj = data.get("benchmark_prices_adj")
    if benchmark_tickers:
        for i, (name, ticker) in enumerate(benchmark_tickers.items()):
            try:
                if bench_prices_adj is not None and not bench_prices_adj.empty and ticker in bench_prices_adj.columns:
                    ser = bench_prices_adj[ticker]
                else:
                    hist = fetch_price_history([ticker], use_adj_close=True)
                    ser = hist[ticker]
                
                # [NEW LOGIC START]
                # Gateway Check: If benchmark history starts significantly after the chart start date,
                # do NOT plot it (it would be misleading or flat-lined).
                # Allow 7 day grace period for holidays/weekend mismatches.
                earliest_bm_date = ser.index[0]
                if (earliest_bm_date - start_date).days > 7:
                    continue
                # [NEW LOGIC END]
                
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
                
                # GIPS FIX: Backward Snap
                if base_price is None:
                     if not ser[ser.index <= start_date].empty:
                         base_price = float(ser.asof(start_date))

                # FIX: Align Benchmark to Portfolio Calendar (Weekends/Holidays)
                # Reindex entire available history to the Portfolio's specific daily/weekend index
                # This ensures consistent X-axis alignment and forward-fills Friday prices to weekends.
                # Use twr_window.index which is the exact plotting X-axis for the portfolio.
                ser_aligned = ser.reindex(twr_window.index, method='ffill')

                # If we found a prior close (base_price), use it as base. 
                # Otherwise attempt to use the first valid price in the aligned series as the 0% anchor.
                if base_price is None:
                     valid_aligned = ser_aligned.dropna()
                     if not valid_aligned.empty:
                         base_price = float(valid_aligned.iloc[0])
                     else:
                         continue # Cannot normalize without a base

                ser_norm = (ser_aligned / base_price - 1.0) * 100.0
                
                fig.add_trace(go.Scatter(
                    x=ser_norm.index,
                    y=ser_norm.values,
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=1.5, shape='spline'),
                    hovertemplate=f"<b>{name}</b>: %{{y:.2f}}%<extra></extra>"
                ))
            except:
                pass
                
    fig.update_layout(
        
        yaxis_title="Return (%)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450
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
    if total_val > 0:
        merged["actual_pct"] = merged["value"] / total_val * 100
    else:
        merged["actual_pct"] = 0.0
    merged["delta"] = merged["actual_pct"] - merged["target_pct"]
    
    # Generate Custom Text Labels (Hide < 5%)
    display_text = []
    merged_actual = merged[merged["value"] > 0].copy()
    for _, row in merged_actual.iterrows():
        if row["actual_pct"] < 5.0:
            display_text.append("")
        else:
            display_text.append(f"{row['short_name']}<br>{row['actual_pct']:.1f}%")
            
    # Pie Chart (Donut)
    pie_fig = go.Figure(go.Pie(
        labels=merged_actual["short_name"],
        values=merged_actual["value"],
        text=display_text,
        hole=0.4,
        textinfo='text',
        marker=dict(colors=GLOBAL_PALETTE),
        sort=False,
        direction='clockwise',
        rotation=-90,
        textfont=dict(color='white'),
        hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Share: %{percent:.2%}<extra></extra>"
    ))
    
    # Center annotation showing total portfolio value
    pie_fig.add_annotation(
        text=f"<b>${total_val:,.0f}</b>",
        x=0.5, y=0.5,
        font=dict(size=16, color='white'),
        showarrow=False,
        xref='paper', yref='paper',
    )
    
    pie_fig.update_layout(
        
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
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
    # Include classes with current value OR target weight
    merged_bar = merged[(merged["value"] > 0) | (merged["target_pct"] > 0)].copy()
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=merged_bar["short_name"],
        y=merged_bar["actual_pct"],
        name="Actual %",
        marker_color=GLOBAL_PALETTE[0],
        marker_line=dict(
            color=_hex_to_rgba(GLOBAL_PALETTE[0], 0.5),
            width=1,
        ),
        customdata=merged_bar["value"],
        hovertemplate="<b>Actual</b>: %{y:.2f}%<br>Value: %{customdata:$,.2f}<extra></extra>"
    ))
    bar_fig.add_trace(go.Bar(
        x=merged_bar["short_name"],
        y=merged_bar["target_pct"],
        name="Target %",
        marker_color=GLOBAL_PALETTE[1],
        marker_line=dict(
            color=_hex_to_rgba(GLOBAL_PALETTE[1], 0.5),
            width=1,
        ),
        hovertemplate="<b>Target</b>: %{y:.2f}%<extra></extra>"
    ))
    
    # Delta overlay: Drift (Actual − Target) as thin bars, green=overweight, red=underweight
    drift_vals = merged_bar["delta"].values
    drift_colors = ['#22c55e' if d >= 0 else '#ef4444' for d in drift_vals]
    bar_fig.add_trace(go.Bar(
        x=merged_bar["short_name"],
        y=drift_vals,
        name="Drift",
        marker_color=drift_colors,
        marker_line=dict(
            color=[_hex_to_rgba(c, 0.7) for c in drift_colors],
            width=1,
        ),
        width=0.15,
        customdata=np.column_stack([merged_bar["actual_pct"].values, merged_bar["target_pct"].values]),
        hovertemplate="<b>Drift</b>: %{y:.2f}%<br>Actual: %{customdata[0]:.2f}%<br>Target: %{customdata[1]:.2f}%<extra></extra>"
    ))
    
    bar_fig.update_layout(
        
        barmode='group',
        yaxis_title="Percentage (%)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        height=450
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
            template="plotly_dark",
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
    
    # Pull out the largest slice slightly for emphasis
    pull_vals = [0.0] * len(filtered)
    if len(pull_vals) > 0:
        pull_vals[0] = 0.05  # First row is largest (sorted desc)
    
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
        pull=pull_vals,
        textfont=dict(color='white'),
        hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Share: %{percent:.2%}<extra></extra>"
    ))
    
    # Larger center annotation with AC name and total dollar value
    fig.add_annotation(
        text=f"<b>{full_name}</b><br>${total_val:,.0f}",
        x=0.5, y=0.5,
        font=dict(size=16, color='white'),
        showarrow=False,
        xref='paper', yref='paper',
    )
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
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
    
    # Gradient bars: higher exposure = more saturated accent, lower = muted
    base_color = GLOBAL_PALETTE[0]  # "#4C6A92" steel blue
    max_exp = sector_df["Exposure"].max() if not sector_df["Exposure"].empty else 1.0
    if max_exp == 0:
        max_exp = 1.0
    gradient_colors = [
        _hex_to_rgba(base_color, 0.4 + 0.6 * (exp / max_exp)) for exp in sector_df["Exposure"]
    ]
    
    # Force text positioning: inside for bars > 3%, outside for small ones
    text_positions = ['inside' if exp > 3.0 else 'outside' for exp in sector_df["Exposure"]]
    
    fig = go.Figure(go.Bar(
        y=sector_df["Sector"],
        x=sector_df["Exposure"],
        orientation='h',
        marker_color=gradient_colors,
        marker_line=dict(
            color=_hex_to_rgba(base_color, 0.5),
            width=1,
        ),
        text=sector_df["Exposure"].apply(lambda x: f"{x:.2f}%"),
        textposition=text_positions,
        hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(
        
        xaxis_title="Exposure (%)",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            gridwidth=1,
        ),
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
            line=dict(
                color=GLOBAL_PALETTE[i % len(GLOBAL_PALETTE)],
                width=0.5,
                shape='spline',
                smoothing=1.2
            ),
            hovertemplate=f"<b>{col}</b>: %{{y:.2f}}%<extra></extra>"
        ))
    
    # Subtle top-line glow on the total (sum) boundary
    total_pct = pct.sum(axis=1)
    fig.add_trace(go.Scatter(
        x=total_pct.index,
        y=total_pct.values,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.1)', width=6, shape='spline'),
        hoverinfo='skip',
        showlegend=False,
    ))
        
    fig.update_layout(
        
        yaxis_title="Allocation (%)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified",
        height=450,
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
    
    Now uses GIPS-compliant Backward Snap logic for start_date to ensure
    consistency with 'MTD' and '1M' portfolio calculations.
    """
    parts = year_month_str.split('-')
    year = int(parts[0])
    month = int(parts[1])
    
    # Target: Last calendar day of the PREVIOUS month
    calendar_start = pd.Timestamp(year, month, 1) - pd.Timedelta(days=1)
    end_date = calendar_start + pd.offsets.MonthEnd(1)
    
    pv = data["pv"]
    if pv.empty: return pd.DataFrame()

    # Backward Snap: Find last valid PV date <= calendar_start
    # This aligns the anchor with the MTD/1M calculations in financial_math.py
    pv_idx = pv.index
    prev_dates = pv_idx[pv_idx <= calendar_start]
    
    if len(prev_dates) > 0:
        start_date = prev_dates.max()
    else:
        # Fallback if history starts mid-month or this is the first month
        # Capture inception by going back one day before the first data point
        start_date = pv_idx.min() - pd.Timedelta(days=1)

    # Use the robust Frongello engine
    df = _calculate_frongello_linking(data, start_date=start_date, end_date=end_date)
    
    if df.empty: return pd.DataFrame()
    
    # --- FIX: Calculate and Append Residual ---
    residual_pct, residual_pl = _calculate_residual_return(data, df, start_date, end_date)
    
    if abs(residual_pl) > 0.01 or abs(residual_pct) > 0.01:
        df = pd.concat([df, pd.DataFrame([{
            "Asset Class": "Recon/Residual",
            "Effect": residual_pl,
            "Contribution (%)": residual_pct,
            "meta_frongello_sum_factors": 0,
            "meta_frongello_avg_denom": 0
        }])], ignore_index=True)

    return df.sort_values("Contribution (%)", ascending=False)


def get_weekly_attribution_breakdown(data, date_str):
    """
    Decomposes a specific week's Market Effect into Asset Class components
    using Frongello Linking. 
    
    Uses GIPS-compliant backward snap logic (same as Portfolio TWR '1W')
    to ensure returns capture holidays and non-trading days correctly.
    """
    end_date = pd.Timestamp(date_str)
    pv = data["pv"]
    
    if pv.empty: return pd.DataFrame()
    
    # Calculate Start Date matching Portfolio TWR "1W" logic:
    # UPDATED: Must align with Chart binning logic (W-FRI) to ensure P/L matches.
    # Chart bins are [Prev Friday, Current Friday].
    # If end_date is Jan 8 (Thu), Chart bin ends Jan 9 (Fri) and starts Jan 2 (Fri).
    # We must set start_date = Previous Bin Bound (Jan 2).
    
    # 1. Find the Friday of the current week (Bin End)
    # weekday(): Mon=0 ... Fri=4 ... Sun=6
    days_to_fri = (4 - end_date.weekday()) % 7
    bin_end_fri = end_date + pd.Timedelta(days=days_to_fri)
    
    # 2. Start Date is 7 days prior to the Bin End
    start_date = bin_end_fri - pd.Timedelta(days=7)
    
    # For attribution window (start, end], we want returns AFTER start_date.
    # So if start_date is Jan 2, we capture Jan 3..Jan 9 (or Jan 8).
    # This precisely matches the chart's exclusion of the Jan 2 return.
    
    # Safety: If history is short, capture inception
    if start_date < pv.index.min():
        start_date = pv.index.min() - pd.Timedelta(days=1)
        
    # Safety check: if target window is entirely in future (unlikely)
    if start_date >= end_date:
        return pd.DataFrame()

    # Use the robust Frongello engine
    df = _calculate_frongello_linking(data, start_date=start_date, end_date=end_date)
    
    if df.empty: return pd.DataFrame()

    # --- FIX: Calculate and Append Residual ---
    residual_pct, residual_pl = _calculate_residual_return(data, df, start_date, end_date)
    
    # Only append if significant to avoid noise
    if abs(residual_pl) > 0.01 or abs(residual_pct) > 0.01:
        df = pd.concat([df, pd.DataFrame([{
            "Asset Class": "Recon/Residual",
            "Effect": residual_pl,
            "Contribution (%)": residual_pct,
            "meta_frongello_sum_factors": 0,
            "meta_frongello_avg_denom": 0
        }])], ignore_index=True)
    
    return df.sort_values("Contribution (%)", ascending=False)


def get_smart_attribution_chart(data, start_date=None, end_date=None, theme="light"):
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
    
    # 3. Filter by Date Range (if provided)
    if start_date:
        start_ts = pd.Timestamp(start_date)
        # Ensure we don't look before data starts
        if start_ts < pv_daily.index.min():
            start_ts = pv_daily.index.min()
        mkt = mkt[mkt.index >= start_ts]
        ext = ext[ext.index >= start_ts]
        
    if end_date:
        end_ts = pd.Timestamp(end_date)
        mkt = mkt[mkt.index <= end_ts]
        ext = ext[ext.index <= end_ts]
    
    if mkt.empty:
        return go.Figure()

    # Decide on aggregation
    # Weekly until portfolio age reaches 6 months, then monthly
    history_days = (mkt.index.max() - mkt.index.min()).days
    six_months_days = 183
    
    if history_days < six_months_days:
        # Weekly until 6 months of history
        freq = 'W-FRI'
        p_label = 'weekly'
        fmt = '%Y-%m-%d'
    else:
        # Monthly once portfolio is 6 months+ old
        freq = 'ME'
        p_label = 'monthly'
        fmt = '%Y-%m'

    # Apply Resampling
    mkt_agg = mkt.resample(freq).sum()
    ext_agg = ext.resample(freq).sum()
    
    # FIX: Clamp last date to actual end date (prevent future labels)
    true_end = pv.index.max()
    if not mkt_agg.empty and mkt_agg.index[-1] > true_end:
        new_idx = mkt_agg.index.tolist()
        new_idx[-1] = true_end
        mkt_agg.index = pd.Index(new_idx)
        ext_agg.index = pd.Index(new_idx)

    # GIPS COMPLIANCE: Include all periods including inception for correct Cumulative ΔPV reconciliation
    # (Previously dropped the first period to hide initial deposit, but this distorted the total P/L)

    # Set vars for plotting
    period = 'M' if freq == 'ME' else 'D'
    # Use categorical labels to avoid excessive spacing when few periods exist
    x_labels_fmt = mkt_agg.index.strftime(fmt)
    x_values = x_labels_fmt
    custom_data = [{'period': p_label, 'date': label} for label in x_labels_fmt]


    fig = go.Figure()

    # External Flows Bar
    fig.add_trace(go.Bar(
        x=x_values,
        y=ext_agg,
        name="External Flows",
        marker_color=GLOBAL_PALETTE[1],
        marker_line_width=0,
        customdata=[d['date'] for d in custom_data]
    ))

    # Market Effect Bar — conditional green/red coloring by sign
    mkt_bar_colors = ['#22c55e' if v >= 0 else '#ef4444' for v in mkt_agg.values]
    fig.add_trace(go.Bar(
        x=x_values,
        y=mkt_agg,
        name="Market Effect",
        marker_color=mkt_bar_colors,
        marker_line_width=0,
        customdata=[d['date'] for d in custom_data]
    ))
    
    # Add customdata to traces for drilldown
    fig.update_traces(selector=dict(type='bar'), customdata=[d['date'] for d in custom_data], hovertemplate='%{y:$,.2f}<extra>%{customdata}</extra>')


    # Cumulative Line: Sum of Market Effects only (Represents Since Inception Gain)
    cum = mkt_agg.cumsum()
    
    # Glow trace behind cumulative line (wider semi-transparent)
    fig.add_trace(go.Scatter(
        x=x_values,
        y=cum.values,
        mode='lines',
        line=dict(color=_hex_to_rgba(GLOBAL_PALETTE[8], 0.3), width=8),
        yaxis='y2',
        hoverinfo='skip',
        showlegend=False,
    ))
    # Main cumulative line
    fig.add_trace(go.Scatter(
        x=x_values,
        y=cum.values,
        mode='lines+markers',
        name="Cumulative ΔPV",
        line=dict(color=GLOBAL_PALETTE[8], width=2, dash='dot'),
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

    # Determine intelligent x-axis config based on history duration
    # REVISED: Tighter labeling logic to ensure bars are labeled
    if history_days > 365 * 5:       # > 5 Years: Yearly ticks
        xaxis_dtick = "M12"
        xaxis_format = "%Y"
    elif history_days > 365 * 2:     # 2-5 Years: Quarterly ticks
        xaxis_dtick = "M3"
        xaxis_format = "%b '%y"
    elif history_days > 365:         # 1-2 Years: Bi-Monthly
        xaxis_dtick = "M2"
        xaxis_format = "%b '%y"
    elif freq == 'ME':               # Monthly mode (3-12 months): Every month
        xaxis_dtick = "M1"
        xaxis_format = "%b '%y"
    elif freq == 'W-FRI':            # Weekly mode (1-3 months): Every week (approx)
        # 1 week in milliseconds
        xaxis_dtick = 604800000 
        xaxis_format = "%Y-%m-%d"
    else:                            # Daily mode (< 1 month): Daily
        xaxis_dtick = 86400000       # 1 day in milliseconds
        xaxis_format = "%Y-%m-%d"

    fig.update_layout(
        yaxis_title="Change ($)",
        yaxis=dict(range=[y_min, y_max]), # Focus scale on Performance Gain/Loss
        yaxis2=dict(title="Cumulative Gain ($)", overlaying='y', side='right'),
        barmode='relative',
        template="plotly_dark",
        margin=dict(l=40, r=40, t=80, b=40), # Increased top margin (t) to 80
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Period",
        xaxis=dict(
            type='category',
            tickangle=-45
        ),
        bargap=0.15 if freq == 'ME' else 0.3
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
    
    # Only include asset classes with current value > 0
    sec_table = data.get("sec_table_current", pd.DataFrame())
    if sec_table.empty:
        return go.Figure()
    ac_values = sec_table.groupby("asset_class")["market_value"].sum()
    included_classes = set(ac_values[ac_values > 0].index)
    total_value = ac_values[ac_values > 0].sum()
    
    plot_data = []
    for cls, metrics in risk_return.items():
        if cls not in included_classes:
            continue
        weight = (ac_values.get(cls, 0) / total_value * 100) if total_value > 0 else 1
        plot_data.append({
            "Asset Class": cls,
            "Return": metrics["return"],
            "Volatility": metrics["vol"],
            "Weight": max(weight, 2),  # Min 2% so tiny allocations are still visible
        })
    df = pd.DataFrame(plot_data)
    
    if df.empty: return go.Figure()

    # Compute portfolio-weighted averages for quadrant lines
    avg_vol = (df["Volatility"] * df["Weight"]).sum() / df["Weight"].sum()
    avg_ret = (df["Return"] * df["Weight"]).sum() / df["Weight"].sum()

    fig = px.scatter(
        df, x="Volatility", y="Return", hover_name="Asset Class",
        size="Weight", size_max=30,
        color="Asset Class",
        color_discrete_sequence=GLOBAL_PALETTE
    )
    
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<br>Weight: %{marker.size:.1f}%<extra></extra>",
        marker_line_width=2,
        marker_line_color='white',
    )

    # Quadrant reference lines
    fig.add_hline(y=avg_ret, line_dash="dot", line_color="rgba(255,255,255,0.25)", line_width=1,
                  annotation_text="Avg Return", annotation_position="bottom right",
                  annotation_font_color="rgba(255,255,255,0.5)", annotation_font_size=10)
    fig.add_vline(x=avg_vol, line_dash="dot", line_color="rgba(255,255,255,0.25)", line_width=1,
                  annotation_text="Avg Vol", annotation_position="top left",
                  annotation_font_color="rgba(255,255,255,0.5)", annotation_font_size=10)

    # Direct annotation labels on each bubble with collision avoidance
    # Sort by volatility so we process left-to-right
    df_sorted = df.sort_values(["Volatility", "Return"]).reset_index(drop=True)
    placed = []  # list of (x, y) in data coords for placed labels
    
    # Get axis ranges for converting pixel offsets to data coords
    vol_range = df["Volatility"].max() - df["Volatility"].min()
    ret_range = df["Return"].max() - df["Return"].min()
    # Approximate label footprint in data units (rough px-to-data conversion)
    label_h = ret_range * 0.08 if ret_range > 0 else 3
    label_w = vol_range * 0.12 if vol_range > 0 else 3
    
    for _, row in df_sorted.iterrows():
        base_x = row["Volatility"]
        base_y = row["Return"]
        bubble_offset = max(row["Weight"] * 0.25, 8) + 8  # pixels
        
        # Candidate positions: above, below, right, left, upper-right, lower-right
        candidates = [
            (0, bubble_offset),          # above
            (0, -(bubble_offset + 5)),   # below
            (bubble_offset * 1.5, 0),    # right
            (-(bubble_offset * 1.5), 0), # left  
            (bubble_offset, bubble_offset * 0.7),        # upper-right
            (bubble_offset, -(bubble_offset * 0.7)),     # lower-right
        ]
        
        best_shift = candidates[0]  # default: above
        best_min_dist = -1
        
        for dx_px, dy_px in candidates:
            # Convert pixel shifts to approximate data coordinates for overlap check
            approx_x = base_x + (dx_px / 400) * vol_range if vol_range > 0 else base_x
            approx_y = base_y + (dy_px / 400) * ret_range if ret_range > 0 else base_y
            min_dist = float('inf')
            for px, py in placed:
                dist = ((approx_x - px) / max(label_w, 1)) ** 2 + ((approx_y - py) / max(label_h, 1)) ** 2
                min_dist = min(min_dist, dist)
            if not placed or min_dist > best_min_dist:
                best_min_dist = min_dist
                best_shift = (dx_px, dy_px)
                best_approx = (approx_x, approx_y)
        
        placed.append(best_approx if placed or best_min_dist >= 0 else (base_x, base_y + (bubble_offset / 400) * ret_range))
        
        fig.add_annotation(
            x=base_x, y=base_y,
            text=row["Asset Class"],
            showarrow=True if best_shift != candidates[0] else False,
            arrowhead=0,
            arrowwidth=0.8,
            arrowcolor="rgba(255,255,255,0.25)",
            xshift=best_shift[0],
            yshift=best_shift[1],
            font=dict(size=10, color="rgba(255,255,255,0.75)"),
        )
    
    fig.update_layout(
        
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        template="plotly_dark",
        showlegend=True,
        height=550,
        margin=dict(l=40, r=40, t=40, b=80), # Standardized margin
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18, # Brought closer to x-axis title
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
    
    # Glow trace (wider, semi-transparent for neon effect)
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values,
        mode='lines',
        line=dict(color=_hex_to_rgba(GLOBAL_PALETTE[2], 0.3), width=6, shape='spline'),
        hoverinfo='skip',
        showlegend=False,
    ))
    # Main drawdown trace with vertical gradient fill
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values,
        mode='lines',
        fill='tozeroy',
        name='Drawdown',
        line=dict(color=GLOBAL_PALETTE[2], width=1.5, shape='spline'),
        fillgradient=dict(
            type="vertical",
            colorscale=[
                [0.0, _hex_to_rgba(GLOBAL_PALETTE[2], 0.4)],
                [1.0, "rgba(0,0,0,0)"],
            ],
        ),
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
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        height=450,
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
        
        
    # Pre-compute all lines so we can pair lump-sum and contribution for shaded bands
    lump_lines = []   # list of (rate, values)
    contrib_lines = []  # list of (rate, values)

    for i, r in enumerate(rates):
        r_dec = r / 100.0
        lump_vals = [fv_lump(initial_value, r_dec, yr) for yr in years]
        contrib_vals = [fv_lump(initial_value, r_dec, yr) + fv_contrib(monthly_contrib, r_dec, yr) for yr in years]
        lump_lines.append((r, lump_vals))
        contrib_lines.append((r, contrib_vals))

    # Plot paired bands: lump-sum line (bottom) + contribution line (top) with shaded fill between
    for i, r in enumerate(rates):
        _, lump_vals = lump_lines[i]
        _, contrib_vals = contrib_lines[i]

        # Lump Sum Line (solid, bottom of band)
        fig.add_trace(go.Scatter(
            x=years, y=lump_vals,
            mode='lines',
            name=f"{r:.1f}% Lump Sum",
            line=dict(color=colors[i], width=2),
            hovertemplate=f"<b>{r:.1f}% Lump Sum</b>: %{{y:$,.2f}}<extra></extra>"
        ))

        # Contribution Line (dot dash, top of band) — fills down to lump-sum
        fig.add_trace(go.Scatter(
            x=years, y=contrib_vals,
            mode='lines',
            name=f"{r:.1f}% + ${monthly_contrib:,.0f}/mo",
            line=dict(color=colors[i+3], width=2.5, dash='dot'),
            fill='tonexty',
            fillcolor=_hex_to_rgba(colors[i], 0.12),
            hovertemplate=f"<b>{r:.1f}% + ${monthly_contrib:,.0f}/mo</b>: %{{y:$,.2f}}<extra></extra>"
        ))

    # "Current Value" horizontal reference line
    fig.add_hline(
        y=initial_value,
        line_dash="dash",
        line_color="rgba(255,255,255,0.35)",
        line_width=1,
        annotation_text="Today",
        annotation_position="top left",
        annotation_font_color="rgba(255,255,255,0.6)",
        annotation_font_size=11,
    )
        
    fig.update_layout(
        
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        hovermode="x unified",
        height=450
    )
    return fig

def get_flows_chart(data, theme="light", start_date=None, end_date=None):
    """Generates Internal Flows by Asset Class chart.
    
    Args:
        data: Data cache dict
        theme: Theme for styling (default "light")
        start_date: Optional start date to filter flows (default None = all history)
        end_date: Optional end date to filter flows (default None = today)
    """
    tx_raw = data["tx_raw"]
    holdings = data["holdings"]
    
    if tx_raw.empty: return go.Figure()
    
    # Filter by date range if provided (for period-specific analysis)
    if start_date is not None:
        end_dt = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
        start_dt = pd.Timestamp(start_date)
        mask = (tx_raw["date"] >= start_dt) & (tx_raw["date"] <= end_dt)
        tx_raw = tx_raw[mask].copy()
    
    ac_map = holdings.set_index("ticker")["asset_class"].to_dict()
    tx_raw["asset_class"] = tx_raw["ticker"].map(ac_map).fillna("Other")
    
    net_flows = tx_raw.groupby("asset_class")["amount"].sum()
    # Sort by absolute value so biggest movers appear first (top)
    net_flows = net_flows.reindex(net_flows.abs().sort_values().index)
    
    fig = go.Figure(go.Bar(
        y=net_flows.index,
        x=net_flows.values,
        orientation='h',
        marker_color=np.where(net_flows > 0, '#22c55e', '#ef4444'),
        marker_line=dict(width=1, color='rgba(255,255,255,0.15)'),
        text=net_flows.apply(fmt_dollar_clean),
        textposition='auto',
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b>: %{x:$,.2f}<extra></extra>"
    ))
    
    # Prominent vertical zero-line
    fig.add_vline(
        x=0, line_width=2, line_color='rgba(255,255,255,0.5)',
        line_dash='solid'
    )
    
    fig.update_layout(
        xaxis_title="Net Flow ($)",
        template="plotly_dark",
        height=450
    )
    return fig

def get_excess_return_chart(data, benchmark_tickers, theme="light"):
    """Generates Excess Return Bar Chart."""
    twr_df = data["twr_df"]
    pv = data["pv"]
    if twr_df.empty: return go.Figure()
    
    horizons = ["1D", "1W", "MTD", "YTD", "1M", "3M", "6M", "1Y", "SI"]

    # Dynamic Label Logic
    as_of = pv.index.max()
    label_1d = get_display_label_for_1d(as_of)
    display_horizons = [label_1d if h == "1D" else h for h in horizons]
    
    port_rets = twr_df.set_index("Horizon")["Return"]
    
    # Add SI if missing
    if "SI" not in port_rets:
        port_rets["SI"] = data["twr_si_ann"] if pd.notna(data["twr_si_ann"]) else data["twr_si"]
        
    fig = go.Figure()
    
    bench_prices_adj = data.get("benchmark_prices_adj")
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
                    if bench_prices_adj is not None and not bench_prices_adj.empty and bm_ticker in bench_prices_adj.columns:
                        ser = bench_prices_adj[bm_ticker]
                    else:
                        hist = fetch_price_history([bm_ticker], use_adj_close=True)
                        ser = hist[bm_ticker]
                    
                    # [NEW LOGIC START]
                    # Gateway Check: If benchmark history starts significantly after the horizon start,
                    # treat as N/A to prevent calculating return over a truncated period.
                    if ser.empty or (ser.index[0] - start).days > 7:
                        raise ValueError("Insufficient History")
                    # [NEW LOGIC END]
                    
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

                    # GIPS FIX: Backward Snap (asof) ensures we capture the correct base price 
                    # even if the start date is a holiday/weekend.
                    if base_price is None:
                         # Verify if any price exists <= start
                         if not ser[ser.index <= start].empty:
                             base_price = float(ser.asof(start))
                    
                    # Filter for window
                    ser_window = ser[ser.index >= start]
                    # Clip to end date
                    ser_window = ser_window[ser_window.index <= pv.index.max()]
                    
                    if not ser_window.empty:
                        if base_price is None:
                            base_price = float(ser_window.iloc[0])
                            
                        # Use Effective Anchor (Friday) for Benchmark Comparison to minimize "Sunday Drag"
                        effective_end = get_effective_anchor_date(pv.index.max())

                        # Robust End Price (Snapback)
                        end_price = float(ser.asof(effective_end))
                        if pd.isna(end_price):
                             end_price = float(ser_window.iloc[-1])
                             
                        b_cum = end_price / base_price - 1.0
                        
                        # Apply Universal Gate to Benchmark Return
                        # Use effective_end as date to match Portfolio TWR logic
                        b_ret = annualize_return(b_cum, start, effective_end)
                    
                    diff = (p_val - b_ret) * 100
                except:
                    pass
            
            excess_vals.append(diff)
            tooltip_data.append([p_val * 100, b_ret * 100, diff])
                
        # Conditional coloring: green/red for single benchmark, palette colors for multi-benchmark
        if len(benchmark_tickers) == 1:
            bar_colors = ['#22c55e' if v >= 0 else '#ef4444' for v in excess_vals]
        else:
            base = GLOBAL_PALETTE[i % len(GLOBAL_PALETTE)]
            bar_colors = [base] * len(excess_vals)
        
        fig.add_trace(go.Bar(
            x=display_horizons,
            y=excess_vals,
            name=bm_name,
            marker_color=bar_colors,
            customdata=tooltip_data,
            hovertemplate=(
                f"<b>{bm_name}</b><br>"
                "Portfolio: %{customdata[0]:.2f}%<br>"
                "Benchmark: %{customdata[1]:.2f}%<br>"
                "Excess: %{customdata[2]:.2f}%<extra></extra>"
            )
        ))
        
    # Zero-line — subtle dashed hline at y=0 for visual anchor
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        line_width=1,
    )
    
    fig.update_layout(
        
        yaxis_title="Excess Return (%)",
        barmode='group',
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
    
    # Pie Chart (Donut) with center annotation
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
        textfont=dict(color='white'),
        hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Share: %{percent:.2%}<extra></extra>"
    ))
    
    # Center annotation showing total portfolio value
    pie_fig.add_annotation(
        text=f"<b>${total_val_pie:,.0f}</b>",
        x=0.5, y=0.5,
        font=dict(size=16, color='white'),
        showarrow=False,
        xref='paper', yref='paper',
    )
    
    pie_fig.update_layout(
        
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
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
    ticker_merge["delta"] = ticker_merge["actual_pct"] - ticker_merge["target_pct"]
    
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=ticker_merge["ticker"],
        y=ticker_merge["actual_pct"],
        name="Actual %",
        marker_color=GLOBAL_PALETTE[0],
        marker_line=dict(
            color=_hex_to_rgba(GLOBAL_PALETTE[0], 0.5),
            width=1,
        ),
        customdata=ticker_merge["market_value"],
        hovertemplate="<b>Actual</b>: %{y:.2f}%<br>Value: %{customdata:$,.2f}<extra></extra>"
    ))
    bar_fig.add_trace(go.Bar(
        x=ticker_merge["ticker"],
        y=ticker_merge["target_pct"],
        name="Target %",
        marker_color=GLOBAL_PALETTE[1],
        marker_line=dict(
            color=_hex_to_rgba(GLOBAL_PALETTE[1], 0.5),
            width=1,
        ),
        hovertemplate="<b>Target</b>: %{y:.2f}%<extra></extra>"
    ))
    
    # Delta overlay: Drift (Actual − Target) as thin bars, green=overweight, red=underweight
    drift_vals = ticker_merge["delta"].values
    drift_colors = ['#22c55e' if d >= 0 else '#ef4444' for d in drift_vals]
    bar_fig.add_trace(go.Bar(
        x=ticker_merge["ticker"],
        y=drift_vals,
        name="Drift",
        marker_color=drift_colors,
        marker_line=dict(
            color=[_hex_to_rgba(c, 0.7) for c in drift_colors],
            width=1,
        ),
        width=0.15,
        customdata=np.column_stack([ticker_merge["actual_pct"].values, ticker_merge["target_pct"].values]),
        hovertemplate="<b>Drift</b>: %{y:.2f}%<br>Actual: %{customdata[0]:.2f}%<br>Target: %{customdata[1]:.2f}%<extra></extra>"
    ))
    
    bar_fig.update_layout(
        
        barmode='group',
        yaxis_title="Percentage (%)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=40),
        height=450
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
    
    # Filter for non-cash tickers for performance highlights
    perf_universe = sec_table[sec_table["ticker"] != "CASH"]
    
    # 1M
    if "1M" in perf_universe.columns:
        valid = perf_universe.dropna(subset=["1M"])
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
    as_of = data["pv"].index.max() if not data["pv"].empty else None
    label_1d = get_display_label_for_1d(as_of)
    
    if "1D" in perf_universe.columns:
        valid = perf_universe.dropna(subset=["1D"])
        if not valid.empty:
            top = valid.loc[valid["1D"].idxmax()]
            bot = valid.loc[valid["1D"].idxmin()]
            
            rows.append({
                "Metric": f"Best {label_1d} Performer",
                "Value": f"{top['ticker']} ({top['1D']*100:.2f}%, {get_pl(top['ticker'], '1D')})"
            })
            rows.append({
                "Metric": f"Bottom {label_1d} Performer",
                "Value": f"{bot['ticker']} ({bot['1D']*100:.2f}%, {get_pl(bot['ticker'], '1D')})"
            })
        else:
             rows.append({"Metric": f"Best {label_1d} Performer", "Value": "N/A"})
             rows.append({"Metric": f"Bottom {label_1d} Performer", "Value": "N/A"})
             
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
        
        # Calculate First Invested Date for Audit Metadata
        if ac == "CASH":
             # Use first external flow date
             if cf_ext is not None and not cf_ext.empty:
                 first_invested_date = cf_ext["date"].min()
             else:
                 first_invested_date = start_date
        else:
             # Use first transaction date for this asset class
             # Note: 'tx' is already filtered to exclude CASH, so we use the original logic
             ac_tx = tx[tx["asset_class"] == ac]
             if not ac_tx.empty:
                 first_invested_date = ac_tx["date"].min()
             else:
                 first_invested_date = start_date

        # Build DataFrame directly from Series for speed
        df_ac = pd.DataFrame({
            "Date": date_range,
            "Asset Class": ac,
            "Cash Invested": inv,
            "Portfolio Value": val,
            "Growth": growth,
            "First Invested": first_invested_date
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
        "Growth": pv_total_recalc - invested_total,
        "First Invested": start_date
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
    
    # 1. Prepare Data for Stacking (Everything EXCEPT Total)
    # If filter_value == "Total" or "All", show all components stacked.
    
    if filter_value in ["Total", "All"]:
        stack_df = ts_df[ts_df["Asset Class"] != "Total"].copy()
        # Sort by Asset Class for consistent coloring order
        stack_df = stack_df.sort_values(["Asset Class", "Date"])
        
        # Get list of classes to plot
        plot_classes = stack_df["Asset Class"].unique()
        
        # Add Stacked Area Traces (Portfolio Value) with spline smoothing
        for i, ac in enumerate(plot_classes):
            ac_data = stack_df[stack_df["Asset Class"] == ac]
            color = GLOBAL_PALETTE[i % len(GLOBAL_PALETTE)]
            
            fig.add_trace(go.Scatter(
                x=ac_data["Date"],
                y=ac_data["Portfolio Value"],
                mode='lines',
                name=ac,
                stackgroup='one', # Enable Stacking
                line=dict(width=0.5, color=color, shape='spline'),
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
            # Glow trace behind the dashed invested line
            fig.add_trace(go.Scatter(
                x=total_data["Date"],
                y=total_data["Cash Invested"],
                mode='lines',
                line=dict(color=_hex_to_rgba("#FFA500", 0.2), width=10, shape='spline'),
                hoverinfo='skip',
                showlegend=False,
            ))
            # Main dashed invested line with vertical gradient fill
            fig.add_trace(go.Scatter(
                x=total_data["Date"],
                y=total_data["Cash Invested"],
                mode='lines',
                name="Total Net Invested", 
                line=dict(color="#FFA500", width=3, dash='dash', shape='spline'),
                fill='tozeroy',
                fillgradient=dict(
                    type="vertical",
                    colorscale=[
                        [0.0, "rgba(0,0,0,0)"],
                        [1.0, _hex_to_rgba("#FFA500", 0.12)],
                    ],
                ),
                hovertemplate=(
                    "<b>Total Net Invested</b>: %{y:$,.2f}<extra></extra>"
                )
            ))
            
    else:
        # Specific Asset Class View
        # Area chart for Value, Dashed line for Invested
        ac_data = ts_df[ts_df["Asset Class"] == filter_value].sort_values("Date")
        if not ac_data.empty:
            color = GLOBAL_PALETTE[0]
            
            # Area (Value) with spline smoothing
            fig.add_trace(go.Scatter(
                x=ac_data["Date"],
                y=ac_data["Portfolio Value"],
                mode='lines',
                name=f"{filter_value} Value",
                fill='tozeroy',
                line=dict(color=color, width=2, shape='spline'),
                hovertemplate="<b>Value</b>: %{y:$,.2f}<extra></extra>"
            ))
            
            # Glow trace behind invested line
            fig.add_trace(go.Scatter(
                x=ac_data["Date"],
                y=ac_data["Cash Invested"],
                mode='lines',
                line=dict(color=_hex_to_rgba("#FFA500", 0.2), width=8, shape='spline'),
                hoverinfo='skip',
                showlegend=False,
            ))
            
            # Line (Invested) with spline smoothing
            fig.add_trace(go.Scatter(
                x=ac_data["Date"],
                y=ac_data["Cash Invested"],
                mode='lines',
                name="Net Invested", 
                line=dict(color="#FFA500", width=2, dash='dash', shape='spline'),
                hovertemplate="<b>Invested</b>: %{y:$,.2f}<extra></extra>"
            ))

    fig.update_layout( 
        
        yaxis_title="Value ($)",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        height=450,
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
    latest_date_val = summary_df["Date"].max() if not summary_df.empty else pd.Timestamp.now()
    
    summary_df["meta_Growth %_start"] = 0.0
    summary_df["meta_Growth %_end"] = summary_df["Portfolio Value"]
    summary_df["meta_Growth %_flow"] = summary_df["Cash Invested"]
    summary_df["meta_Growth %_inc"] = 0.0
    summary_df["meta_Growth %_denom"] = summary_df["Cash Invested"]
    
    # Use the calculated First Invested date from the time series logic
    # instead of the global inception date.
    if "First Invested" in summary_df.columns:
        summary_df["meta_Growth %_start_date"] = summary_df["First Invested"]
    else:
        summary_df["meta_Growth %_start_date"] = data["inception_date"]
        
    summary_df["meta_Growth %_end_date"] = latest_date_val

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
    
    # GIPS FIX: Use effective anchor
    effective_as_of = data.get("effective_as_of")
    if effective_as_of is None:
        effective_as_of = get_effective_anchor_date(pv.index.max())
    
    cash_recon = {}
    
    for h in horizons:
        # 1. Get Portfolio Total P/L (External Flows)
        if h == "SI":
            port_pl = pl_si
        else:
            port_pl = calculate_horizon_pl(pv, inception_date, cf_ext, h, effective_as_of=effective_as_of)
        
        if port_pl is None:
            cash_recon[h] = None
            continue
        
        # 2. Sum Ticker P/Ls (Internal Flows)
        sum_ticker_pl = 0.0
        
        # Iterate all tickers INCLUDING CASH (use full table to match Portfolio PL)
        # We include CASH because its P/L (Interest) is displayed in the main table,
        # so the Residual should only capture unexplained discrepancies.
        all_tickers = sec_table["ticker"].unique()
        
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
                portfolio_inception=pv_start_date if h == "SI" else None,
                effective_as_of=effective_as_of
            )
            if val is not None:
                sum_ticker_pl += val
        
        # 3. Diff is Cash / Recon P/L
        diff = port_pl - sum_ticker_pl
        
        # Capture date range for audit (using Portfolio Horizon logic)
        h_start = None
        if h == "SI":
            h_start = inception_date
        else:
            h_start = get_portfolio_horizon_start(pv, inception_date, h)
            
        # Snap start logic (must match calculate_horizon_pl)
        if h_start is not None and h_start not in pv.index:
            pv_idx = pv.index.sort_values()
            pos = pv_idx.searchsorted(h_start)
            if pos > 0:
                h_start = pv_idx[pos - 1]
            else:
                h_start = pv_idx[0]
                
        cash_recon[h] = {
            "pl": diff,
            "start_date": h_start,
            "end_date": as_of_dt
        }
    
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
            class_df[["asset_class", "meta_SI_start", "meta_SI_end", "meta_SI_flow", "meta_SI_inc", "meta_SI_denom", "meta_SI_start_date", "meta_SI_end_date"]],
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
            "meta_Value ($)_start_date": row.get("meta_SI_start_date"),
            "meta_Value ($)_end_date": row.get("meta_SI_end_date"),
            "meta_Value ($)_start_date": row.get("meta_SI_start_date"),
            "meta_Value ($)_end_date": row.get("meta_SI_end_date"),
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
    
    # 1. Timeline Setup (TRADING DAYS ONLY)
    # Using pv.index ensures we align with the official TWR calculation
    # and avoid "Phantom Returns" from non-trading day flows.
    full_idx = pv.sort_index().index
    pv_aligned = pv.sort_index()
    
    # 2. Map Tickers to Asset Classes
    ac_map = holdings.set_index("ticker")["asset_class"].to_dict()
    
    # Normalize dates
    if not tx_raw.empty: tx_raw["date"] = pd.to_datetime(tx_raw["date"]).dt.normalize()
    if not dividends.empty: dividends["date"] = pd.to_datetime(dividends["date"]).dt.normalize()
    
    # 3. Daily Shares per Ticker
    # We must construct shares_daily on the TRADING DAY index.
    # Logic: Cumulative sum of shares up to that trading day.
    if not tx_raw.empty:
        # Aggregate trades by date first
        shares_delta = tx_raw.pivot_table(index="date", columns="ticker", values="shares", aggfunc="sum").fillna(0.0)
        
        # Reindex to trading days
        # method='pad' ensures we carry forward the cumulative position
        # However, we need CUMULATIVE first.
        shares_daily_calendar = shares_delta.cumsum()
        
        # Align to trading days (ffill/pad picks up the latest position as of that day)
        # Note: We need 'asof' logic. reindex(method='ffill') works if index is sorted.
        shares_daily = shares_daily_calendar.reindex(full_idx, method='ffill').fillna(0.0)
        
        # Handle pre-inception shares if any (though usually start at 0)
        # If pv starts later than first trade, we capture existing state.
    else:
        shares_daily = pd.DataFrame(index=full_idx)
        
    # 4. Daily Prices per Ticker
    # Prices are already on trading days (mostly), but align strictly to pv index
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
    # CRITICAL FIX: Roll forward flows on non-trading days to next trading day
    flow_daily_ac = pd.DataFrame(0.0, index=full_idx, columns=mv_daily_ac.columns)
    
    def add_flow_mapped(date, col, amount):
        # Find the first trading day >= date
        try:
            loc = full_idx.searchsorted(date)
            if loc < len(full_idx):
                target_date = full_idx[loc]
                flow_daily_ac.loc[target_date, col] += amount
        except:
            pass

    if not tx_raw.empty:
        tx_mapped = tx_raw.copy()
        tx_mapped["asset_class"] = tx_mapped["ticker"].map(ac_map).fillna("Other")
        sec_tx = tx_mapped[tx_mapped["ticker"] != "CASH"]
        if not sec_tx.empty:
            grp = sec_tx.groupby(["date", "asset_class"])["amount"].sum()
            for (d, ac), amt in grp.items():
                add_flow_mapped(d, ac, -amt)
                add_flow_mapped(d, "CASH", amt)
                    
    ext_grp = cf_ext.groupby("date")["amount"].sum()
    for d, amt in ext_grp.items():
        add_flow_mapped(d, "CASH", amt)

    # 9. Daily Income (Dividends)
    inc_daily_ac = pd.DataFrame(0.0, index=full_idx, columns=mv_daily_ac.columns)
    if not dividends.empty:
        div_mapped = dividends.copy()
        div_mapped["asset_class"] = div_mapped["ticker"].map(ac_map).fillna("Other")
        grp = div_mapped.groupby(["date", "asset_class"])["amount"].sum()
        for (d, ac), amt in grp.items():
            # Roll forward dividend date too
            try:
                loc = full_idx.searchsorted(d)
                if loc < len(full_idx):
                    target_date = full_idx[loc]
                    inc_daily_ac.loc[target_date, ac] += amt
                    # FIX: Record dividend as a flow INTO Cash
                    flow_daily_ac.loc[target_date, "CASH"] += amt
            except:
                pass
                
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
    # Use explicit Timestamps for comparison
    if start_date:
        ts_start = pd.Timestamp(start_date)
    else:
        # For SI, we want to include the inception day (which is min()), so we need > (min - 1 day)
        ts_start = daily_df.index.get_level_values(0).min() - pd.Timedelta(days=1)

    ts_end = pd.Timestamp(end_date) if end_date else daily_df.index.get_level_values(0).max()
    
    # Slicing MultiIndex (Date, AC)
    # STRICTLY GREATER (>) to respect Anchor Date boundary (GIPS standard)
    mask = (daily_df.index.get_level_values(0) > ts_start) & \
           (daily_df.index.get_level_values(0) <= ts_end)
    window = daily_df[mask].copy()
    
    if window.empty: return pd.DataFrame()

    # If SI mode (start_date is None), determine specific start dates per asset class
    ac_first_activity = {}
    if start_date is None:
        # Check window for activity: Start Value exists OR Flow occurred
        has_activity = (window["Start_MV"] > 0) | (window["Net_Flow"].abs() > 1e-6)
        active_subset = window[has_activity]
        flat = active_subset.index.to_frame(index=False)
        flat_cols = flat.columns
        # Group by Asset Class (level 1) and find min Date (level 0)
        ac_first_activity = flat.groupby(flat_cols[1])[flat_cols[0]].min().to_dict()

    # Determine Effective SNAP-BACK Dates for Audit Context (GIPS Consistency)
    # The math uses the first trading day in the window (window start),
    # but the logic relies on Start_MV which comes from the PRIOR trading day (Anchor).
    
    actual_start_idx = window.index.get_level_values(0).min()
    actual_end_idx = window.index.get_level_values(0).max()
    
    # Find the Anchor Date (The trading day strictly before the window start)
    # We need the full index to look backward
    full_dates = daily_df.index.get_level_values(0).unique().sort_values()
    loc = full_dates.searchsorted(actual_start_idx)
    
    if loc > 0:
        anchor_date = full_dates[loc - 1]
    else:
        # If starting at very beginning of history (index 0), anchor is the start date itself (Inception)
        anchor_date = actual_start_idx
        
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
    
    # Calculate Period Aggregates for Audit Modal
    # Note: For Start/End MV presentation, we want the period boundaries.
    # Start of period = Start_MV of the first day
    # End of period = End_MV of the last day
    ac_period_start = start_mv.iloc[0]
    ac_period_end = end_mv.iloc[-1]
    ac_period_flow = flows.sum()
    ac_period_inc = income.sum()
    
    # 10. Compile Results
    results = []
    for ac in total_linked_contrib.index:
        results.append({
            "Asset Class": ac,
            "Effect": total_effect[ac],
            "Contribution (%)": total_linked_contrib[ac] * 100.0,
            
            # Audit Meta
            "meta_frongello_sum_factors": link_factors.sum(),
            "meta_frongello_avg_denom": denom_p.mean(),
            
            # New Meta for Effect Calculation
            "meta_ac_start": ac_period_start.get(ac, 0.0),
            "meta_ac_end": ac_period_end.get(ac, 0.0),
            "meta_ac_flow": ac_period_flow.get(ac, 0.0),
            "meta_ac_inc": ac_period_inc.get(ac, 0.0),
            # Use Anchor Date to reflect GIPS snap-back logic in Audit Modal
            # If SI Mode, use asset specific start date if available
            "meta_ac_start_date": ac_first_activity.get(ac, anchor_date) if start_date is None else anchor_date,
            "meta_ac_end_date": actual_end_idx,
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
        if horizon == "Since Inception":
            horizon = "SI"
        elif horizon == "Last Close":
            horizon = "1D"
        
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
        request_data["meta_Return_start_date"] = start
        request_data["meta_Return_end_date"] = end
        request_data["meta_Return_flow"] = row_data.get("meta_Return_flow")
        
        # Pass Annualization Context
        request_data["meta_Return_is_annualized"] = row_data.get("meta_Return_is_annualized")
        request_data["meta_Return_days"] = row_data.get("meta_Return_days")
        
        return request_data
        
    return request_data

def _calculate_residual_return(data, df_explained, start_date=None, end_date=None):
    """
    Helper to calculate the 'residual' return (Cash/Recon).
    
    GIPS COMPLIANCE:
    - For SI (start_date is None): Uses data["pl_si"] from portfolio_engine.py as the
      authoritative Source of Truth. This ensures the Attribution sum + Residual
      reconciles EXACTLY to the P/L displayed on the Overview page.
    - For non-SI periods (daily/weekly/monthly): Calculates P/L independently
      since the engine doesn't precompute P/L for arbitrary date ranges.
    
    The Residual captures:
    - Timing differences between Frongello daily effects and GIPS P/L formula
    - Rounding/floating-point precision
    - Any flows or income not attributed to specific asset classes
    """
    # 1. TWR Residual (Must be CUMULATIVE to match Frongello Sum)
    pv = data["pv"]
    cf_ext = data.get("cf_ext")
    inception = data["inception_date"]
    
    # Establish calculation window attributes
    calc_start = inception if start_date is None else pd.Timestamp(start_date)
    if not pv.empty:
        calc_end = pv.index.max() if end_date is None else pd.Timestamp(end_date)
    else:
        calc_end = calc_start

    # TWR Calculation Logic
    if not pv.empty:
        # TWR Engine expects the exact Inception Date (Day 1) to capture the initial funding return
        # But Frongello/Attribution logic often passes (Day 1 - 1 Day) as the anchor.
        # We must snap forward to Inception Date if the passed start is before data exists.
        twr_start = calc_start
        if twr_start < pv.index.min():
             twr_start = pv.index.min()
             
        # Compute Cumulative TWR (no annualization)
        twr_cum = compute_period_twr(pv, cf_ext, twr_start, calc_end)
    else:
        twr_cum = 0.0
        
    if pd.isna(twr_cum): twr_cum = 0.0
    
    explained_twr_pct = df_explained["Contribution (%)"].sum()
    residual_pct = (twr_cum * 100.0) - explained_twr_pct
    
    # 2. P/L Residual
    # ================================================================
    # GIPS COMPLIANCE FIX:
    # For SI mode (start_date is None), use the authoritative pl_si
    # from portfolio_engine.py. This value is calculated with proper
    # inception handling:
    #   - MV_start = 0 (nothing existed before inception)
    #   - Flows >= inception date are captured (Day 1 funding)
    #   - MV_end uses effective_as_of anchor
    #
    # This guarantees: Sum(Asset Class Effects) + Residual = Overview SI P/L
    # ================================================================
    
    is_si_mode = (start_date is None)
    
    if is_si_mode:
        # Use the engine's authoritative SI P/L (Source of Truth)
        pl_target = data.get("pl_si", 0.0)
        if pd.isna(pl_target):
            pl_target = 0.0
    else:
        # Non-SI periods: Calculate P/L independently for the specific window
        if not pv.empty:
            # Determine explicit MV Start/End for P/L
            
            # A. Start Value
            mv_start = 0.0
            if calc_start in pv.index:
                mv_start = float(pv.loc[calc_start])
            elif calc_start < pv.index.min():
                # Before inception -> 0 Value
                mv_start = 0.0
            else:
                # Snap backward to nearest valid trading day (Anchor)
                loc = pv.index.searchsorted(calc_start)
                if loc > 0:
                    mv_start = float(pv.iloc[loc-1])
                else: 
                    mv_start = 0.0

            # B. End Value
            if calc_end in pv.index:
                mv_end = float(pv.loc[calc_end])
            else:
                # Snap backward (GIPS standard for reporting end date)
                loc = pv.index.searchsorted(calc_end)
                if loc < len(pv.index) and pv.index[loc] == calc_end:
                     mv_end = float(pv.iloc[loc])
                elif loc > 0:
                     mv_end = float(pv.iloc[loc-1])
                else:
                     mv_end = float(pv.iloc[-1])

            # C. Flows (strictly > calc_start, <= calc_end)
            if cf_ext is not None and not cf_ext.empty:
                relevant_flows = cf_ext[(cf_ext["date"] > calc_start) & (cf_ext["date"] <= calc_end)]
                total_invested = relevant_flows["amount"].sum()
            else:
                total_invested = 0.0
                
            pl_target = mv_end - mv_start - total_invested
        else:
            pl_target = 0.0
        
    explained_pl = df_explained["Effect"].sum()
    residual_pl = pl_target - explained_pl
    
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

def get_active_strategy_table(data, benchmarks=None):
    """
    Returns summary table of Active Risk metrics (Beta, Tracking Error)
    compared to major benchmarks.
    """
    if benchmarks is None:
        benchmarks = {
            "S&P 500": "SPY",
            "Cons 40/60": "AOK",
            "Global 60/40": "AOR"
        }
    
    rows = []
    
    for name, ticker in benchmarks.items():
        metrics = calculate_active_metrics(data, ticker)
        
        beta = metrics.get("beta")
        te = metrics.get("te")
        
        beta_str = f"{beta:.2f}" if isinstance(beta, (int, float)) else "N/A"
        te_str = f"{te:.2f}%" if isinstance(te, (int, float)) else "N/A"
        
        # Avoid duplicate ticker display if already in name
        display_name = name
        if f"({ticker})" not in name:
            display_name = f"{name} ({ticker})"
            
        rows.append({
            "Benchmark": display_name,
            "Beta": beta_str,
            "Tracking Error": te_str
        })
        
    return pd.DataFrame(rows)

def get_data_source_summary(data):
    """
    Analyzes the data cache to determine if FMP was used for all sectors.
    """
    if not data:
        return None
        
    sec_table = data.get("sec_table_current", pd.DataFrame())
    
    tickers = [t for t in sec_table["ticker"].unique() if t != "CASH"]
    
    fallbacks = []
    source_counts = defaultdict(int)
    all_fmp = True
    
    for t in tickers:
        meta = _METADATA_CACHE.get(t, {})
        
        # If it's the old style (direct weights dict), it's likely from the original code which tried FMP then YF
        # But if we are seeing 403s now, any existing cache is probably YF or old FMP.
        # Let's assume Unknown = Yahoo Finance for current state clarity.
        if isinstance(meta, dict):
            source = meta.get("source", "Yahoo Finance (Cached)")
        else:
            source = "Yahoo Finance (Cached)"
        
        # Consolidate Source Labels for cleaner UI
        if source == "YF":
            source_label = "Yahoo Finance"
        elif source == "Equity Fallback":
            source_label = "Equity Lookup"
        elif source == "Unknown":
            source_label = "Yahoo Finance (Cached)"
        else:
            source_label = source
            
        source_counts[source_label] += 1
        
        if source_label != "FMP":
            all_fmp = False
            fallbacks.append((t, source_label))
            
    return {
        "all_fmp": all_fmp,
        "sources": dict(source_counts),
        "fallbacks": fallbacks,
        "has_errors": False # Explicitly silenced
    }


def get_price_source_summary(data):
    """
    Returns the price source metadata for UI display.
    
    This summarizes whether prices came from FMP, yfinance, or a mix of both.
    Used by the price source badge component.
    """
    if not data:
        return None
    
    prices = data.get("prices", pd.DataFrame())
    
    if prices.empty or not hasattr(prices, 'attrs'):
        return None
    
    source_metadata = prices.attrs.get('source_metadata', {})
    
    if not source_metadata:
        # Default to yfinance-only if no metadata present
        return {
            "FMP": [],
            "yfinance": list(prices.columns),
            "mixed": [],
            "fmp_range": (None, None),
            "yf_range": (prices.index.min(), prices.index.max()) if not prices.empty else (None, None),
        }
    
    return source_metadata




# ============================================================
# TAX AUTHORITY VISUALS
# ============================================================

def get_tax_liability_sunburst(open_lots, realized_events, theme="light"):
    """
    Upgraded Institutional Sunburst for Total Tax Threat.
    Hierarchy: Total Liability -> Status (Realized vs Unrealized) -> Term (ST/LT) -> Ticker.
    Fixes branchvalues="total" crash by filtering for positive liabilities only.
    """
    # 1. Filter for positive liabilities ONLY to ensure branchvalues="total" math works
    # This prevents the crash where Children sum to > Parent due to negative tax credits
    ol_pos = open_lots[open_lots["Est Tax Liability"] > 0].copy() if not open_lots.empty else pd.DataFrame()
    re_pos = realized_events[realized_events["Tax Impact"] > 0].copy() if not realized_events.empty else pd.DataFrame()

    if ol_pos.empty and re_pos.empty:
        return go.Figure().update_layout(
            title="No Tax Liability Detected", 
            template="plotly_dark"
        )

    # 2. Aggregate Data
    unrealized_total = ol_pos["Est Tax Liability"].sum() if not ol_pos.empty else 0
    realized_total = re_pos["Tax Impact"].sum() if not re_pos.empty else 0
    total_liab = unrealized_total + realized_total
    
    data = []
    
    # --- LEVEL 0: ROOT ---
    data.append({
        "id": "Total Liability", "parent": "", "label": "Total Liability", 
        "value": total_liab, "color": GLOBAL_PALETTE[0], "status": "Root"
    })

    # --- LEVEL 1: STATUS ---
    if realized_total > 0:
        data.append({
            "id": "Realized", "parent": "Total Liability", "label": "Realized", 
            "value": realized_total, "color": "#2c3e50", "status": "Realized"
        })
    if unrealized_total > 0:
        data.append({
            "id": "Unrealized", "parent": "Total Liability", "label": "Unrealized", 
            "value": unrealized_total, "color": "#7f8c8d", "status": "Unrealized"
        })

    # --- LEVEL 2: TERM ---
    # Use muted palette colors that harmonize with CYBORG theme
    st_dark = GLOBAL_PALETTE[2]   # muted red
    lt_dark = GLOBAL_PALETTE[0]   # steel blue
    st_light = GLOBAL_PALETTE[3]  # soft red-gray
    lt_light = GLOBAL_PALETTE[1]  # soft gray-blue

    if not re_pos.empty:
        rt = re_pos.groupby("Term")["Tax Impact"].sum()
        for term, val in rt.items():
            color = st_dark if "Short" in term else lt_dark
            data.append({
                "id": f"Realized_{term}", "parent": "Realized", "label": term, 
                "value": val, "color": color, "status": "Realized"
            })
    if not ol_pos.empty:
        ut = ol_pos.groupby("Term")["Est Tax Liability"].sum()
        for term, val in ut.items():
            color = st_light if "Short" in term else lt_light
            data.append({
                "id": f"Unrealized_{term}", "parent": "Unrealized", "label": term, 
                "value": val, "color": color, "status": "Unrealized"
            })

    # --- LEVEL 3: TICKER ---
    if not re_pos.empty:
        rtk = re_pos.groupby(["Term", "Ticker"])["Tax Impact"].sum().reset_index()
        for _, row in rtk.iterrows():
            color = st_dark if "Short" in row["Term"] else lt_dark
            data.append({
                "id": f"Realized_{row['Term']}_{row['Ticker']}", "parent": f"Realized_{row['Term']}", 
                "label": row["Ticker"], "value": row["Tax Impact"], "color": color, "status": "Realized"
            })
    if not ol_pos.empty:
        utk = ol_pos.groupby(["Term", "Ticker"])["Est Tax Liability"].sum().reset_index()
        for _, row in utk.iterrows():
            color = st_light if "Short" in row["Term"] else lt_light
            data.append({
                "id": f"Unrealized_{row['Term']}_{row['Ticker']}", "parent": f"Unrealized_{row['Term']}", 
                "label": row["Ticker"], "value": row["Est Tax Liability"], "color": color, "status": "Unrealized"
            })

    df_sun = pd.DataFrame(data)

    # Custom Tooltip Logic using customdata
    # Status column is already in df_sun
    df_sun["status_marker"] = df_sun["status"].apply(lambda x: "(Locked In)" if x == "Realized" else ("(Potential)" if x == "Unrealized" else ""))

    fig = go.Figure(go.Sunburst(
        ids=df_sun["id"],
        labels=df_sun["label"],
        parents=df_sun["parent"],
        values=df_sun["value"],
        branchvalues="total",
        marker=dict(colors=df_sun["color"]),
        customdata=df_sun["status_marker"],
        hovertemplate="<b>%{label}</b><br>Liability: $%{value:,.2f} %{customdata}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(t=10, l=10, r=10, b=10),
        height=350,
        annotations=[
            dict(
                text=f"<b>{fmt_dollar_clean(total_liab)}</b>",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=14, color="white"),
            )
        ]
    )
    return fig

def get_tax_tactical_radar(open_lots, theme="light"):
    """
    Tactical Radar for Harvest vs Hold decisions.
    X: Days Held, Y: Unrealized P/L, Size: Market Value.
    """
    if open_lots.empty:
        return go.Figure()

    df = open_lots.copy()
    
    # Days Held calculation if not present
    if "Days Held" not in df.columns:
        df["Days Held"] = (datetime.now() - df["Date Acquired"]).dt.days

    # Colors based on Gain/Loss
    df["Status"] = np.where(df["Unrealized P/L"] >= 0, "Gain", "Loss")
    color_map = {"Gain": GLOBAL_PALETTE[4], "Loss": GLOBAL_PALETTE[2]} # Greenish for gain, Reddish for loss

    # Fix Date Formatting for Tooltip (Remove timestamp)
    df["Date Acquired Display"] = pd.to_datetime(df["Date Acquired"]).dt.strftime("%Y-%m-%d")

    # Opacity gradient — larger positions more opaque, smaller ones fade
    mv_vals = df["Market Value"]
    mv_min, mv_max = mv_vals.min(), mv_vals.max()
    if mv_max > mv_min:
        df["opacity"] = 0.35 + 0.65 * (mv_vals - mv_min) / (mv_max - mv_min)
    else:
        df["opacity"] = 0.8

    fig = px.scatter(
        df,
        x="Days Held",
        y="Unrealized P/L",
        size="Market Value",
        color="Status",
        color_discrete_map=color_map,
        hover_name="Ticker",
        opacity=None,  # we set per-point below
        custom_data=["Ticker", "Shares", "Date Acquired Display"]
    )

    # Apply per-point opacity and marker styling for depth
    for trace in fig.data:
        status = trace.name  # "Gain" or "Loss"
        mask = df["Status"] == status
        trace.marker.opacity = df.loc[mask, "opacity"].values
        trace.marker.line = dict(width=1, color='rgba(255,255,255,0.3)')

    # Hover Template customization
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>" +
            "Shares: %{customdata[1]:,.2f}<br>" +
            "Acquired: %{customdata[2]}<br>" +
            "Days Held: %{x}<br>" +
            "P/L: $%{y:,.2f}<extra></extra>"
        )
    )

    # Vertical line at 365 (LT Cliff)
    fig.add_vline(x=365, line_dash="dash", line_color="gray", annotation_text="LT Cliff", annotation_position="top left")
    
    # Horizontal line at 0 P/L
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    # Quadrant labels for tactical context
    x_max = df["Days Held"].max() * 1.05 if not df.empty else 730
    y_max = df["Unrealized P/L"].max() if not df.empty else 1000
    y_min = df["Unrealized P/L"].min() if not df.empty else -1000
    quadrant_labels = [
        dict(x=182, y=y_min * 0.7, text="HARVEST", showarrow=False,
             font=dict(size=13, color="rgba(192,80,77,0.45)"), xref="x", yref="y"),
        dict(x=x_max * 0.75, y=y_min * 0.7, text="HOLD", showarrow=False,
             font=dict(size=13, color="rgba(140,156,177,0.45)"), xref="x", yref="y"),
        dict(x=x_max * 0.75, y=y_max * 0.7, text="LT GAIN", showarrow=False,
             font=dict(size=13, color="rgba(155,187,89,0.45)"), xref="x", yref="y"),
        dict(x=182, y=y_max * 0.7, text="ST GAIN", showarrow=False,
             font=dict(size=13, color="rgba(242,194,0,0.45)"), xref="x", yref="y"),
    ]

    fig.update_layout(
        template="plotly_dark",
        margin=dict(t=30, l=50, r=30, b=50),
        xaxis_title="Days Held",
        yaxis_title="Unrealized P/L ($)",
        showlegend=False,
        height=350,
        annotations=quadrant_labels
    )
    
    # Format X and Y axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', tickformat="$,.0f")

    return fig

def get_rebalancing_recommendations(cash_to_deploy=0, allow_sales=False):
    """
    Generates a list of buy/sell recommendations based on current portfolio state and targets.
    Replicates logic from pages/rebalancing.py without modifying the UI.
    """
    data = get_data()
    if not data:
        return []
    
    # Extract required data
    sec_table = data["sec_table_current"].copy()
    holdings = data["holdings"].copy()
    prices = data["prices"]
    
    if sec_table.empty:
        return []
    
    # Exclude CASH from investment calculations
    invested_df = sec_table[sec_table["ticker"] != "CASH"].copy()
    
    # Ensure target_pct is available
    if "target_pct" not in invested_df.columns:
        invested_df = invested_df.merge(
            holdings[["ticker", "target_pct"]],
            on="ticker",
            how="left"
        )
        invested_df["target_pct"] = invested_df["target_pct"].fillna(0.0)
    
    # Current portfolio value (excluding cash)
    current_total = invested_df["market_value"].sum()
    
    # Pro-forma total (current + new cash)
    proforma_total = current_total + cash_to_deploy
    
    # Current weights
    invested_df["current_weight_pct"] = (invested_df["market_value"] / current_total * 100) if current_total > 0 else 0
    
    target_df = invested_df.copy()
    target_df["target_dollar"] = (target_df["target_pct"] / 100) * proforma_total
    target_df["raw_diff"] = target_df["target_dollar"] - target_df["market_value"]
    target_df["drift"] = (target_df["target_pct"] - target_df["current_weight_pct"])
    
    # Initialize columns
    target_df["action"] = "Hold"
    target_df["recommend_amount"] = 0.0
    
    sale_proceeds = 0.0
    
    # --- SALES LOGIC ---
    if allow_sales:
        overweight_mask = target_df["raw_diff"] < -0.01
        candidates = {}
        for _, row in target_df[overweight_mask].iterrows():
            candidates[row["ticker"]] = abs(row["raw_diff"])
            
        sales_res = calculate_tax_optimized_sales(candidates, avoid_st_gains=True)
        sales_df_res = sales_res["sales_df"]
        sale_proceeds = sales_res["total_proceeds"]
        
        if not sales_df_res.empty:
            sales_grp = sales_df_res.groupby("Ticker")[["Proceeds"]].sum()
            for ticker, row in sales_grp.iterrows():
                mask = target_df["ticker"] == ticker
                if mask.any():
                    target_df.loc[mask, "recommend_amount"] = -row["Proceeds"]
                    target_df.loc[mask, "action"] = "Sell"

    # --- BUYS LOGIC ---
    total_cash_available = float(cash_to_deploy) + sale_proceeds
    remaining_cash = total_cash_available
    
    buy_mask = (target_df["raw_diff"] > 0.01) & (target_df["recommend_amount"] == 0)
    target_df.loc[buy_mask, "full_buy_need"] = target_df.loc[buy_mask, "raw_diff"]
    target_df["buy_allocation"] = 0.0
    
    # Waterfall Allocation
    for i in range(10):
        if remaining_cash < 0.01:
            break
        mask_room = (target_df["buy_allocation"] < target_df.get("full_buy_need", 0)) & buy_mask
        if not mask_room.any():
            break
        
        current_drift_sum = target_df.loc[mask_room, "drift"].clip(lower=0).sum()
        
        if current_drift_sum > 0:
            allocation_weights = target_df.loc[mask_room, "drift"].clip(lower=0) / current_drift_sum
            step_size = min(remaining_cash, remaining_cash * 0.5) # Conservative steps
            if i == 9: step_size = remaining_cash # Last step dump
            
            # Distribute
            amounts = allocation_weights * step_size
            # Cap at need
            needs = target_df.loc[mask_room, "full_buy_need"] - target_df.loc[mask_room, "buy_allocation"]
            actuals = np.minimum(amounts, needs)
            
            target_df.loc[mask_room, "buy_allocation"] += actuals
            remaining_cash -= actuals.sum()
        else:
            break
            
    # Finalize Buys
    buy_rows = target_df["buy_allocation"] > 0.01
    target_df.loc[buy_rows, "recommend_amount"] = target_df.loc[buy_rows, "buy_allocation"]
    target_df.loc[buy_rows, "action"] = "Buy"
    
    # Format Output
    recommendations = []
    for _, row in target_df.iterrows():
        if row["action"] != "Hold":
            # Get price for share count
            price = 0
            if row["ticker"] in prices:
                price = prices[row["ticker"]].iloc[-1]
            
            shares = 0
            if price > 0:
                shares = abs(row["recommend_amount"]) / price
                
            recommendations.append({
                "ticker": row["ticker"],
                "action": row["action"],
                "amount": abs(row["recommend_amount"]),
                "shares": shares,
                "price": price,
                "drift": row["drift"]
            })
            
    # Sort: Sells first, then Buys (largest to smallest)
    recommendations.sort(key=lambda x: (0 if x["action"] == "Sell" else 1, -x["amount"]))
    
    return recommendations
