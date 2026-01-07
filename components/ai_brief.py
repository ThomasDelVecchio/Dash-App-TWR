import pandas as pd
import numpy as np
from report_formatting import fmt_pct_clean, fmt_dollar_clean
from financial_math import get_portfolio_horizon_start, modified_dietz_for_ticker_window, annualize_return
from data_loader import fetch_price_history

def generate_ai_summary(data):
    """
    Generates a natural-language executive summary of the portfolio.
    Simulates an LLM response using template-based logic.
    """
    if not data:
        return "Portfolio data is currently initializing..."
        
    # Extract Key Metrics
    metrics = data.get("snapshot_metrics", {})
    # If not pre-calculated, calculate basics
    pv = data["pv"]
    current_mv = pv.iloc[-1] if not pv.empty else 0.0
    
    twr_df = data["twr_df"]
    
    # Returns
    def get_ret(h):
        row = twr_df[twr_df["Horizon"] == h]
        return row["Return"].iloc[0] if not row.empty else 0.0
        
    ret_mtd = get_ret("MTD")
    ret_1d = get_ret("1D")
    
    # --- DOLLAR IMPACT (1D) ---
    day_pl = 0.0
    if abs(ret_1d) > 0:
        # Approximate daily P/L: Current MV - (Current MV / (1 + r))
        # This assumes no external flows today, which is standard for quick intraday estimates.
        day_pl = current_mv - (current_mv / (1 + ret_1d))
        
    day_pl_str = fmt_dollar_clean(day_pl)
    if day_pl > 0: day_pl_str = "+" + day_pl_str
    
    # --- MARKET CONTEXT (S&P 500) ---
    spy_txt = ""
    
    # Fetch SPY specifically using Adj Close for Total Return comparison
    try:
        spy_hist = fetch_price_history(["SPY"], use_adj_close=True)
        if not spy_hist.empty and "SPY" in spy_hist.columns:
            spy_prices = spy_hist["SPY"].dropna()
            
            # Consistent 1D Logic using Helper
            inception_date = data.get("inception_date", pd.Timestamp.now())
            start_1d = get_portfolio_horizon_start(pv, inception_date, "1D")
            
            if start_1d is not None and not spy_prices.empty:
                # Filter SPY to match portfolio window exactly
                spy_window = spy_prices[spy_prices.index >= start_1d]
                # Clip to end date of portfolio
                spy_window = spy_window[spy_window.index <= pv.index.max()]
                
                if len(spy_window) >= 2:
                    # Calculate return over specific window
                    spy_1d = spy_window.iloc[-1] / spy_window.iloc[0] - 1.0
                    
                    # Context Text
                    diff = ret_1d - spy_1d
                    if diff > 0.001:
                        spy_txt = f", outperforming the S&P 500 ({spy_1d*100:+.2f}%)"
                    elif diff < -0.001:
                        spy_txt = f", trailing the S&P 500 ({spy_1d*100:+.2f}%)"
                    else:
                        spy_txt = f", tracking the S&P 500 ({spy_1d*100:+.2f}%)"
    except Exception:
        pass
    
    # Top Movers (1D)
    sec_current = data["sec_table_current"]
    if not sec_current.empty and "1D" in sec_current.columns:
        top_gainers = sec_current.sort_values("1D", ascending=False).head(2)
        top_losers = sec_current.sort_values("1D", ascending=True).head(2)
        
        best_stock = top_gainers.iloc[0]["ticker"]
        best_ret = top_gainers.iloc[0]["1D"]
        worst_stock = top_losers.iloc[0]["ticker"]
        worst_ret = top_losers.iloc[0]["1D"]
    else:
        best_stock, best_ret, worst_stock, worst_ret = "N/A", 0, "N/A", 0
        
    # Asset Allocation Drift (Overweight AND Underweight)
    holdings = data["holdings"]
    ac_weights = sec_current.groupby("asset_class")["weight"].sum() * 100
    targets = holdings.groupby("asset_class")["target_pct"].sum()
    
    max_over = 0
    over_ac = ""
    
    max_under = 0 # Track separately (will be negative)
    under_ac = ""
    
    for ac, w in ac_weights.items():
        t = targets.get(ac, 0)
        drift = w - t
        
        # Overweight
        if drift > 0 and drift > max_over:
            max_over = drift
            over_ac = ac
            
        # Underweight
        if drift < 0 and drift < max_under:
            max_under = drift
            under_ac = ac
            
    # Construct Narrative
    
    # Intro
    sentiment = "steady"
    if abs(ret_1d) > 0.01: sentiment = "volatile"
    if ret_1d > 0.005: sentiment = "strong"
    if ret_1d < -0.005: sentiment = "challenging"
    
    # "Today's session is strong, with the portfolio up +1.2% (+$1,250), outperforming the S&P 500 (+0.8%)."
    intro = f"Today's session is **{sentiment}**, with the portfolio currently **{ret_1d*100:+.2f}%** (**{day_pl_str}**){spy_txt}. "
    
    # MTD Context
    mtd_txt = ""
    if abs(ret_1d) < 0.0001: # Essentially flat
        mtd_txt = f"Monthly performance remains unchanged (**{ret_mtd*100:+.2f}%** MTD)."
    elif ret_1d > 0: # Positive Day
        if ret_mtd > 0:
            mtd_txt = f"This adds to a positive month (**{ret_mtd*100:+.2f}%** MTD)."
        else:
            mtd_txt = f"This helps recover some monthly losses (**{ret_mtd*100:+.2f}%** MTD)."
    else: # Negative Day
        if ret_mtd > 0:
            mtd_txt = f"This weighs on an otherwise positive month (**{ret_mtd*100:+.2f}%** MTD)."
        else:
            mtd_txt = f"This adds to monthly losses (**{ret_mtd*100:+.2f}%** MTD)."
    intro += mtd_txt
        
    # Drivers
    # "Leading the charge is NVDA (+5%), while TSLA (-2%) is creating some drag."
    drivers = f"Leading the charge is **{best_stock}** (**{best_ret*100:+.2f}%**), while **{worst_stock}** (**{worst_ret*100:+.2f}%**) is creating some drag."
    
    # Allocation
    # "On the allocation front, you are overweight US Tech (+6%), but significantly underweight International (-4%)."
    alloc_points = []
    if max_over > 5.0:
        alloc_points.append(f"overweight **{over_ac}** (+{max_over:.1f}%)")
    if abs(max_under) > 5.0:
        alloc_points.append(f"underallocated in **{under_ac}** ({max_under:.1f}%)")
        
    if alloc_points:
        alloc = "On the allocation front, note that you are " + " and ".join(alloc_points) + "."
    else:
        alloc = "Asset allocation remains balanced and close to targets."
        
    # Flows (if any recent)
    cf_ext = data["cf_ext"]
    last_flow_date = cf_ext["date"].max() if not cf_ext.empty else None
    flows_txt = ""
    if last_flow_date and (pd.Timestamp.now() - last_flow_date).days < 7:
        recent_amt = cf_ext[cf_ext["date"] == last_flow_date]["amount"].sum()
        flows_txt = f"\n\nRecent activity: **{fmt_dollar_clean(recent_amt)}** net flow on {last_flow_date.strftime('%m/%d')}."
        
    # Combine
    summary = f"{intro}\n\n{drivers}\n\n{alloc}{flows_txt}"
    
    return summary

def generate_ai_summary_period(data, start_date=None, end_date=None):
    """
    Generates a natural-language executive summary that respects a specific period.
    Supports Time Machine analysis via start_date and end_date.
    Annualizes returns only if duration > 1 year (handled by financial_math.annualize_return).
    """
    import dash_wrappers as dw
    if not data:
        return "Portfolio data is currently initializing..."
    
    # 1. Get TWR Curve for calculation
    twr_curve = dw._get_daily_twr_curve(data) 
    if twr_curve.empty:
        return "Insufficient data for summary."
        
    # Define Bounds
    eff_end = pd.Timestamp(end_date) if end_date else twr_curve.index.max()
    eff_start = pd.Timestamp(start_date) if start_date else twr_curve.index.min()
    
    # --- 1. PERIOD RETURN ---
    period_ret = 0.0
    
    # Find closest index <= eff_end
    end_locs = twr_curve.index[twr_curve.index <= eff_end]
    if not end_locs.empty:
        idx_end = end_locs[-1]
        val_end = twr_curve.loc[idx_end]
        
        # Determine Base Value (val_start)
        val_start = twr_curve.asof(eff_start) if eff_start > twr_curve.index.min() else 1.0
        if pd.isna(val_start): val_start = 1.0

        period_ret_cum = (val_end / val_start) - 1.0
        # Apply Annualization logic (>= 1yr in financial_math)
        period_ret = annualize_return(period_ret_cum, eff_start, eff_end)
        
    # --- 2. PERIOD P/L (DOLLAR) ---
    pv = data.get("pv")
    cf_ext = data.get("cf_ext")
    period_pl = 0.0
    
    if pv is not None and not pv.empty:
        # End MV
        end_locs_pv = pv.index[pv.index <= eff_end]
        if not end_locs_pv.empty:
            idx_end_pv = end_locs_pv[-1]
            end_mv = pv.loc[idx_end_pv]
        else:
            end_mv = 0.0
            
        # Start MV
        start_locs_pv = pv.index[pv.index < eff_start]
        if not start_locs_pv.empty:
            idx_start_pv = start_locs_pv[-1]
            start_mv = pv.loc[idx_start_pv]
        else:
            start_mv = 0.0
            
        # Net Flows in Window [start, end]
        flows_sum = 0.0
        if cf_ext is not None and not cf_ext.empty:
            mask = (cf_ext["date"] >= eff_start) & (cf_ext["date"] <= eff_end)
            flows_sum = cf_ext.loc[mask, "amount"].sum()
            
        period_pl = end_mv - start_mv - flows_sum

    pl_str = fmt_dollar_clean(period_pl)
    if period_pl > 0: pl_str = "+" + pl_str

    # --- 3. MARKET CONTEXT (S&P 500) ---
    spy_txt = ""
    try:
        spy_hist = fetch_price_history(["SPY"], use_adj_close=True)
        if not spy_hist.empty and "SPY" in spy_hist.columns:
            spy_prices = spy_hist["SPY"].dropna()
            
            # GIPS COMPLIANCE FIX: MATCH ANCHOR DATE LOGIC
            # If SI: use last price before inception to capture day 1. 
            # Otherwise use price ON start date (capturing return from close onwards).
            base_price = None
            market_start = data.get("inception_date", twr_curve.index.min())

            if eff_start <= market_start:
                 # Inception Logic: Look for price strictly before start
                 history_before = spy_prices[spy_prices.index < eff_start]
                 if not history_before.empty:
                      base_price = float(history_before.iloc[-1])
            
            # Fallback to .asof(eff_start) if mid-horizon or no pre-inception data
            if base_price is None:
                 base_price = spy_prices.asof(eff_start)
            
            end_price = spy_prices.asof(eff_end)
            
            if base_price and end_price:
                spy_ret = (end_price / base_price) - 1.0
                
                diff = period_ret - spy_ret
                if diff > 0.001:
                    spy_txt = f", outperforming the S&P 500 ({spy_ret*100:+.2f}%)"
                elif diff < -0.001:
                    spy_txt = f", trailing the S&P 500 ({spy_ret*100:+.2f}%)"
                else:
                    spy_txt = f", tracking the S&P 500 ({spy_ret*100:+.2f}%)"
    except Exception: pass

    # --- 4. TOP MOVERS (MODIFIED DIETZ) ---
    best_stock, best_ret, worst_stock, worst_ret = "N/A", 0, "N/A", 0
    try:
        # Optimization: use pre-calculated sec_table_current if available for tickers
        held_tickers = data["sec_table_current"]["ticker"].unique()
        tx_raw = data.get("tx_raw")
        dividends = data.get("dividends")
        prices = data.get("prices") or fetch_price_history(list(held_tickers))

        if not prices.empty:
            tx_grouped = tx_raw.groupby("ticker") if tx_raw is not None and not tx_raw.empty else {}
            div_grouped = dividends.groupby("ticker") if dividends is not None and not dividends.empty else {}

            perf_records = []
            for tik in held_tickers:
                if tik == 'CASH': continue
                tx_t = tx_grouped.get_group(tik) if (isinstance(tx_grouped, pd.core.groupby.DataFrameGroupBy) and tik in tx_grouped.groups) else pd.DataFrame(columns=["date", "shares", "amount"])
                div_t = div_grouped.get_group(tik) if (isinstance(div_grouped, pd.core.groupby.DataFrameGroupBy) and tik in div_grouped.groups) else pd.DataFrame(columns=["date", "amount"])
                
                if tik in prices.columns:
                    md = modified_dietz_for_ticker_window(tik, prices[tik], tx_t, eff_start, eff_end, div_t)
                    if not pd.isna(md):
                        # Apply Annualization logic (>= 1yr in financial_math)
                        md_ann = annualize_return(md, eff_start, eff_end)
                        perf_records.append({"ticker": tik, "ret": md_ann})
            
            if perf_records:
                perf_df = pd.DataFrame(perf_records)
                top_gainers = perf_df.nlargest(1, "ret")
                top_losers = perf_df.nsmallest(1, "ret")
                if not top_gainers.empty:
                    best_stock, best_ret = top_gainers.iloc[0]["ticker"], top_gainers.iloc[0]["ret"]
                if not top_losers.empty:
                    worst_stock, worst_ret = top_losers.iloc[0]["ticker"], top_losers.iloc[0]["ret"]
    except Exception: pass

    # --- 5. ASSET ALLOCATION DRIFT ---
    sec_current, holdings = data.get("sec_table_current", pd.DataFrame()), data.get("holdings", pd.DataFrame())
    max_over, over_ac, max_under, under_ac = 0, "", 0, ""
    if not sec_current.empty and not holdings.empty:
        ac_weights, targets = sec_current.groupby("asset_class")["weight"].sum() * 100, holdings.groupby("asset_class")["target_pct"].sum()
        for ac, w in ac_weights.items():
            drift = w - targets.get(ac, 0)
            if drift > max_over: max_over, over_ac = drift, ac
            if drift < max_under: max_under, under_ac = drift, ac

    # --- 6. NARRATIVE ---
    sentiment = "robust" if period_ret > 0.10 else "challenging" if period_ret < -0.10 else "steady"
    intro = f"Performance over this period has been **{sentiment}**, with the portfolio returning **{period_ret*100:+.2f}%** (**{pl_str}**){spy_txt}. "
    drivers = f"Leading the charge was **{best_stock}** (**{best_ret*100:+.2f}%**), while **{worst_stock}** (**{worst_ret*100:+.2f}%**) proved to be drag on performance." if best_stock != "N/A" else ""
    alloc_points = []
    if max_over > 5.0: alloc_points.append(f"overweight **{over_ac}** (+{max_over:.1f}%)")
    if abs(max_under) > 5.0: alloc_points.append(f"underallocated in **{under_ac}** ({max_under:.1f}%)")
    alloc_txt = " Positioning currently shows you are " + ", and ".join(alloc_points) + "." if alloc_points else ""
    
    return intro + drivers + alloc_txt
