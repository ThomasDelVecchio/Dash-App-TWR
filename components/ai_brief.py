import pandas as pd
import numpy as np
from report_formatting import fmt_pct_clean, fmt_dollar_clean
from financial_math import get_portfolio_horizon_start
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
    
    if ret_mtd > 0:
        intro += f"This adds to a positive month (**{ret_mtd*100:+.2f}%** MTD)."
    else:
        intro += f"This continues to weigh on monthly performance (**{ret_mtd*100:+.2f}%** MTD)."
        
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
