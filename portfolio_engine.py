import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

from data_loader import (
    load_holdings,
    load_cashflows_external,
    load_transactions_raw,
    fetch_price_history,
    load_dividends,
    CASHFLOWS_FILE,
)
from financial_math import (
    build_portfolio_value_series_from_flows,
    compute_period_twr,
    compute_horizon_twr,
    compute_security_modified_dietz,
    get_portfolio_horizon_start,
    modified_dietz_for_ticker_window,
    modified_dietz_for_asset_class_window,
    HORIZONS,
    ANNUALIZE_HORIZONS,
)


def run_engine(end_date=None):
    """
    Runs the full calculation pipeline but returns clean DataFrames instead of printing.
    NO math or logic is changed anywhere.
    """
    holdings = load_holdings()
    cashflows_ext = load_cashflows_external()
    transactions_raw = load_transactions_raw()
    dividends = load_dividends()

    # Determine full ticker universe before any clipping
    # This ensures we fetch prices for ALL tickers that appear in the full history,
    # which is required because build_portfolio_value_series_from_flows reads the full cashflows file.
    full_tickers = sorted(
        (set(holdings["ticker"]) | set(transactions_raw["ticker"])) - {"CASH"}
    )

    # =============================================================
    # TIME MACHINE LOGIC
    # =============================================================
    if end_date is not None:
        end_date = pd.Timestamp(end_date)
        
        # 1. Clip Transactions & Flows
        transactions_raw = transactions_raw[transactions_raw["date"] <= end_date]
        cashflows_ext = cashflows_ext[cashflows_ext["date"] <= end_date]
        dividends = dividends[dividends["date"] <= end_date]
        
        # 2. Reconstruct Holdings at end_date (Shares + Cash)
        #    This is required so build_portfolio_value_series_from_flows 
        #    sanity check passes, and mv_df reflects the correct point-in-time state.
        
        # Shares from trades
        if not transactions_raw.empty:
            computed_shares = transactions_raw.groupby("ticker")["shares"].sum()
        else:
            computed_shares = pd.Series(dtype=float)
            
        # Cash Balance (External + Net Trading + Dividends)
        ext_cash = cashflows_ext["amount"].sum()
        trading_cash = transactions_raw["amount"].sum() if not transactions_raw.empty else 0.0
        div_cash = dividends["amount"].sum() if not dividends.empty else 0.0
        total_cash = ext_cash + trading_cash + div_cash
        
        # Build new holdings rows
        new_rows = []
        # Keep tickers from original input to preserve asset class mapping for closed positions
        original_tickers = set(holdings["ticker"].unique()) if not holdings.empty else set()

        for t, s in computed_shares.items():
            # Always include to support "Show Exited Tickers" filter
            new_rows.append({"ticker": t, "shares": s})
        new_rows.append({"ticker": "CASH", "shares": total_cash})
        
        new_holdings = pd.DataFrame(new_rows)
        
        # Merge metadata (Asset Class, Target %) from original static file
        # Note: Historical tickers not in current holdings will get defaults.
        if not holdings.empty:
            meta_cols = ["ticker", "asset_class", "target_pct"]
            # Ensure columns exist in source
            valid_cols = [c for c in meta_cols if c in holdings.columns]
            if "ticker" in valid_cols:
                new_holdings = new_holdings.merge(holdings[valid_cols], on="ticker", how="left")
        
        # Fill missing
        if "asset_class" not in new_holdings.columns: new_holdings["asset_class"] = "Unknown"
        if "target_pct" not in new_holdings.columns: new_holdings["target_pct"] = 0.0
        
        new_holdings["asset_class"] = new_holdings["asset_class"].fillna("Unknown")
        new_holdings["target_pct"] = new_holdings["target_pct"].fillna(0.0)
        
        holdings = new_holdings

    # Use the full universe of tickers so that build_portfolio_value_series_from_flows
    # (which sees the entire cashflow history) finds columns for every ticker it encounters,
    # even if that ticker has 0 positions/activity in the clipped window.
    prices = fetch_price_history(full_tickers)
    
    # Clip Prices for Time Machine
    if end_date is not None:
        prices = prices[prices.index <= end_date]
        
        # Ensure prices extends to end_date (ffill if missing)
        # This handles cases where yfinance is missing data for the selected end_date,
        # or if end_date is a non-trading day but has flows.
        # Without this, build_portfolio_value_series_from_flows stops early and misses final flows,
        # causing a reconciliation error against holdings.
        if not prices.empty and prices.index.max() < end_date:
            last_row = prices.iloc[[-1]].copy()
            last_row.index = [end_date]
            prices = pd.concat([prices, last_row])

    # FIX for Time Machine:
    # build_portfolio_value_series_from_flows reads file from disk.
    # We must provide a clipped file to ensure consistency with clipped holdings.
    if end_date is not None:
        # Load, clip, save temp
        temp_path = "temp_clipped_cashflows.csv"
        raw_flows = pd.read_csv(CASHFLOWS_FILE)
        
        # Normalize columns to handle "Date" vs "date"
        raw_flows.columns = [c.lower() for c in raw_flows.columns]
        
        # Ensure date column is datetime for comparison
        if "date" in raw_flows.columns:
            raw_flows["date"] = pd.to_datetime(raw_flows["date"])
            raw_flows = raw_flows[raw_flows["date"] <= end_date]
        
        raw_flows.to_csv(temp_path, index=False)
        
        try:
            pv = build_portfolio_value_series_from_flows(holdings, prices, cashflows_path=temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        # Augment holdings with exited tickers from transactions for "Show Exited Tickers"
        if not transactions_raw.empty:
            current_tickers = set(holdings["ticker"])
            tx_tickers = set(transactions_raw["ticker"])
            exited = tx_tickers - current_tickers - {"CASH"}
            
            if exited:
                # We assume these are 0 shares if not in holdings file
                rows_to_add = []
                for t in exited:
                     rows_to_add.append({"ticker": t, "shares": 0.0, "asset_class": "Unknown", "target_pct": 0.0})
                
                if rows_to_add:
                    holdings = pd.concat([holdings, pd.DataFrame(rows_to_add)], ignore_index=True)

        pv = build_portfolio_value_series_from_flows(holdings, prices)

    # =============================================================
    # FIX: Clip PV to true inception date (institutionally correct)
    # =============================================================
    # True inception = earliest of (first external flow, first trade)

    if not cashflows_ext.empty:
        first_ext = cashflows_ext["date"].min()
    else:
        first_ext = None

    if not transactions_raw.empty:
        first_trade = transactions_raw["date"].min()
    else:
        first_trade = None

    # Determine true inception
    dates = [d for d in [first_ext, first_trade] if d is not None]
    if dates:
        true_inception = min(dates)
    else:
        true_inception = pv.index.min()

    # Clip PV to start no earlier than true inception
    pv = pv[pv.index >= true_inception]

    # Safety check — PV must exist after clipping
    if pv.empty:
        raise RuntimeError("PV could not be aligned to true inception date.")


    # Inception date logic (unchanged)
    # Determine correct inception = earliest economic activity
    dates = []

    if not cashflows_ext.empty:
        dates.append(cashflows_ext["date"].min())

    if not transactions_raw.empty:
        dates.append(transactions_raw["date"].min())

    dates.append(pv.index.min())

    inception_date = min(dates)

    # External flows only where PV exists
    cf = cashflows_ext[cashflows_ext["date"] >= pv.index.min()].copy()


    # ------ PORTFOLIO TWR (including SI) ------
    # We define ALL horizons, including SI, for the unified loop.
    ALL_HORIZONS = HORIZONS + ["SI"]
    
    results = {}
    for h in ALL_HORIZONS:
        results[h] = compute_horizon_twr(pv, cf, inception_date, h)

    # Convert dict → DataFrame (excluding SI from main columns if preferred, but usually we want it)
    # The original return structure had SI separate (twr_since_inception).
    # We will keep 'twr_since_inception' separate for backward compatibility of return signature.
    twr_since_inception = results.pop("SI", np.nan)
    
    twr_df = (
        pd.DataFrame(results, index=[0])
        .T.reset_index()
        .rename(columns={"index": "Horizon", 0: "Return"})
    )
    
    # ---- ANNUALIZED SINCE-INCEPTION TWR (if > 1 year) ----
    as_of = pv.index.max()
    days_since_inception = (as_of - inception_date).days
    if pd.notna(twr_since_inception) and days_since_inception > 365:
        years_since_inception = days_since_inception / 365.0
        twr_since_inception_annualized = (1.0 + twr_since_inception) ** (1.0 / years_since_inception) - 1.0
    else:
        twr_since_inception_annualized = np.nan


    # ---- SINCE-INCEPTION PORTFOLIO P/L (ECONOMIC, MATCHES BUILD_REPORT) ----
    # P/L_SI = MV_end − MV_start − net_external_flows(start, end)
    as_of = pv.index.max()
    start = inception_date

    # Map inception_date onto the first PV date on/after it
    if start not in pv.index:
        pv_idx = pv.index.sort_values()
        pos = pv_idx.searchsorted(start)
        start = pv_idx[pos]

    # --- FIX: Handle Inception explicitly to capture Day 1 ---
    if start == inception_date:
        mv_start = 0.0
        # Use >= to include flows ON the start date (Funding Day)
        if not cashflows_ext.empty:
            mask = (cashflows_ext["date"] >= start) & (cashflows_ext["date"] <= as_of)
            net_ext = float(cashflows_ext.loc[mask, "amount"].sum())
        else:
            net_ext = 0.0
    else:
        mv_start = float(pv.loc[start])
        # Use > to exclude start date flows (Standard Period)
        if not cashflows_ext.empty:
            mask = (cashflows_ext["date"] > start) & (cashflows_ext["date"] <= as_of)
            net_ext = float(cashflows_ext.loc[mask, "amount"].sum())
        else:
            net_ext = 0.0

    mv_end = float(pv.loc[as_of])

    pl_since_inception = mv_end - mv_start - net_ext


    # ------ MV + weights (unchanged math) ------
    latest_prices = prices.iloc[-1]
    mv_rows = []
    total_mv = 0.0

    for _, row in holdings.iterrows():
        t = row["ticker"]
        q = row["shares"]

        if t == "CASH":
            mv = q
        else:
            mv = q * latest_prices.get(t, np.nan)

        mv_rows.append({"ticker": t, "shares": q, "market_value": mv})
        if not np.isnan(mv):
            total_mv += mv

    mv_df = pd.DataFrame(mv_rows)
    mv_df["weight"] = mv_df["market_value"] / total_mv
    mv_df = mv_df.merge(
        holdings[["ticker", "asset_class", "target_pct"]],
        on="ticker",
        how="left"
    )

    # ------ Security-level MD (Unified Loop for ALL Horizons + SI) ------
    sec_md_df = compute_security_modified_dietz(
        transactions_raw, prices, holdings, dividends=dividends, horizons=ALL_HORIZONS
    )

    if sec_md_df.empty:
        sec_table = pd.DataFrame()
        class_df = pd.DataFrame()
        return twr_df, sec_table, class_df, pv, twr_since_inception, twr_since_inception_annualized, pl_since_inception

    # Build SEC TABLE (same ordering, same logic)
    cols_to_show = (
        ["ticker", "asset_class", "shares", "market_value", "weight",
         "first_date", "last_date", "days_held"] + HORIZONS + ["SI"]
    )

    sec_table = mv_df.merge(sec_md_df, on="ticker", how="left")
    
    # Identify Audit Meta Columns present in sec_md_df
    meta_cols = [c for c in sec_md_df.columns if c.startswith("meta_")]
    
    for c in cols_to_show:
        if c not in sec_table.columns:
            sec_table[c] = np.nan
            
    # Include meta columns in the final table
    final_cols = cols_to_show + meta_cols
    # Filter only existing
    final_cols = [c for c in final_cols if c in sec_table.columns]
            
    sec_table = sec_table[final_cols].sort_values("market_value", ascending=False)

    # NEW: Hardcode CASH returns to 0.0% across all horizons
    cash_mask = sec_table["ticker"] == "CASH"
    for h in ALL_HORIZONS:
        if h in sec_table.columns:
            sec_table.loc[cash_mask, h] = 0.0

    # ------ Asset-class MD (Unified Loop for ALL Horizons + SI) ------
    
    # ---------------------------------------------------------
    # FIX: Build Full Universe Map (History + Current)
    # This ensures liquidated asset classes are still calculated.
    # ---------------------------------------------------------
    all_tickers = set(transactions_raw["ticker"].unique()) | set(holdings["ticker"].unique())
    all_tickers.discard("CASH")
    
    # Map Ticker -> Asset Class (Default to 'Unknown' if missing from holdings file)
    ticker_ac_map = holdings.set_index("ticker")["asset_class"].to_dict()
    
    ac_groups = {}
    for t in all_tickers:
        ac = ticker_ac_map.get(t, "Unknown")
        if ac not in ac_groups:
            ac_groups[ac] = []
        ac_groups[ac].append(t)
        
    class_rows = []
    for asset_class, class_tickers in ac_groups.items():
        row = {"asset_class": asset_class}

        # NEW: Hardcode Cash asset class returns to 0.0%
        if asset_class == "CASH":
            for h in ALL_HORIZONS:
                row[h] = 0.0
        else:
            # Pre-calculate asset class inception for gating logic
            class_inception = None
            class_latest_inception = None  # Track the latest ticker inception in the class
            for t in class_tickers:
                if t == "CASH": continue
                tx_t = transactions_raw[transactions_raw["ticker"] == t]
                if not tx_t.empty:
                    first_trade_t = tx_t["date"].min()
                    if class_inception is None:
                        class_inception = first_trade_t
                        class_latest_inception = first_trade_t
                    else:
                        class_inception = min(class_inception, first_trade_t)
                        class_latest_inception = max(class_latest_inception, first_trade_t)

            for h in ALL_HORIZONS:
                # ---------------------------------------------------------
                # Determine Start Date
                # ---------------------------------------------------------
                
                # SI Special Case: Uses Asset Class Inception
                if h == "SI":
                    if class_inception is None:
                        row[h] = np.nan
                        continue
                    start_date = class_inception
                    eligible_tickers = class_tickers  # All tickers eligible for SI
                    
                # Standard Horizons: Use Portfolio Logic
                else:
                    start_date = get_portfolio_horizon_start(pv, inception_date, h)
                    
                    # Check 1: Horizon is invalid (e.g. Portfolio too new for 3Y)
                    if start_date is None or start_date >= as_of:
                        row[h] = np.nan
                        continue

                    # =========================================================
                    # GIPS FIX: Only include tickers with full horizon history
                    # =========================================================
                    # A ticker is eligible for horizon h ONLY if its first trade
                    # was STRICTLY BEFORE the horizon start date.
                    # This prevents new positions from inflating AC returns.
                    eligible_tickers = []
                    for t in class_tickers:
                        if t == "CASH": continue
                        tx_t = transactions_raw[transactions_raw["ticker"] == t]
                        if not tx_t.empty:
                            first_trade_t = tx_t["date"].min()
                            # GIPS: Security must have existed STRICTLY BEFORE horizon start
                            if first_trade_t < start_date:
                                eligible_tickers.append(t)
                    
                    # If no tickers are eligible, return N/A for this horizon
                    if not eligible_tickers:
                        row[h] = np.nan
                        continue
                    
                    # Recalculate class inception for eligible tickers only
                    # (This ensures proper snapping if all eligible tickers started mid-period)
                    eligible_class_inception = None
                    for t in eligible_tickers:
                        tx_t = transactions_raw[transactions_raw["ticker"] == t]
                        if not tx_t.empty:
                            first_trade_t = tx_t["date"].min()
                            if eligible_class_inception is None:
                                eligible_class_inception = first_trade_t
                            else:
                                eligible_class_inception = min(eligible_class_inception, first_trade_t)
                    
                    # Snap to eligible class inception if needed
                    if eligible_class_inception is not None and eligible_class_inception > start_date:
                        start_date = eligible_class_inception

                # MODIFIED DIETZ CALCULATION
                # Now that start_date is correct and tickers are filtered, this will produce accurate returns.
                
                ret = modified_dietz_for_asset_class_window(
                    tickers=eligible_tickers,
                    prices=prices,
                    tx_all=transactions_raw,
                    start=start_date,
                    end=as_of,
                    dividends=dividends,
                    return_components=True,
                )
                
                if isinstance(ret, dict):
                    row[h] = ret["return"]
                    row[f"meta_{h}_start"] = ret["start_val"]
                    row[f"meta_{h}_end"] = ret["end_val"]
                    row[f"meta_{h}_flow"] = ret["net_flow"]
                    row[f"meta_{h}_inc"] = ret["income"]
                    row[f"meta_{h}_denom"] = ret["denom"]
                else:
                    row[h] = ret
                    
        class_rows.append(row)

    class_df = pd.DataFrame(class_rows)

    # Sort by MV
    class_mv = mv_df.groupby("asset_class", as_index=False)["market_value"].sum()
    class_mv = class_mv.rename(columns={"market_value": "class_market_value"})
    class_df = class_df.merge(class_mv, on="asset_class", how="left")
    class_df = class_df.sort_values("class_market_value", ascending=False)
    # class_df = class_df[["asset_class"] + HORIZONS]  <-- REMOVED: This strips meta columns!


    # ------ Annualize multi-year horizons (3Y, 5Y) for reporting ------

    for label, years in ANNUALIZE_HORIZONS.items():
        # Portfolio TWR (twr_df is long-form)
        mask = twr_df["Horizon"] == label
        if mask.any():
            vals = twr_df.loc[mask, "Return"]
            twr_df.loc[mask, "Return"] = np.where(
                vals.notna(),
                (1.0 + vals) ** (1.0 / years) - 1.0,
                np.nan,
            )

        # Security-level MD table (wide form)
        if not sec_table.empty and label in sec_table.columns:
            vals = sec_table[label]
            sec_table[label] = np.where(
                vals.notna(),
                (1.0 + vals) ** (1.0 / years) - 1.0,
                np.nan,
            )

        # Asset-class MD table (wide form)
        if not class_df.empty and label in class_df.columns:
            vals = class_df[label]
            class_df[label] = np.where(
                vals.notna(),
                (1.0 + vals) ** (1.0 / years) - 1.0,
                np.nan,
            )

    return twr_df, sec_table, class_df, pv, twr_since_inception, twr_since_inception_annualized, pl_since_inception


# ============================================================
# Helpers from build_report.py (Logic Extracted)
# ============================================================

def calculate_horizon_pl(pv: pd.Series, inception_date: pd.Timestamp, cf_ext: pd.DataFrame, h: str):
    """
    Portfolio P/L over horizon h using the SAME horizon start as TWR.
    P/L = MV_end − MV_start − net_external_flows(start, end)
    """
    as_of = pv.index.max()

    # Simplified: 'get_portfolio_horizon_start' now handles SI directly
    start = get_portfolio_horizon_start(pv, inception_date, h)
    
    if start is None or start >= as_of:
        return None

    # ----- Map horizon start onto actual PV index -----
    if start not in pv.index:
        pv_idx = pv.index.sort_values()
        pos = pv_idx.searchsorted(start)
        if pos >= len(pv_idx):
            return None
        start = pv_idx[pos]

    mv_start = float(pv.loc[start])
    mv_end   = float(pv.loc[as_of])


    # flows strictly after start, up to and including as_of
    net_flows = 0.0
    if cf_ext is not None and not cf_ext.empty:
        mask = (cf_ext["date"] > start) & (cf_ext["date"] <= as_of)
        net_flows = float(cf_ext.loc[mask, "amount"].sum())

    pl = mv_end - mv_start - net_flows
    return pl

def calculate_ticker_pl(ticker, h, prices, pv_as_of, transactions, sec_only, raw_start=None, dividends=None, portfolio_inception=None, return_components=False):
    """
    Correct economic P/L for a single ticker over a horizon.
    P/L = MV_end - MV_start - Net Capital Flows + Income
    
    IMPORTANT: For SI calculations, aligns with portfolio inception to ensure
    Cash/Recon reconciliation is accurate.
    """
    if ticker == "CASH":
        # Treat CASH as 0% return, 0 P/L for horizons in this table.
        if return_components:
            return {"pl": 0.0, "start": 0.0, "end": 0.0, "flow": 0.0, "inc": 0.0, "denom": 0.0}
        return 0.0

    # price series
    if ticker not in prices.columns:
        return None
    series = prices[ticker].dropna()
    if series.empty:
        return None

    as_of_price = series.index.max()
    as_of = min(pv_as_of, as_of_price)

    # ----- Load transactions for this ticker -----
    tx = transactions[transactions["ticker"] == ticker].copy()
    tx = tx.sort_values("date")

    if tx.empty:
        return None

    first_trade = tx["date"].min()

    # =================================================================
    # SI: Use portfolio inception for alignment with Portfolio P/L
    # This ensures Cash/Recon = Portfolio P/L - Sum(Ticker P/Ls) reconciles
    # =================================================================
    earliest_px = series.index.min()
    
    if h == "SI":
        # FIX: Use portfolio inception date if provided, otherwise fall back to first trade - 1 day
        # This aligns ticker SI P/L with portfolio SI P/L for proper reconciliation
        if portfolio_inception is not None:
            # Use portfolio inception - ensures day 1 gains are NOT double-counted
            raw_start = portfolio_inception
        else:
            # Legacy behavior: Start BEFORE the first trade
            raw_start = first_trade - pd.Timedelta(days=1)
    
    if raw_start is None or raw_start >= as_of:
        return None
        
    series_dates = series.index.sort_values()
    
    # -------------------------------------------------------------
    # DETERMINING START DATE
    # -------------------------------------------------------------
    
    if h == "SI":
         # For SI, we just want a date before the first trade. 
         # If raw_start is before price history, we keep it as is (no price).
         if raw_start < earliest_px:
             start = raw_start
         else:
             # If raw_start is within price history, snap to it
             # (This implies first_trade was later than earliest_px)
             idx = series_dates.searchsorted(raw_start, side="right") - 1
             if idx < 0: start = raw_start
             else: start = series_dates[idx]
             
    elif h == "MTD":
        # raw_start = prior-month-end; use FIRST price *after* that date
        idx = series_dates.searchsorted(raw_start)
        if idx >= len(series_dates):
            return None
        start = series_dates[idx]

    elif h == "1D":
        # 1D should match the portfolio horizon exactly
        idx = series_dates.searchsorted(raw_start)
        if idx >= len(series_dates):
            return None
        start = series_dates[idx]

    else:
        # All other horizons: nearest prior price
        idx = series_dates.searchsorted(raw_start, side="right") - 1
        if idx < 0:
            # Start date is before price history
            # If shares are 0 at this point, it's valid to start here with MV=0
            start = raw_start
        else:
            start = series_dates[idx]

    if start >= as_of:
        return None

    # Horizon must not start before first trade (EXCEPT if we want to capture the first trade flow)
    # If start < first_trade, shares_start=0, net_internal includes first_trade. This is GOOD.
    # So we do NOT want to clamp start = max(start, first_trade) if it means losing the flow.
    # But for display purposes... 
    # Actually, for P/L, start < first_trade is the BEST way to represent "New Position".
    
    # ----- Prices -----
    px_start = 0.0
    try:
        if start in series.index:
            px_start = float(series.loc[start])
        elif start >= earliest_px:
            # Fallback for snap logic
            # Find closest previous
            idx = series_dates.searchsorted(start, side="right") - 1
            if idx >= 0:
                px_start = float(series.iloc[idx])
                
        px_end = float(series.loc[as_of])
    except Exception:
        return None
        
    # ----- Shares at start -----
    # FIX: For SI with portfolio_inception, use raw_start (actual date) not start (snapped to price date)
    shares_boundary = raw_start if (h == "SI" and portfolio_inception is not None) else start
    
    # CRITICAL FIX: For SI at portfolio inception, shares_start MUST be 0
    # (nothing existed BEFORE inception). Use < for SI, <= for other horizons.
    if h == "SI" and portfolio_inception is not None:
        # Strictly BEFORE inception - no shares existed
        mask = tx["date"] < shares_boundary
    else:
        # On or before the horizon start
        mask = tx["date"] <= shares_boundary
    shares_start = tx.loc[mask, "shares"].sum() if mask.any() else 0.0

    # Safety: If we held shares but have no price, we can't calculate P/L
    if abs(shares_start) > 1e-6 and start < earliest_px:
        return None

    # ----- Shares at end -----
    row = sec_only[sec_only["ticker"] == ticker]
    if row.empty:
        return None
    shares_end = float(row["shares"].iloc[0])

    # ----- Internal flows inside window -----
    # FIX: For SI, include flows ON inception day (they're part of the SI window)
    if h == "SI" and portfolio_inception is not None:
        mask2 = (tx["date"] >= shares_boundary) & (tx["date"] <= as_of)
    else:
        mask2 = (tx["date"] > shares_boundary) & (tx["date"] <= as_of)
    # Our file uses amount negative for buys (cash out), positive for sells (cash in).
    # When computing economic P/L, we subtract net internal flows (same as before).
    # Net Capital Flow = -(Buy Amount + Sell Amount)
    net_internal = -tx.loc[mask2, "amount"].sum()

    # ----- Dividends (Income) inside window -----
    total_divs = 0.0
    if dividends is not None and not dividends.empty:
        div_t = dividends[dividends["ticker"] == ticker]
        if not div_t.empty:
            # FIX: For SI, include dividends ON inception day
            if h == "SI" and portfolio_inception is not None:
                mask_div = (div_t["date"] >= shares_boundary) & (div_t["date"] <= as_of)
            else:
                mask_div = (div_t["date"] > shares_boundary) & (div_t["date"] <= as_of)
            total_divs = div_t.loc[mask_div, "amount"].sum()

    # ----- Economic P/L -----
    mv_start = shares_start * px_start
    mv_end = shares_end * px_end

    # Total P/L = (Ending Value - Beginning Value) - (Net Capital Additions) + Income
    # FIX: Using Unadjusted Close implies dividends are NOT accounted for in the price change.
    # We must add total_divs explicitly (Income).
    pl = mv_end - mv_start - net_internal + total_divs

    if return_components:
        return {
            "pl": pl,
            "start": mv_start,
            "end": mv_end,
            "flow": net_internal,
            "inc": total_divs,
            "denom": mv_start + net_internal # Approx
        }

    return pl

def calculate_asset_class_pl(asset_class, h, prices, pv, inception_date, tx_raw, sec_table, dividends, return_components=False):
    """
    Computes DIRECT asset class P/L (not by summing ticker P/Ls).
    Uses Modified Dietz methodology at the aggregate level to properly
    handle positions opened/closed mid-period.
    """
    # Get all tickers in this asset class
    if sec_table.empty:
        if return_components: return {"pl": 0.0, "start": 0.0, "end": 0.0, "flow": 0.0, "inc": 0.0, "denom": 0.0}
        return 0.0
        
    tickers_in_ac = sec_table[sec_table["asset_class"] == asset_class]["ticker"].dropna().unique().tolist()
    
    if not tickers_in_ac or asset_class == "CASH":
        if return_components: return {"pl": 0.0, "start": 0.0, "end": 0.0, "flow": 0.0, "inc": 0.0, "denom": 0.0}
        return 0.0

    # This function now mirrors `calculate_ticker_pl` but aggregates across all tickers in the asset class.
    
    # 1. Determine Horizon Window
    as_of = pv.index.max()
    
    # For SI, the start date needs to be handled carefully to align with how ticker P/L is calculated.
    # The ticker SI P/L uses the *portfolio* inception date for reconciliation purposes.
    # We must do the same here.
    pv_start_date = pv.index.min()

    # Simplified: 'get_portfolio_horizon_start' now handles SI directly
    raw_start = get_portfolio_horizon_start(pv, inception_date, h)

    if raw_start is None or raw_start >= as_of:
        return None

    # 2. Aggregate P/L Components
    mv_start_total = 0.0
    mv_end_total = 0.0
    net_flows_total = 0.0
    divs_total = 0.0
    
    # Iterate over each ticker in the asset class
    for t in tickers_in_ac:
        if t == "CASH": continue
        
        # We need the shares_end value for the ticker
        sec_row = sec_table[sec_table["ticker"] == t]
        if sec_row.empty:
            sec_row = pd.DataFrame([{"ticker": t, "shares": 0.0}])

        # We pass portfolio_inception for the SI case to ensure alignment
        portfolio_inception_for_ticker = pv_start_date if h == "SI" else None
        
        components = calculate_ticker_pl(
            t, h, prices, as_of, tx_raw, sec_row, 
            raw_start=raw_start if h != "SI" else None, # Pass raw_start only for non-SI, SI has its own logic
            dividends=dividends,
            portfolio_inception=portfolio_inception_for_ticker,
            return_components=True
        )
        
        if components and isinstance(components, dict):
            mv_start_total += components.get("start", 0.0)
            mv_end_total += components.get("end", 0.0)
            net_flows_total += components.get("flow", 0.0)
            divs_total += components.get("inc", 0.0)
            
    # 3. Final Calculation
    pl = mv_end_total - mv_start_total - net_flows_total + divs_total
    
    if return_components:
        return {
            "pl": pl,
            "start": mv_start_total,
            "end": mv_end_total,
            "flow": net_flows_total,
            "inc": divs_total,
            "denom": mv_start_total + net_flows_total # Approximation
        }
        
    return pl


def compute_drawdown_series(twr_series):
    """
    Calculates the Drawdown series, Max Drawdown, and Recovery Period
    from a cumulative return series (e.g. Growth of $1).
    
    Returns:
        tuple: (drawdown_series (%), max_drawdown (%), recovery_days (int))
    """
    if twr_series.empty:
        return pd.Series(dtype=float), 0.0, 0
    
    # Calculate High Water Mark
    hwm = twr_series.cummax()
    
    # Calculate Drawdown
    drawdown = (twr_series - hwm) / hwm
    
    # Max Drawdown (Deepest Trough)
    max_dd = drawdown.min()
    
    # Recovery Period (Days)
    # Find the index of the max drawdown
    if max_dd == 0.0:
        return drawdown * 100.0, 0.0, 0
        
    trough_date = drawdown.idxmin()
    
    # Find the next time we touched 0.0 (or exceeded previous HWM)
    # Slice from trough onwards
    future = drawdown[drawdown.index > trough_date]
    recovered = future[future >= 0]
    
    if not recovered.empty:
        recovery_date = recovered.index[0]
        recovery_days = (recovery_date - trough_date).days
    else:
        # Still underwater
        recovery_days = (twr_series.index.max() - trough_date).days
        
    return drawdown * 100.0, max_dd * 100.0, recovery_days
