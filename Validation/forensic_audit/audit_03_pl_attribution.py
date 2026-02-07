import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dash_wrappers import run_analytics_engine
from portfolio_engine import (
    calculate_horizon_pl,
    calculate_ticker_pl,
    calculate_asset_class_pl
)
from financial_math import get_portfolio_horizon_start

def log_success(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")
    # Don't exit immediately, let all horizons run
    # sys.exit(1) 

def test_pl_attribution():
    print("\n--- Testing P/L Attribution (The Accounting Ledger) ---")
    
    # 1. Load Live Data
    print("Loading live data via Analytics Engine...")
    data = run_analytics_engine()
    print("Data loaded.")
    
    pv = data["pv"]
    inception_date = data["inception_date"]
    cf_ext = data["cf_ext"]
    sec_table = data["sec_table"]
    tx_raw = data["tx_raw"]
    dividends = data["dividends"]
    prices = data["prices"]
    
    if pv.empty:
        print("[SKIP] PV is empty, cannot audit P/L.")
        return

    # Horizons to check
    horizons = ["1W", "1M", "MTD", "YTD", "1Y", "SI"]
    
    # Track failures
    failures = []
    
    for h in horizons:
        # A. Portfolio P/L (Top Down)
        if h == "SI":
            # For SI, use pl_si from engine which is robust
            port_pl = data["pl_si"]
        else:
            port_pl = calculate_horizon_pl(pv, inception_date, cf_ext, h)
            
        if port_pl is None:
            print(f"[{h}] Insufficient data for Portfolio P/L.")
            continue
            
        # B. Sum of Tickers P/L (Bottom Up)
        sum_ticker_pl = 0.0
        
        # We need to iterate ALL tickers that ever existed, not just current.
        # But calculate_ticker_pl needs a row from sec_table to know "shares_end".
        # If ticker is exited, it might be in sec_table with shares=0 if run_engine preserves it.
        # run_engine does: "exited = tx_tickers - current_tickers ... rows_to_add".
        # So sec_table SHOULD contain all tickers.
        
        tickers = sec_table["ticker"].unique()
        
        as_of = pv.index.max()
        # Resolve start date for tickers
        if h == "SI":
            raw_start = None
            pv_start_date = pv.index.min() # For SI alignment
        else:
            raw_start = get_portfolio_horizon_start(pv, inception_date, h)
            pv_start_date = None
            
        ticker_pl_map = {}
        total_divs_in_pl = 0.0
        
        for t in tickers:
            # Need sec_row for shares_end
            sec_row = sec_table[sec_table["ticker"] == t]
            
            # Use return_components to debug if needed
            # skip_gips_gate=True: Attribution must include ALL tickers
            # (including those opened mid-period) to fully reconcile with
            # Portfolio P/L. The GIPS gate is a presentation concern for
            # the display tables, not an attribution concern.
            comps = calculate_ticker_pl(
                t, h, prices, as_of, tx_raw, sec_row, 
                raw_start=raw_start, 
                dividends=dividends,
                portfolio_inception=pv_start_date if h == "SI" else None,
                return_components=True,
                skip_gips_gate=True
            )
            
            val = 0.0
            if isinstance(comps, dict):
                val = comps["pl"]
                total_divs_in_pl += comps["inc"]
            elif comps is not None:
                val = comps
            else:
                # If calculation failed (e.g. missing price), log it so we know why there's a gap
                # Only warn for SI as shorter horizons might legitimately lack data
                if h == "SI":
                    print(f"  [WARN] {t}: P/L calculation returned None (likely missing price data on audit date). Treated as 0.0")
                
            sum_ticker_pl += val
            ticker_pl_map[t] = val
            
        # C. Compare
        # Gap = Portfolio - Sum(Tickers)
        # This gap represents "Unattributed P/L" (Cash Interest, etc).
        # It should be small or explainable.
        gap = port_pl - sum_ticker_pl
        
        # Tolerance: $15.00 (adjusted for live data fluctuations)
        # If Gap is large, it's a failure of attribution.
        
        if abs(gap) < 15.0:
            log_success(f"[{h}] Match! Port: {port_pl:,.2f} | Sum: {sum_ticker_pl:,.2f} | Gap: {gap:.2f}")
        else:
            msg = f"[{h}] Mismatch. Port: {port_pl:,.2f} | Sum: {sum_ticker_pl:,.2f} | Gap: {gap:,.2f}"
            log_fail(msg)
            failures.append(msg)

    # Dividend Integrity Check
    print("\n--- Dividend Integrity Check ---")
    if not dividends.empty:
        # Pick a random dividend
        sample = dividends.iloc[0]
        t = sample["ticker"]
        d = sample["date"]
        amt = sample["amount"]
        
        print(f"Checking Dividend: {t} on {d.date()} for ${amt}")
        
        sec_row = sec_table[sec_table["ticker"] == t]
        comps = calculate_ticker_pl(
            t, "SI", prices, pv.index.max(), tx_raw, sec_row, 
            portfolio_inception=pv.index.min(),
            dividends=dividends,
            return_components=True
        )
        
        if isinstance(comps, dict):
            inc = comps["inc"]
            if inc >= amt:
                log_success(f"Dividend accounted for. Total Income for {t}: ${inc} >= Sample ${amt}")
            else:
                log_fail(f"Dividend MISSING? Total Income for {t}: ${inc} < Sample ${amt}")
        else:
            log_fail("Could not retrieve components for Dividend Check")
    else:
        print("No dividends found to check.")

    if failures:
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_pl_attribution()
        print("\n[SUCCESS] audit_03_pl_attribution.py passed.")
        sys.exit(0)
    except SystemExit as e:
        if e.code != 0:
            print("\n[FAILURE] Audits failed.")
        sys.exit(e.code)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
