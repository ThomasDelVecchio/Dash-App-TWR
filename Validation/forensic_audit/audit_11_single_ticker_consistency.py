import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dash_wrappers import run_analytics_engine
from financial_math import HORIZONS

def log_success(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def test_single_ticker_consistency():
    print("\n--- Testing Single-Ticker Asset Class Consistency (GIPS Standards) ---")
    print("Objective: Ensure Asset Class returns match Ticker returns exactly when 1:1.")
    
    # Load Data
    print("Loading engine data...")
    data = run_analytics_engine()
    
    # Extract Tables
    class_df = data.get("class_df")
    sec_table = data.get("sec_table") # Full table (including exited if any)
    
    if class_df is None or class_df.empty:
        log_fail("class_df is empty or missing.")
        
    if sec_table is None or sec_table.empty:
        log_fail("sec_table is empty or missing.")
        
    # Build Map: Asset Class -> List of Tickers
    # Use sec_table to find all tickers associated with each asset class
    ac_map = {}
    
    # We must consider ALL tickers (active + exited) to determine if an Asset Class is truly "Single Ticker"
    # sec_table contains all tickers that have ever existed (if engine is correct).
    
    for _, row in sec_table.iterrows():
        ac = row["asset_class"]
        t = row["ticker"]
        if t == "CASH": continue
        
        if ac not in ac_map:
            ac_map[ac] = set()
        ac_map[ac].add(t)
        
    # Identify Single-Ticker Asset Classes
    single_ticker_classes = {ac: list(tickers)[0] for ac, tickers in ac_map.items() if len(tickers) == 1}
    
    if not single_ticker_classes:
        print("[SKIP] No single-ticker asset classes found to test.")
        return

    print(f"Found {len(single_ticker_classes)} single-ticker asset classes: {single_ticker_classes}")
    
    # Check Consistency
    # Horizons to check: standard list + SI
    check_horizons = HORIZONS + ["SI"]
    
    failures = []
    
    for ac, ticker in single_ticker_classes.items():
        print(f"\nChecking {ac} vs {ticker}...")
        
        # Get Asset Class Row
        ac_row = class_df[class_df["asset_class"] == ac]
        if ac_row.empty:
            failures.append(f"{ac}: Missing from class_df")
            continue
        ac_row = ac_row.iloc[0]
        
        # Get Ticker Row
        t_row = sec_table[sec_table["ticker"] == ticker]
        if t_row.empty:
            failures.append(f"{ticker}: Missing from sec_table")
            continue
        t_row = t_row.iloc[0]
        
        for h in check_horizons:
            # Asset Class Return
            ac_ret = ac_row.get(h)
            # Ticker Return
            t_ret = t_row.get(h)
            
            # Handle NaNs
            ac_nan = pd.isna(ac_ret)
            t_nan = pd.isna(t_ret)
            
            if ac_nan and t_nan:
                # Both NaN is consistent
                continue
            elif ac_nan != t_nan:
                failures.append(f"{ac} [{h}]: Mismatch (AC={ac_ret}, Ticker={t_ret})")
                continue
                
            # Both numeric
            diff = abs(ac_ret - t_ret)
            # Tolerance: 1e-4 (1 basis point is generous, should be near machine epsilon)
            # But let's use 1e-6 to be strict as they should be mathematically identical now
            if diff > 1e-6:
                failures.append(f"{ac} [{h}]: Mismatch (AC={ac_ret:.6f}, Ticker={t_ret:.6f}, Diff={diff:.6f})")
            else:
                # Success
                pass
                
    if failures:
        print("\n" + "\n".join(failures))
        log_fail(f"Found {len(failures)} consistency violations.")
    else:
        log_success("All single-ticker asset classes match their tickers exactly across all horizons.")

if __name__ == "__main__":
    test_single_ticker_consistency()
