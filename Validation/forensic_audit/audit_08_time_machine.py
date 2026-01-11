import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from portfolio_engine import run_engine
    from data_loader import load_transactions_raw
except ImportError:
    # Fallback
    sys.path.append(os.getcwd())
    from portfolio_engine import run_engine
    from data_loader import load_transactions_raw

def log_success(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def log_warn(msg):
    print(f"[WARN] {msg}")

def log_skip(msg):
    print(f"[SKIP] {msg}")

def test_time_machine():
    print("\n--- Testing Time Machine Consistency ---")
    
    # 1. Run Live Engine (Current)
    print("Running Live Engine...")
    try:
        # Unpack based on known signature: twr, sec, class, pv, ..., ..., pl
        # We'll rely on index or key access if it returned a dict, but run_engine returns tuple usually.
        # Assuming tuple: (twr_df, sec_table, class_df, pv, cf, inception, pl_si)
        # If the return signature changes, this needs update.
        res_now = run_engine()
        pv_now = res_now[3] 
        sec_now = res_now[1]
    except Exception as e:
        log_fail(f"Live Engine Crash: {e}")

    if pv_now.empty:
        log_skip("PV is empty (No data).")
        return

    # 2. Pick a Historical Date (e.g., T-14)
    max_date = pv_now.index.max()
    target_date = max_date - timedelta(days=14)
    
    # Ensure target_date is within data range
    if target_date < pv_now.index.min():
        log_skip("History too short for T-14 test.")
        return
        
    # Snap to nearest valid PV date <= target_date
    valid_dates = pv_now.index[pv_now.index <= target_date]
    if valid_dates.empty:
        log_skip("No valid dates found in history window.")
        return
    hist_date = valid_dates[-1] # Take the max of valid dates
    
    print(f"Time Machine Target: {hist_date.date()}")

    # 3. Run Historical Engine
    print("Running Historical Engine...")
    try:
        res_hist = run_engine(end_date=hist_date)
        pv_hist = res_hist[3]
        sec_hist = res_hist[1]
    except Exception as e:
        log_fail(f"Historical Engine Crash: {e}")

    # 4. PV Consistency Check
    print("\nChecking PV Consistency...")
    # Matches historical run's LAST value with Live run's value AT that date
    val_hist_final = pv_hist.iloc[-1]
    
    # Live data at that date
    if hist_date in pv_now.index:
        val_now_at_hist = pv_now.loc[hist_date]
    else:
        # Should not happen if we picked hist_date FROM pv_now, but safest to use asof
        val_now_at_hist = pv_now.asof(hist_date)
    
    diff = abs(val_hist_final - val_now_at_hist)
    # Tolerance $0.05
    if diff < 0.05:
        log_success(f"PV Matches: History={val_hist_final:.2f} vs Live={val_now_at_hist:.2f}")
    else:
        log_fail(f"PV Mismatch: History={val_hist_final:.2f} vs Live={val_now_at_hist:.2f} (Diff={diff:.2f})")

    # 5. Future Leakage Check
    print("\nChecking Future Leakage (Transactions)...")
    # sec_hist should NOT contain tickers that were only bought AFTER hist_date
    tx_raw = load_transactions_raw()
    
    # Find tickers that first appear AFTER hist_date
    first_dates = tx_raw.groupby("ticker")["date"].min()
    future_tickers = first_dates[first_dates > hist_date].index.tolist()
    
    leaked = []
    current_hist_tickers = sec_hist["ticker"].unique()
    for t in future_tickers:
        if t in current_hist_tickers:
            # Check if quantity is non-zero (maybe it exists but 0 shares?)
            # Even 0 shares might imply leakage if the ticker row shouldn't exist yet.
            # But let's check shares > 0
            shares = sec_hist.loc[sec_hist["ticker"] == t, "shares"].sum()
            if abs(shares) > 1e-6:
                leaked.append(t)
    
    if not leaked:
        log_success("No future tickers detected in historical view.")
    else:
        log_fail(f"Future tickers leaked into history: {leaked}")

if __name__ == "__main__":
    test_time_machine()
    print("\n[SUCCESS] audit_08_time_machine.py passed.")
