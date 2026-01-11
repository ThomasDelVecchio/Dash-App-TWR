import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from dash_wrappers import run_analytics_engine, _get_daily_twr_curve
    from financial_math import compute_period_twr
except ImportError:
    sys.path.append(os.getcwd())
    from dash_wrappers import run_analytics_engine, _get_daily_twr_curve
    from financial_math import compute_period_twr

def log_success(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def log_skip(msg):
    print(f"[SKIP] {msg}")

def test_custom_report_logic():
    print("\n--- Testing Custom Report Logic (Curve Slicing) ---")
    
    # 1. Load Data
    print("Loading Live Data...")
    try:
        data = run_analytics_engine()
    except Exception as e:
        log_fail(f"Engine Load Crash: {e}")
        
    pv = data.get("pv")
    if pv is None or pv.empty:
        log_skip("PV empty.")
        return

    # 2. Get/Gen Cumulative Curve
    # The Custom Report typically uses a normalized TWR curve (Day 0 = 100 or 1.0)
    # _get_daily_twr_curve returns a Series of cumulative returns (1.0, 1.05, ...)
    twr_curve = _get_daily_twr_curve(data)
    
    if twr_curve.empty:
        log_skip("TWR Curve empty.")
        return
        
    # 3. Define Random Windows
    # Test Window 1: Last 30 Days (Standard)
    # Test Window 2: Random middle chunk
    
    max_date = twr_curve.index.max()
    min_date = twr_curve.index.min()
    
    test_windows = []
    
    # Window A: Recent Month
    start_a = max_date - timedelta(days=30)
    if start_a > min_date:
        test_windows.append(("Last 30 Days", start_a, max_date))
        
    # Window B: Random Middle (if enough history)
    mid_point = min_date + (max_date - min_date) / 2
    start_b = mid_point
    end_b = mid_point + timedelta(days=20)
    if end_b < max_date:
         test_windows.append(("Middle 20 Days", start_b, end_b))
         
    if not test_windows:
        log_skip("Not enough history for window tests.")
        return
        
    for name, s, e in test_windows:
        print(f"\nScanning {name} ({s.date()} to {e.date()})...")
        
        # Method 1: Curve Slicing (Custom Report Quick Method)
        # Logic: (Cumulative_End / Cumulative_Start) - 1
        # Need to allow for 's' falling between curve points? 
        # _get_daily_twr_curve is daily, so asof should work.
        
        val_start = twr_curve.asof(s)
        val_end = twr_curve.asof(e)
        
        if pd.isna(val_start) or pd.isna(val_end) or val_start == 0:
            print(f"  [SKIP] Invalid curve values: Start={val_start}, End={val_end}")
            continue
            
        twr_slice = (val_end / val_start) - 1
        
        # Method 2: Rigorous Recalculation (Source of Truth)
        # Using compute_period_twr on the raw PV and Flows for that exact window
        # IMPORTANT: compute_period_twr filters for pv.index >= start. (Forward Snap).
        # But Curve Slice .asof(start) uses Backward Snap (Last valid price).
        # We must snap 's' backward to the last valid trading day to ensure
        # the Rigorous method includes the return of the first trading day in the window.
        
        valid_dates_before = pv.index[pv.index <= s]
        if valid_dates_before.empty:
            print(f"  [SKIP] No valid dates before start {s.date()}")
            continue
            
        effective_start = valid_dates_before[-1]
        
        twr_rigorous = compute_period_twr(
            data["pv"], 
            data["cf_ext"], 
            effective_start, 
            e
        )
        
        # Comparison
        diff = abs(twr_slice - twr_rigorous)
        
        # Tolerance: 1 basis point (1e-4)
        if diff < 1e-4:
            log_success(f"{name}: Match! Slice={twr_slice:.6f} vs Rigorous={twr_rigorous:.6f}")
        else:
            log_fail(f"{name}: Mismatch! Slice={twr_slice:.6f} vs Rigorous={twr_rigorous:.6f} (Diff={diff:.6f})")

if __name__ == "__main__":
    test_custom_report_logic()
    print("\n[SUCCESS] audit_09_custom_report.py passed.")
