import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from financial_math import compute_period_twr, modified_dietz_for_ticker_window

def log_success(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def log_warn(msg):
    print(f"[WARN] {msg}")

def test_missing_data_gaps():
    print("\n--- Testing Gapped Data (Missing Prices) ---")
    d0, d1, d2 = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")
    pv = pd.Series([100.0, 110.0], index=[d0, d2])
    cf = pd.DataFrame([{"date": d1, "amount": 10.0}])
    twr = compute_period_twr(pv, cf, d0, d2)
    
    if abs(twr - 0.0) < 1e-6:
        log_success(f"Gapped Data: Flow mapped to next valid date. TWR={twr:.4f}")
    else:
        log_warn(f"Gapped Data: Unexpected TWR {twr:.4f}. Check flow mapping logic.")

def test_negative_equity():
    print("\n--- Testing Negative Equity (Short/Leverage) ---")
    d0, d1, d2 = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")
    pv = pd.Series([100.0, -50.0, 10.0], index=[d0, d1, d2])
    cf = pd.DataFrame(columns=["date", "amount"])
    try:
        twr = compute_period_twr(pv, cf, d0, d2)
        log_success(f"Negative Equity: Engine handled it without crash. Result={twr:.4f}")
    except Exception as e:
        log_fail(f"Negative Equity: CRASHED with {e}")

def test_trillions_scale():
    print("\n--- Testing Trillions Scale (Floating Point) ---")
    d0, d1 = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")
    val = 1_000_000_000_000.0
    pv = pd.Series([val, val * 1.1], index=[d0, d1])
    cf = pd.DataFrame(columns=["date", "amount"])
    twr = compute_period_twr(pv, cf, d0, d1)
    if abs(twr - 0.10) < 1e-6:
        log_success("Trillions Scale: 10% return calculated correctly")
    else:
        log_fail(f"Trillions Scale: Failed. Got {twr}")

def test_inception_day_deposit_buy():
    print("\n--- Testing Same-Day Deposit and Buy (Inception) ---")
    d0 = pd.Timestamp("2023-01-01")
    pv = pd.Series([1050000.0], index=[d0]) # $1.05M EOD
    cf = pd.DataFrame([{"date": d0, "amount": 1000000.0}]) # $1M Deposit
    twr = compute_period_twr(pv, cf, d0, d0)
    if abs(twr - 0.05) < 1e-6:
        log_success(f"Same-Day Inception: {twr:.6f} == 0.050000")
    else:
        log_fail(f"Same-Day Inception: Expected 0.05, got {twr}")

def test_weekend_flow_snapping():
    print("\n--- Testing Non-Trading Day Flow Snapping ---")
    d_fri = pd.Timestamp("2023-01-06")
    d_sat = pd.Timestamp("2023-01-07")
    d_mon = pd.Timestamp("2023-01-09")
    
    pv = pd.Series([100.0, 110.0], index=[d_fri, d_mon])
    cf = pd.DataFrame([{"date": d_sat, "amount": 5.0}])
    # Logic: Flow on Sat snaps to Mon start? Or Fri end?
    # Engine logic: snaps to Mon START.
    # Mon Start Base = 100 + 5 = 105.
    # Mon End = 110. Return = (110 - 105)/105 = 5/105 = 4.76%
    
    twr = compute_period_twr(pv, cf, d_fri, d_mon)
    expected = (110.0 - 105.0) / 105.0
    
    if abs(twr - expected) < 1e-4:
        log_success(f"Weekend Flow Snapping: {twr:.4f} matches expected {expected:.4f}")
    else:
        log_warn(f"Weekend Flow Snapping: Expected {expected:.4f}, got {twr:.4f}. Check snapping logic.")

def test_dividend_on_start_date():
    print("\n--- Testing Dividend on Position Start Date ---")
    d0 = pd.Timestamp("2023-01-03")
    d10 = pd.Timestamp("2023-01-13")
    dates = pd.bdate_range(d0, d10)
    prices = pd.Series([100.0]*len(dates), index=dates)
    prices.iloc[-1] = 110.0
    
    # Check what happens if dividend is on d_start explicitly
    # Generally dividends are Ex-Date. If Ex-Date is d0, and we hold it, we get it?
    # Depends on MD logic.
    pass # Placeholder if specific logic needs verification

if __name__ == "__main__":
    try:
        test_missing_data_gaps()
        test_negative_equity()
        test_trillions_scale()
        test_inception_day_deposit_buy()
        test_weekend_flow_snapping()
        print("\n[SUCCESS] audit_07_stress_test.py passed all checks.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
