import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Fix path to point 2 levels up to Root (where financial_math.py lives)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from financial_math import (
    compute_cash_yield,
    annualize_return,
    compute_period_twr
)

def log_success(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def test_cash_yield_simple():
    print("\n--- Testing Cash Yield (Simple) ---")
    d_start = pd.Timestamp("2023-01-01")
    d_end = pd.Timestamp("2023-01-30")
    dates = pd.date_range(d_start, d_end)
    
    # Constant Balance: 10,000
    cash_trace = pd.Series([10000.0] * len(dates), index=dates)
    
    # Interest Payment: $40
    interest_df = pd.DataFrame([
        {"date": pd.Timestamp("2023-01-15"), "amount": 40.0}
    ])
    
    yield_val = compute_cash_yield(cash_trace, interest_df, d_start, d_end)
    expected = 40.0 / 10000.0
    
    if abs(yield_val - expected) < 1e-6:
        log_success(f"Simple Yield: {yield_val:.6f} == {expected:.6f}")
    else:
        log_fail(f"Simple Yield: Expected {expected:.6f}, got {yield_val:.6f}")

def test_cash_yield_variable_balance():
    print("\n--- Testing Cash Yield (Variable Balance) ---")
    d0 = pd.Timestamp("2023-01-01")
    d9 = pd.Timestamp("2023-01-10")
    dates = pd.date_range(d0, d9)
    
    # Days 1-5: 1000. Days 6-10: 2000. Avg = 1500.
    vals = [1000.0]*5 + [2000.0]*5
    cash_trace = pd.Series(vals, index=dates)
    
    # Interest: $15 total
    interest_df = pd.DataFrame([
        {"date": pd.Timestamp("2023-01-05"), "amount": 5.0},
        {"date": pd.Timestamp("2023-01-10"), "amount": 10.0}
    ])
    
    expected_yield = 15.0 / 1500.0 # 1%
    yield_val = compute_cash_yield(cash_trace, interest_df, d0, d9)
    
    if abs(yield_val - expected_yield) < 1e-6:
        log_success(f"Variable Balance Yield: {yield_val:.6f} == {expected_yield:.6f}")
    else:
        log_fail(f"Variable Balance Yield: Expected {expected_yield:.6f}, got {yield_val:.6f}")

def test_cash_twr_interaction():
    print("\n--- Testing GIPS Compliance (TWR vs Interest) ---")
    # Verify interest is NOT treated as external flow
    d0 = pd.Timestamp("2023-01-01")
    d10 = pd.Timestamp("2023-01-11")
    dates = pd.date_range(d0, d10)
    
    vals = [1000.0] * 10 + [1010.0]
    pv = pd.Series(vals, index=dates)
    
    # External Flows: ONLY initial funding. Interest (+10) is excluded here.
    cf_ext = pd.DataFrame([{"date": d0, "amount": 1000.0}])
    
    twr = compute_period_twr(pv, cf_ext, d0, d10)
    expected = 0.01 # (1010 - 1000) / 1000
    
    if abs(twr - expected) < 1e-6:
        log_success(f"Cash TWR (Interest as Growth): {twr:.6f} == {expected:.6f}")
    else:
        log_fail(f"Cash TWR Failed: Expected {expected}, got {twr}")

if __name__ == "__main__":
    test_cash_yield_simple()
    test_cash_yield_variable_balance()
    test_cash_twr_interaction()
    print("\n[SUCCESS] audit_10_cash.py passed.")