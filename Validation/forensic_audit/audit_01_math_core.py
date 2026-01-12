import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from financial_math import (
    compute_period_twr,
    modified_dietz_for_ticker_window,
    modified_dietz_for_asset_class_window
)
# Import engine functions for P/L math
from portfolio_engine import (
    calculate_horizon_pl,
    calculate_ticker_pl
)

def log_success(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def log_warn(msg):
    print(f"[WARN] {msg}")

def test_twr_synthetic():
    print("\n--- Testing TWR Core Logic ---")
    
    # Dates
    d0 = pd.Timestamp("2023-01-01")
    d1 = pd.Timestamp("2023-01-02")
    d5 = pd.Timestamp("2023-01-05")
    d10 = pd.Timestamp("2023-01-10")

    # ---------------------------------------------------------
    # Test Case A: Standard Scenario
    cf = pd.DataFrame([{"date": d5, "amount": 50.0}])
    pv_vals = []
    for d in pd.date_range(d0, d10, freq="D"):
        if d < d5: pv_vals.append(100.0)
        elif d == d5: pv_vals.append(165.0) # (100+50)*1.10
        else: pv_vals.append(165.0)
    pv = pd.Series(pv_vals, index=pd.date_range(d0, d10, freq="D"))
    
    twr = compute_period_twr(pv, cf, d0, d10)
    if abs(twr - 0.10) < 1e-6:
        log_success(f"Standard TWR: {twr:.6f} == 0.100000")
    else:
        log_fail(f"Standard TWR: Expected 0.10, got {twr}")

    # ---------------------------------------------------------
    # Test Case B: Day 1 Funding
    twr_b = compute_period_twr(pd.Series([105.0], index=[d0]), pd.DataFrame([{"date": d0, "amount": 100.0}]), d0, d0)
    if abs(twr_b - 0.05) < 1e-6:
        log_success(f"Day 1 Funding TWR: {twr_b:.6f} == 0.050000")
    else:
        log_fail(f"Day 1 Funding TWR: Expected 0.05, got {twr_b}")
        
    # ---------------------------------------------------------
    # Test Case C: Large Withdrawal
    twr_c = compute_period_twr(pd.Series([100.0, 11.0], index=[d0, d1]), pd.DataFrame([{"date": d1, "amount": -90.0}]), d0, d1)
    if abs(twr_c - 0.10) < 1e-6:
        log_success(f"Large Withdrawal TWR: {twr_c:.6f} == 0.100000")
    else:
        log_fail(f"Large Withdrawal TWR: Expected 0.10, got {twr_c}")
    
    # ---------------------------------------------------------
    # Test Case D: Extreme Volatility
    twr_d = compute_period_twr(pd.Series([100.0, 1100.0], index=[d0, d1]), pd.DataFrame(columns=["date", "amount"]), d0, d1)
    if abs(twr_d - 10.0) < 1e-6:
        log_success(f"Extreme Volatility: {twr_d:.6f} == 10.000000")
    else:
        log_fail(f"Extreme Volatility: Expected 10.0, got {twr_d}")

    # ---------------------------------------------------------
    # Test Case E: Zero Value Recovery
    d2, d3 = pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04")
    pv_e = pd.Series([100.0, 0.0, 100.0, 110.0], index=[d0, d1, d2, d3])
    cf_e = pd.DataFrame([{"date": d2, "amount": 100.0}])
    twr_e = compute_period_twr(pv_e, cf_e, d0, d3)
    if abs(twr_e - (-1.0)) < 1e-6:
        log_success(f"Zero Value Recovery: {twr_e:.6f} == -1.000000")
    else:
        log_fail(f"Zero Value Recovery: Expected -1.0, got {twr_e}")

    # ---------------------------------------------------------
    # Test Case F: Leap Year
    leap_d0, leap_d1, leap_d2 = pd.Timestamp("2024-02-28"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-01")
    twr_f = compute_period_twr(pd.Series([100.0, 200.0, 220.0], index=[leap_d0, leap_d1, leap_d2]), pd.DataFrame([{"date": leap_d1, "amount": 100.0}]), leap_d0, leap_d2)
    if abs(twr_f - 0.10) < 1e-6:
        log_success(f"Leap Year: {twr_f:.6f} == 0.100000")
    else:
        log_fail(f"Leap Year: Expected 0.10, got {twr_f}")

    # ---------------------------------------------------------
    # Test Case G: Geometric Chaining Invariant
    # TWR(A->C) should equal (1+TWR(A->B)) * (1+TWR(B->C)) - 1
    print("  Checking Geometric Chaining...")
    pv_chain = pd.Series([100.0, 105.0, 110.25], index=[d0, d5, d10])
    cf_chain = pd.DataFrame(columns=["date", "amount"])
    
    twr_full = compute_period_twr(pv_chain, cf_chain, d0, d10)
    twr_p1 = compute_period_twr(pv_chain, cf_chain, d0, d5)
    twr_p2 = compute_period_twr(pv_chain, cf_chain, d5, d10)
    
    chained = (1 + twr_p1) * (1 + twr_p2) - 1
    if abs(twr_full - chained) < 1e-10:
        log_success(f"Geometric Chaining: {twr_full:.6f} == {chained:.6f}")
    else:
        log_fail(f"Geometric Chaining Failed: Full={twr_full}, Chained={chained}")


def test_md_synthetic():
    print("\n--- Testing Modified Dietz Logic ---")
    
    # Use trading days (June 2023) to avoid weekend/holiday gaps
    d_start = pd.Timestamp("2023-06-01") # Thursday
    d_flow = pd.Timestamp("2023-06-12")  # Monday
    d_end = pd.Timestamp("2023-06-20")   # Tuesday
    
    prices = pd.Series([10.0, 12.0, 15.0], index=[d_start, d_flow, d_end])
    tx_df = pd.DataFrame([
        {"date": d_start, "ticker": "TEST", "shares": 10.0, "amount": -100.0},
        {"date": d_flow,  "ticker": "TEST", "shares": 5.0,  "amount": -60.0}
    ])
    
    # 1. Basic MD
    ret = modified_dietz_for_ticker_window("TEST", prices, tx_df, d_start, d_end)
    # Weight calc: Total days (June 1-20 inclusive) = 20.
    # Flow on June 12: Active days (12-20 inclusive) = 9.
    # Weight = 9/20 = 0.45
    expected = (225.0 - 160.0) / (100.0 + 0.45 * 60.0)
    if abs(ret - expected) < 1e-6:
        log_success(f"Modified Dietz: {ret:.6f} == {expected:.6f}")
    else:
        log_fail(f"Modified Dietz: Expected {expected:.6f}, got {ret:.6f}")

    # 2. Component Identity
    print("  Checking MD Component Identity...")
    res = modified_dietz_for_ticker_window("TEST", prices, tx_df, d_start, d_end, return_components=True)
    if isinstance(res, dict):
        V1, V0, C = res["end_val"], res["start_val"], res["net_flow"]
        wf = res["weighted_flow"]
        calc_gain = V1 - V0 - C
        # Note: md_gain calculation here logic check: return = gain / (V0 + wf) -> gain = return * (V0 + wf)
        md_gain_val = res["return"] * (V0 + wf)
        if abs(calc_gain - md_gain_val) < 1e-2:
            log_success(f"Component Identity: Gain {calc_gain:.2f} matches derived {md_gain_val:.2f}")
        else:
            log_fail(f"Component Identity Mismatch: {calc_gain} vs {md_gain_val}")
    else:
        log_warn("MD did not return components dictionary.")

    # 3. Dividend Check
    print("  Checking MD Dividend Capture...")
    divs = pd.DataFrame([{"date": d_flow, "ticker": "TEST", "amount": 10.0}])
    res_div = modified_dietz_for_ticker_window("TEST", prices, tx_df, d_start, d_end, dividends=divs, return_components=True)
    if isinstance(res_div, dict) and abs(res_div.get("income", 0) - 10.0) < 1e-6:
         log_success("Dividend correctly captured in MD components.")
    else:
        log_fail(f"Dividend verification failed. Got {res_div.get('income', 0)}")


def test_asset_class_md_synthetic():
    print("\n--- Testing Asset Class Modified Dietz Logic ---")
    # Use trading days (June 2023) to avoid weekend/holiday gaps
    d_start, d_flow, d_end = pd.Timestamp("2023-06-01"), pd.Timestamp("2023-06-12"), pd.Timestamp("2023-06-20")
    
    prices = pd.DataFrame({
        "T1": pd.Series([10.0, 10.0, 12.0], index=[d_start, d_flow, d_end]),
        "T2": pd.Series([10.0, 10.0, 12.0], index=[d_start, d_flow, d_end])
    })
    tx_df = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "ticker": "T1", "shares": 10.0, "amount": -100.0},
        {"date": d_flow, "ticker": "T2", "shares": 5.0, "amount": -50.0}
    ])
    
    ret = modified_dietz_for_asset_class_window(["T1", "T2"], prices, tx_df, d_start, d_end)
    # Gain: V1(120+60)-V0(100)-NetInv(50) = 30.
    # Denom: V0(100) + Flow(50)*0.45 = 122.5
    expected = 30.0 / 122.5
    if abs(ret - expected) < 1e-6:
        log_success(f"Asset Class MD: {ret:.6f} == {expected:.6f}")
    else:
        log_fail(f"Asset Class MD: Expected {expected:.6f}, got {ret:.6f}")

def test_pl_math_synthetic():
    print("\n--- Testing P/L Math (Sum of Parts) ---")
    # Use trading days: June 1 (Thu) to June 2 (Fri)
    d_start, d_end = pd.Timestamp("2023-06-01"), pd.Timestamp("2023-06-02")
    pv = pd.Series([100.0, 110.0], index=[d_start, d_end])
    # Empty flows for PL calc
    port_pl = calculate_horizon_pl(pv, d_start, pd.DataFrame(columns=["date", "amount"]), "1D")
    
    prices = pd.DataFrame({
        "T1": pd.Series([10.0, 11.0], index=[d_start, d_end]),
        "T2": pd.Series([10.0, 11.0], index=[d_start, d_end])
    })
    tx_df = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-01"), "ticker": "T1", "shares": 5.0, "amount": -50.0},
        {"date": pd.Timestamp("2020-01-01"), "ticker": "T2", "shares": 5.0, "amount": -50.0}
    ])
    sec_table = pd.DataFrame([{"ticker": "T1", "shares": 5.0}, {"ticker": "T2", "shares": 5.0}])
    
    t1_pl = calculate_ticker_pl("T1", "1D", prices, d_end, tx_df, sec_table, raw_start=d_start)
    t2_pl = calculate_ticker_pl("T2", "1D", prices, d_end, tx_df, sec_table, raw_start=d_start)
    
    # Handle possible None return
    t1_pl = t1_pl if t1_pl else 0.0
    t2_pl = t2_pl if t2_pl else 0.0
    port_pl = port_pl if port_pl else 0.0

    if abs(port_pl - (t1_pl + t2_pl)) < 1e-6:
        log_success(f"P/L Math: Portfolio {port_pl} == Sum Tickers {(t1_pl + t2_pl)}")
    else:
        log_fail(f"P/L Math: Portfolio {port_pl} != Sum Tickers {(t1_pl + t2_pl)}")

if __name__ == "__main__":
    test_twr_synthetic()
    test_md_synthetic()
    test_asset_class_md_synthetic()
    test_pl_math_synthetic()
    print("\n[SUCCESS] audit_01_math_core.py passed.")