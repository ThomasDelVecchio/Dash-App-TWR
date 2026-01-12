import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import dash_wrappers as dw

def log_success(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")

def test_wrapper_holiday_snap():
    print("\n--- Testing Dash Wrapper Horizon Analysis (Holiday Snap) ---")
    
    # Mock Data Setup representing Good Friday 2024
    dates = [
        pd.Timestamp("2024-03-27"), # Wed
        pd.Timestamp("2024-03-28"), # Thu
        pd.Timestamp("2024-04-01"), # Mon (Fri 29th was holiday)
    ]
    
    values = [100.0, 101.0, 102.0]
    pv = pd.Series(values, index=dates)
    
    # Mock minimal Data Dict
    # dash_wrappers.get_horizon_analysis needs:
    # pv, inception_date, cf_ext, twr_df, pl_si, twr_si, twr_si_ann
    
    mock_data = {
        "pv": pv,
        "inception_date": dates[0],
        "cf_ext": pd.DataFrame(columns=["date", "amount"]),
        "twr_df": pd.DataFrame({"Horizon": ["1D"], "Return": [0.0]}), # Dummy, will be recalculated/ignored for PL
        "pl_si": 2.0,
        "twr_si": 0.02,
        "twr_si_ann": 0.02,
    }
    
    # Run Function to Test
    results_df = dw.get_horizon_analysis(mock_data)
    
    # Analyze '1D' Row
    # If run on Mon Apr 01, 1D should look back to Thu Mar 28
    # If it failed snap, it might think 1D is Fri (NaN) or something else.
    
    row_1d = results_df[results_df["Horizon"] == "1D"]
    
    if row_1d.empty:
        log_fail("1D Horizon missing from results.")
        return

    # Check P/L
    # Mon(102) - Thu(101) = 1.0
    pl_val = float(row_1d["P/L"].iloc[0])
    
    print(f"1D P/L: {pl_val} (Expected 1.0)")
    
    if abs(pl_val - 1.0) < 1e-6:
        log_success("1D P/L correctly calculated using mocked holiday data.")
    else:
        log_fail(f"1D P/L incorrect. Got {pl_val}, expected 1.0")
        
    # Check Meta Start Date
    # verify it passed the correct start date into the meta columns
    start_date = row_1d["meta_P/L_start_date"].iloc[0]
    expected_start = pd.Timestamp("2024-03-28")
    
    print(f"1D Start Date: {start_date.date()} (Expected 2024-03-28)")
    
    if start_date == expected_start:
        log_success("1D Start Date correctly snapped to prior trading day (Thursday).")
    else:
        log_fail(f"1D Start Date incorrect. Got {start_date}, expected {expected_start}")

if __name__ == "__main__":
    test_wrapper_holiday_snap()
