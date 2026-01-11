import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path to import local modules
sys.path.append(os.getcwd())

from financial_math import annualize_return, compute_period_twr, modified_dietz_for_ticker_window

def test_annualization_logic():
    print("=======================================================")
    print("AUDIT: UNIVERSAL ANNUALIZATION LOGIC GATE (365.25 DAYS)")
    print("=======================================================")
    start = pd.Timestamp("2020-01-01")
    
    # Case 1: 180 Days (Cumulative)
    # Expected: No Annualization
    end_180 = start + timedelta(days=180)
    ret_180 = 0.10 # 10%
    res_180 = annualize_return(ret_180, start, end_180)
    print(f"SCENARIO 1: 180 Days (<= 1.0 Year)")
    print(f"  Input Return: 10.0000%")
    print(f"  Output:       {res_180:.4%}")
    print(f"  Expected:     10.0000% (Cumulative)")
    print(f"  Result:       {'PASS' if abs(res_180 - 0.10) < 1e-6 else 'FAIL'}")
    print("-" * 40)
    
    # Case 2: 365 Days (Cumulative)
    # 365 / 365.25 = 0.9993 < 1.0 -> Cumulative
    end_365 = start + timedelta(days=365)
    ret_365 = 0.10
    res_365 = annualize_return(ret_365, start, end_365)
    print(f"SCENARIO 2: 365 Days (Exactly 365)")
    print(f"  Input Return: 10.0000%")
    print(f"  Output:       {res_365:.4%}")
    print(f"  Expected:     10.0000% (Cumulative)")
    print(f"  Result:       {'PASS' if abs(res_365 - 0.10) < 1e-6 else 'FAIL'}")
    print("-" * 40)
    
    # Case 3: 366 Days (Annualized)
    # 366 / 365.25 = 1.0020 > 1.0 -> Annualized
    end_366 = start + timedelta(days=366)
    ret_366 = 0.10
    res_366 = annualize_return(ret_366, start, end_366)
    
    years_366 = 366 / 365.25
    expected_366 = (1.10) ** (1/years_366) - 1
    
    print(f"SCENARIO 3: 366 Days (Leap Year Edge Case)")
    print(f"  Input Return: 10.0000%")
    print(f"  Output:       {res_366:.4%}")
    print(f"  Expected:     {expected_366:.4%} (Annualized)")
    print(f"  Result:       {'PASS' if abs(res_366 - expected_366) < 1e-6 else 'FAIL'}")
    print("-" * 40)
    
    # Case 4: 3 Years (Annualized)
    # 3 * 365 = 1095 days (approx).
    end_3y = start + timedelta(days=1096) # 3 years including leap day
    ret_3y = 0.331 # 33.1% cumulative (approx 10% ann)
    res_3y = annualize_return(ret_3y, start, end_3y)
    
    days_3y = (end_3y - start).days
    years_3y = days_3y / 365.25
    expected_3y = (1 + ret_3y) ** (1/years_3y) - 1
    
    print(f"SCENARIO 4: 3 Years ({days_3y} Days)")
    print(f"  Input Return: {ret_3y:.2%}")
    print(f"  Output:       {res_3y:.4%}")
    print(f"  Expected:     {expected_3y:.4%} (Annualized)")
    print(f"  Result:       {'PASS' if abs(res_3y - expected_3y) < 1e-6 else 'FAIL'}")
    print("=" * 55)

def test_integration_portfolio():
    print("\n=======================================================")
    print("AUDIT: PORTFOLIO ENGINE INTEGRATION TEST")
    print("=======================================================")
    
    # Mock Data
    start_date = pd.Timestamp("2020-01-01")
    cf = pd.DataFrame(columns=["date", "amount"]) # No flows
    
    # 365 Day Scenario
    end_365 = start_date + timedelta(days=365)
    pv_365 = pd.Series([100.0, 110.0], index=[start_date, end_365])
    # TWR of 10%. Duration 365 days.
    
    # Simulate Engine Call
    twr_cum = compute_period_twr(pv_365, cf, start_date, end_365)
    twr_final = annualize_return(twr_cum, start_date, end_365)
    
    print(f"PORTFOLIO ENGINE: 365 Days")
    print(f"  Cumulative TWR: {twr_cum:.4%}")
    print(f"  Final Output:   {twr_final:.4%}")
    print(f"  Result:         {'PASS' if abs(twr_final - 0.10) < 1e-6 else 'FAIL'}")
    print("-" * 40)
    
    # 366 Day Scenario
    end_366 = start_date + timedelta(days=366)
    pv_366 = pd.Series([100.0, 110.0], index=[start_date, end_366])
    twr_cum_366 = compute_period_twr(pv_366, cf, start_date, end_366)
    twr_final_366 = annualize_return(twr_cum_366, start_date, end_366)
    
    years_366 = 366 / 365.25
    expected_366 = (1.10) ** (1/years_366) - 1
    
    print(f"PORTFOLIO ENGINE: 366 Days")
    print(f"  Cumulative TWR: {twr_cum_366:.4%}")
    print(f"  Final Output:   {twr_final_366:.4%}")
    print(f"  Result:         {'PASS' if abs(twr_final_366 - expected_366) < 1e-6 else 'FAIL'}")
    print("=" * 55)

if __name__ == "__main__":
    test_annualization_logic()
    test_integration_portfolio()
