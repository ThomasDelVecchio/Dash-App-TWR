import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from financial_math import compute_horizon_twr, get_portfolio_horizon_start, is_market_holiday

def log_success(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")

def test_holiday_lookback_1w():
    print("\n--- Testing 1W Rolling Horizon (Holiday Gap) ---")
    
    # Scenario:
    # as_of: Jan 17 (Tue)
    # Target 1W: Jan 10 (Tue) -> MISSING (Simulated Holiday)
    # Prior Valid: Jan 09 (Mon) -> Value 100.0
    # Next Valid:  Jan 11 (Wed) -> Value 110.0
    # End Value:   Jan 17 (Tue) -> Value 120.0
    
    dates = [
        pd.Timestamp("2023-01-09"), # Mon
        pd.Timestamp("2023-01-11"), # Wed
        pd.Timestamp("2023-01-17"), # Tue (As Of)
    ]
    
    values = [100.0, 110.0, 120.0]
    
    pv = pd.Series(values, index=dates)
    cf = pd.DataFrame(columns=["date", "amount"]) 
    
    as_of = pd.Timestamp("2023-01-17")
    start_inception = dates[0]
    
    # Expected (Backward Snap to Jan 9): (120 - 100)/100 = 20.0%
    # Buggy (Forward Snap to Jan 11):    (120 - 110)/110 = 9.09%
    
    twr = compute_horizon_twr(pv, cf, start_inception, "1W")
    
    print(f"Computed 1W TWR: {twr:.4f}")
    
    if abs(twr - 0.20) < 1e-6:
        log_success("1W correctly snapped BACKWARD to prior available price.")
    else:
        log_fail(f"1W snapped incorrectly. Expected 0.20, got {twr}")

def test_1d_holiday_snapback():
    print("\n--- Testing 1D Report on Holiday (Good Friday Logic) ---")
    
    # Scenario: 
    # Market Open: Wed Mar 27, Thu Mar 28
    # Market Closed: Fri Mar 29 (Good Friday 2024)
    # 
    # User runs report on Fri Mar 29.
    # 1D Return should act as "Last Trading Session"
    # i.e. Close Thursday vs Close Wednesday.
    
    dates = [
        pd.Timestamp("2024-03-27"), # Wed
        pd.Timestamp("2024-03-28"), # Thu
        pd.Timestamp("2024-03-29"), # Fri (Holiday, but usually prices fill forward)
    ]
    
    # Price moves: 100 -> 101 (1%)
    values = [100.0, 101.0, 101.0] 
    
    pv = pd.Series(values, index=dates)
    
    # Verify Friday is detected as holiday
    friday = pd.Timestamp("2024-03-29")
    if not is_market_holiday(friday):
        log_fail("System failed to identify Good Friday 2024 as a holiday.")
        return

    log_success("Good Friday identified as market holiday.")
    
    # Test get_portfolio_horizon_start
    # as_of = Friday
    # prev_dates < Friday = Thursday
    # If standard 1D -> start = Thursday.
    # Return = Friday(101) / Thursday(101) - 1 = 0.0% (Meaningless)
    # 
    # If Snapback 1D -> start = Wednesday.
    # Return = Thursday(101/Friday(101)) / Wednesday(100) - 1 = 1.0% (Correct)
    
    start_date = get_portfolio_horizon_start(pv, dates[0], "1D")
    
    print(f"As Of: {friday.date()}")
    print(f"Calculated Start Date: {start_date.date()}")
    
    expected_start = pd.Timestamp("2024-03-27") # Wednesday
    
    if start_date == expected_start:
        log_success("1D Start Date snapped back to Wednesday (T-2) correctly.")
    else:
        log_fail(f"1D Start Date incorrect. Expected {expected_start.date()}, got {start_date.date()}")

def test_1d_weekend_snapback():
    print("\n--- Testing 1D Report on Weekend (Saturday Logic) ---")
    
    # Scenario: 
    # Market Open: Thu, Fri
    # Report run: Saturday
    # 1D should be Fri vs Thu.
    
    dates = [
        pd.Timestamp("2024-01-04"), # Thu
        pd.Timestamp("2024-01-05"), # Fri
        pd.Timestamp("2024-01-06"), # Sat (Weekend)
    ]
    
    values = [100.0, 102.0, 102.0]
    
    pv = pd.Series(values, index=dates)
    
    saturday = pd.Timestamp("2024-01-06")
    if not is_market_holiday(saturday): # Helper returns True for weekends too
        log_fail("System failed to identify Saturday as non-trading.")
        return
        
    start_date = get_portfolio_horizon_start(pv, dates[0], "1D")
    
    expected_start = pd.Timestamp("2024-01-04") # Thursday
    
    print(f"As Of: {saturday.date()}")
    print(f"Calculated Start Date: {start_date.date()}")
    
    if start_date == expected_start:
        log_success("1D Start Date snapped back to Thursday correctly for Weekend report.")
    else:
        log_fail(f"1D Start Date incorrect. Expected {expected_start.date()}, got {start_date.date()}")

if __name__ == "__main__":
    test_holiday_lookback_1w()
    test_1d_holiday_snapback()
    test_1d_weekend_snapback()
