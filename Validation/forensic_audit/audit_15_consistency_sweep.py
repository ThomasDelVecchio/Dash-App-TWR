
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from financial_math import (
    get_horizon_target_date, 
    get_portfolio_horizon_start,
    get_effective_anchor_date,
    is_market_holiday
)

def log(msg, status="INFO"):
    print(f"[{status}] {msg}")

def test_effective_anchor_logic():
    log("Testing Effective Anchor Logic (Weekend/Holiday Snap-back)...")
    
    # Case 1: Friday (Trading Day) -> Should return itself
    fri = pd.Timestamp("2023-10-27") 
    anchor = get_effective_anchor_date(fri)
    assert anchor == fri, f"Friday {fri} should anchor to itself, got {anchor}"
    log(f"  PASS: Friday {fri} -> {anchor}")
    
    # Case 2: Saturday -> Should snap to Friday
    sat = pd.Timestamp("2023-10-28")
    anchor = get_effective_anchor_date(sat)
    assert anchor == fri, f"Saturday {sat} should anchor to Friday {fri}, got {anchor}"
    log(f"  PASS: Saturday {sat} -> {anchor}")
    
    # Case 3: Sunday -> Should snap to Friday
    sun = pd.Timestamp("2023-10-29")
    anchor = get_effective_anchor_date(sun)
    assert anchor == fri, f"Sunday {sun} should anchor to Friday {fri}, got {anchor}"
    log(f"  PASS: Sunday {sun} -> {anchor}")
    
    # Case 4: NYSE Holiday (e.g. Christmas 2023 - Mon Dec 25)
    # Fri Dec 22 is open. Sat/Sun closed. Mon closed.
    # So Dec 25 should snap to Dec 22.
    xmas = pd.Timestamp("2023-12-25")
    fri_before = pd.Timestamp("2023-12-22")
    anchor = get_effective_anchor_date(xmas)
    assert anchor == fri_before, f"Holiday {xmas} should anchor to {fri_before}, got {anchor}"
    log(f"  PASS: Holiday {xmas} -> {anchor}")

    log("Effective Anchor Logic Verified.\n")

def test_horizon_target_dates():
    log("Testing Horizon Target Date Calculation...")
    
    # Base: Friday Oct 27, 2023
    as_of = pd.Timestamp("2023-10-27")
    
    # 1D: Should be 1 day prior to anchor (Thu Oct 26)
    t_1d = get_horizon_target_date(as_of, "1D")
    expected_1d = pd.Timestamp("2023-10-26")
    assert t_1d == expected_1d, f"1D Target for {as_of} should be {expected_1d}, got {t_1d}"
    
    # 1W: Should be 7 days prior to anchor (Fri Oct 20)
    t_1w = get_horizon_target_date(as_of, "1W")
    expected_1w = pd.Timestamp("2023-10-20")
    assert t_1w == expected_1w, f"1W Target for {as_of} should be {expected_1w}, got {t_1w}"
    
    # 1M: One calendar month prior (Sep 27)
    t_1m = get_horizon_target_date(as_of, "1M")
    expected_1m = pd.Timestamp("2023-09-27")
    assert t_1m == expected_1m, f"1M Target for {as_of} should be {expected_1m}, got {t_1m}"
    
    # MTD: Last day of prior month (Sep 30)
    t_mtd = get_horizon_target_date(as_of, "MTD")
    expected_mtd = pd.Timestamp("2023-09-30")
    assert t_mtd == expected_mtd, f"MTD Target for {as_of} should be {expected_mtd}, got {t_mtd}"
    
    # YTD: Last day of prior year (Dec 31, 2022)
    t_ytd = get_horizon_target_date(as_of, "YTD")
    expected_ytd = pd.Timestamp("2022-12-31")
    assert t_ytd == expected_ytd, f"YTD Target for {as_of} should be {expected_ytd}, got {t_ytd}"

    log("Standard Horizon Targets Verified.")
    
    # --- Weekend Edge Case ---
    # As Of: Saturday Oct 28. Anchor -> Fri Oct 27.
    # 1W Target should be Fri Oct 20 (7 days before anchor).
    sat = pd.Timestamp("2023-10-28")
    t_1w_sat = get_horizon_target_date(sat, "1W")
    expected_1w_sat = pd.Timestamp("2023-10-20")
    assert t_1w_sat == expected_1w_sat, f"1W Target for Sat {sat} should be {expected_1w_sat}, got {t_1w_sat}"
    log("Weekend As-Of Anchor Logic Verified.\n")

def test_portfolio_start_integration():
    log("Testing Integration with Portfolio Start Logic...")
    
    # Mock PV Series (Daily, Trading Days)
    # Range: 2023-01-01 to 2023-10-27
    dates = pd.date_range("2023-01-01", "2023-10-27", freq="B") # Business days
    pv = pd.Series(1000.0, index=dates)
    inception = dates[0]
    
    # Test 1M Horizon
    # As Of: Oct 27. Target: Sep 27.
    # PV has Sep 27 (Wed). So Start should be Sep 27.
    start_1m = get_portfolio_horizon_start(pv, inception, "1M")
    expected_1m = pd.Timestamp("2023-09-27")
    assert start_1m == expected_1m, f"Portfolio 1M Start should be {expected_1m}, got {start_1m}"
    
    # Test MTD Horizon
    # Target: Sep 30 (Sat).
    # PV does not have Sep 30. Last valid PV <= Target is Fri Sep 29.
    # (Backward Snap is implemented in get_portfolio_horizon_start logic? No, logic says 'prev_dates.max()')
    # Let's check: prev_dates = pv_idx[pv_idx <= target_date]
    # target = Sep 30. pv has Sep 29. 29 <= 30. So max is Sep 29.
    start_mtd = get_portfolio_horizon_start(pv, inception, "MTD")
    expected_mtd = pd.Timestamp("2023-09-29")
    assert start_mtd == expected_mtd, f"Portfolio MTD Start should be {expected_mtd} (Backward Snap), got {start_mtd}"
    
    log("Portfolio Start Integration (Backward Snap) Verified.\n")

if __name__ == "__main__":
    try:
        log("=== Starting Consistency Sweep (Audit 15) ===")
        test_effective_anchor_logic()
        test_horizon_target_dates()
        test_portfolio_start_integration()
        log("=== ALL SYSTEMS GO: Consistency Sweep Passed. Reference Logic is Unified. ===", "SUCCESS")
    except AssertionError as e:
        log(f"AUDIT FAILED: {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        log(f"RUNTIME ERROR: {e}", "CRITICAL")
        sys.exit(1)
