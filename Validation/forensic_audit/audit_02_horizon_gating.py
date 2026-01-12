import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from financial_math import (
    compute_horizon_twr,
    compute_security_modified_dietz,
    annualize_return,
    is_annualized,
    get_portfolio_horizon_start,
    modified_dietz_for_asset_class_window,
)

def log_success(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def test_portfolio_gating():
    print("\n--- Testing Portfolio Horizon Gating ---")
    
    # Use Friday June 30, 2023 to avoid weekend effective anchor snapping
    end_date = pd.Timestamp("2023-06-30")
    
    # Define exact day counts for horizons
    horizon_days_map = {
        "1W": 7,
        "1M": 30, # Approx for logic check, actual logic is calendar
        "3M": 90,
        "6M": 180,
        "1Y": 365,
        "3Y": 365*3,
        "5Y": 365*5
    }
    
    # We will create a PV series that is exactly N days long and check if it passes/gates
    
    # Case 1: Just Short of 1Y (364 days)
    # Start = End - 364 days.
    # Lived days = (End - Start) + 1? No, (End - Start).
    # If Start=Jan 1, End=Jan 2. Lived = 2 days?
    # Engine logic: lived_days = (as_of - inception).days + 1.
    
    # 1Y requires 365 days.
    # Let's try history length = 364 days.
    start_short = end_date - timedelta(days=363) # 363 days diff + 1 = 364 lived
    pv_short = pd.Series(100.0, index=pd.date_range(start_short, end_date))
    # Fix: empty cf must have columns
    twr_1y_fail = compute_horizon_twr(pv_short, pd.DataFrame(columns=["date", "amount"]), start_short, "1Y")
    
    if np.isnan(twr_1y_fail):
        log_success("1Y TWR Correctly Gated (NaN) for 364-day history")
    else:
        log_fail(f"1Y TWR returned value {twr_1y_fail} for 364-day history (Should be NaN)")

    # Case 2: Exactly 1Y (Requires data covering full 365 day window)
    # We need pv.min() <= (end - 365).
    # So we need timedelta(days=365).
    # This results in 366 data points (inclusive), covering exactly 365 days of return.
    start_exact = end_date - timedelta(days=365) 
    pv_exact = pd.Series(100.0, index=pd.date_range(start_exact, end_date))
    # Fix: Initialize DataFrame with columns to avoid KeyError 'date'
    twr_1y_pass = compute_horizon_twr(pv_exact, pd.DataFrame(columns=["date", "amount"]), start_exact, "1Y")
    
    if not np.isnan(twr_1y_pass):
         log_success("1Y TWR Exists for 365-day history")
    else:
         log_fail("1Y TWR is NaN for 365-day history (Should exist)")

    # Case 3: 3Y Annualization Check
    # Create 3 years of data. +10% each year.
    # Total = 1.1^3 = 1.331.
    # Annualized should be 10%.
    start_3y = end_date - timedelta(days=365*3) # 3 years exactly?
    # 3 * 365 = 1095 days.
    # Lived = 1096 days.
    
    dates_3y = pd.date_range(start_3y, end_date, freq="D")
    
    # Construct PV growing at 10% CAGR
    # PV = 100 * (1.10)^(years)
    years = (dates_3y - start_3y).days / 365.0 # Approx
    vals = 100.0 * (1.10 ** years)
    pv_3y = pd.Series(vals, index=dates_3y)
    
    # Fix: empty cf
    twr_3y = compute_horizon_twr(pv_3y, pd.DataFrame(columns=["date", "amount"]), start_3y, "3Y")
    
    # 365*3 = 1095 days. Logic uses 365.25 for annualization.
    # Duration = 1095 / 365.25 = 2.997 years.
    # So it might NOT be annualized if strictly < 3 years?
    # But horizon label "3Y" pulls data from 3*365 days ago.
    # Then `annualize_return` checks if years > 1.0.
    # Yes, 2.99 years > 1.0. So it WILL annualize.
    
    # Expected: The return over the period is 1.10^3 - 1 = 33.1% cumulative.
    # Annualized: (1.331)^(1/2.997) - 1 approx 10%.
    
    if abs(twr_3y - 0.10) < 0.01:
        log_success(f"3Y TWR Annualized correctly: {twr_3y:.4f} (~10%)")
    else:
        log_fail(f"3Y TWR verification failed. Got {twr_3y}, expected ~0.10")

    # Case 4: YTD Logic
    # If inception > Jan 1, YTD should be NaN (as per code logic).
    jan1 = pd.Timestamp("2023-01-01")
    inception_late = pd.Timestamp("2023-02-01")
    pv_late = pd.Series([100.0, 110.0], index=[inception_late, end_date])
    
    # Fix: empty cf
    twr_ytd = compute_horizon_twr(pv_late, pd.DataFrame(columns=["date", "amount"]), inception_late, "YTD")
    if np.isnan(twr_ytd):
        log_success("YTD TWR Correctly Gated (NaN) for post-Jan 1 inception")
    else:
        log_fail(f"YTD TWR returned {twr_ytd} for post-Jan 1 inception (Should be NaN)")

def test_security_gating():
    print("\n--- Testing Security Horizon Gating ---")
    
    end_date = pd.Timestamp("2023-12-31")
    
    # Create tickers with specific ages
    tickers_setup = [
        {"name": "AGE_10D", "days": 10},
        {"name": "AGE_40D", "days": 40},
        {"name": "AGE_400D", "days": 400},
        {"name": "AGE_5Y",   "days": 365*5 + 10}
    ]
    
    prices_data = {}
    tx_data = []
    holdings_data = []
    
    for t in tickers_setup:
        name = t["name"]
        days = t["days"]
        start = end_date - timedelta(days=days)
        dates = pd.date_range(start, end_date, freq="D")
        prices_data[name] = pd.Series(100.0, index=dates)
        
        tx_data.append({"date": start, "ticker": name, "shares": 10, "amount": -1000})
        holdings_data.append({"ticker": name, "shares": 10})

    prices = pd.DataFrame(prices_data)
    transactions = pd.DataFrame(tx_data)
    holdings = pd.DataFrame(holdings_data)
    
    # Horizons to test
    horizons = ["1W", "1M", "3M", "1Y", "3Y", "5Y", "SI"]
    
    df = compute_security_modified_dietz(
        transactions, prices, holdings, horizons=horizons
    )
    
    # Helper to get result
    def get_res(ticker, horizon):
        row = df[df["ticker"] == ticker]
        if row.empty: return np.nan
        return row.iloc[0][horizon]

    # CHECK AGE_10D (Should pass 1W, SI. Fail 1M+)
    if not np.isnan(get_res("AGE_10D", "1W")): log_success("AGE_10D passed 1W")
    else: log_fail("AGE_10D failed 1W")
    
    if np.isnan(get_res("AGE_10D", "1M")): log_success("AGE_10D gated 1M")
    else: log_fail("AGE_10D failed gate 1M")
    
    if not np.isnan(get_res("AGE_10D", "SI")): log_success("AGE_10D passed SI")
    else: log_fail("AGE_10D failed SI")

    # CHECK AGE_400D (Should pass 1Y. Fail 3Y)
    if not np.isnan(get_res("AGE_400D", "1Y")): log_success("AGE_400D passed 1Y")
    else: log_fail("AGE_400D failed 1Y")
    
    if np.isnan(get_res("AGE_400D", "3Y")): log_success("AGE_400D gated 3Y")
    else: log_fail("AGE_400D failed gate 3Y")
    
    # CHECK AGE_5Y (Should pass 5Y)
    if not np.isnan(get_res("AGE_5Y", "5Y")): log_success("AGE_5Y passed 5Y")
    else: log_fail("AGE_5Y failed 5Y")

def test_future_dates():
    print("\n--- Testing Future Date Handling ---")
    
    end_date = pd.Timestamp("2023-01-01") # Only 1 day of data
    pv = pd.Series([100.0], index=[end_date])
    cf = pd.DataFrame(columns=["date", "amount"])
    inception = end_date
    
    twr_1d = compute_horizon_twr(pv, cf, inception, "1D")
    if np.isnan(twr_1d):
        log_success("1D TWR NaN for 1-day history (Correct)")
    else:
        log_fail(f"1D TWR returned {twr_1d} for 1-day history")


def test_horizon_start_date_consistency():
    """
    GIPS COMPLIANCE TEST: Verify that horizon start dates are consistent
    between ticker-level and asset-class-level calculations.
    
    This test catches the bug where:
    - Ticker-level 1W showed 1/1/2026 (forward snap to next trading day)
    - Asset-class level showed 12/31/2025 (backward snap to prior trading day)
    
    GIPS requires consistent backward snapping across all calculations.
    """
    print("\n--- Testing Horizon Start Date Consistency (GIPS Compliance) ---")
    
    # Simulate a scenario with a holiday gap (like New Year's Day)
    # Trading days: Dec 31, 2025 (Wed), Jan 2, 2026 (Fri after NYD holiday)
    
    # Create price series with holiday gap
    dates = [
        pd.Timestamp("2025-12-24"),  # Wed
        pd.Timestamp("2025-12-26"),  # Fri (after Christmas)
        pd.Timestamp("2025-12-29"),  # Mon
        pd.Timestamp("2025-12-30"),  # Tue  
        pd.Timestamp("2025-12-31"),  # Wed (NYE)
        # Jan 1 is holiday - NO DATA
        pd.Timestamp("2026-01-02"),  # Fri
        pd.Timestamp("2026-01-05"),  # Mon
        pd.Timestamp("2026-01-06"),  # Tue
        pd.Timestamp("2026-01-07"),  # Wed
        pd.Timestamp("2026-01-08"),  # Thu (as_of)
    ]
    
    prices = pd.Series([100.0] * len(dates), index=dates)
    pv = pd.Series([10000.0] * len(dates), index=dates)
    
    as_of = pd.Timestamp("2026-01-08")
    inception = dates[0]
    
    # 1W target = as_of - 7 days = 2026-01-01 (NYD - holiday!)
    # Correct backward snap should find 2025-12-31
    
    # Test Portfolio-level horizon start
    port_start_1w = get_portfolio_horizon_start(pv, inception, "1W")
    
    expected_1w_start = pd.Timestamp("2025-12-31")  # Should backward snap to Dec 31
    
    if port_start_1w == expected_1w_start:
        log_success(f"Portfolio 1W start date correctly backward-snapped to {port_start_1w.date()}")
    else:
        log_fail(f"Portfolio 1W start date {port_start_1w} != expected {expected_1w_start} (Holiday snap failure)")
    
    # Test all horizons for backward snap consistency
    print("\n  Testing backward snap consistency for all horizons:")
    
    horizons_to_test = ["1W", "1M", "3M", "6M", "1Y"]
    
    for h in horizons_to_test:
        start = get_portfolio_horizon_start(pv, inception, h)
        if start is None:
            print(f"    {h}: N/A (insufficient data)")
            continue
            
        # Verify start is in pv.index (backward snapped correctly)
        if start in pv.index:
            log_success(f"{h}: Start date {start.date()} is a valid trading day")
        else:
            # Check if it was properly snapped
            pv_dates = pv.index.sort_values()
            nearest_back = pv_dates[pv_dates <= start]
            if len(nearest_back) > 0:
                log_fail(f"{h}: Start {start.date()} not in pv.index, should snap to {nearest_back.max().date()}")
            else:
                print(f"    {h}: Start {start.date()} is before all data (acceptable for new portfolios)")


def test_ticker_vs_asset_class_start_date_alignment():
    """
    GIPS COMPLIANCE TEST: Verify that for the same horizon and same trading days,
    ticker-level and asset-class-level meta_start_date values align.
    
    This ensures the audit modal shows consistent dates across different views.
    """
    print("\n--- Testing Ticker vs Asset Class Start Date Alignment ---")
    
    # Create test data: Single ticker in single asset class
    # This should guarantee identical start dates
    
    dates = pd.date_range("2025-01-01", "2026-01-08", freq="B")  # Business days only
    
    prices_data = {"AAPL": pd.Series(100.0, index=dates)}
    prices = pd.DataFrame(prices_data)
    
    transactions = pd.DataFrame([
        {"date": pd.Timestamp("2025-01-02"), "ticker": "AAPL", "shares": 100, "amount": -10000}
    ])
    
    holdings = pd.DataFrame([
        {"ticker": "AAPL", "shares": 100, "asset_class": "US Large Cap"}
    ])
    
    as_of = dates[-1]
    
    # Get ticker-level Modified Dietz with meta columns
    sec_md = compute_security_modified_dietz(
        transactions, prices, holdings, 
        horizons=["1W", "1M", "MTD", "3M", "6M", "YTD", "1Y", "SI"]
    )
    
    # Get asset-class level Modified Dietz
    # For 1W, start = as_of - 7 days
    target_1w = as_of - pd.Timedelta(days=7)
    
    ac_ret = modified_dietz_for_asset_class_window(
        tickers=["AAPL"],
        prices=prices,
        tx_all=transactions,
        start=target_1w,
        end=as_of,
        return_components=True
    )
    
    # Compare start dates
    ticker_row = sec_md[sec_md["ticker"] == "AAPL"].iloc[0]
    
    ticker_1w_start = ticker_row.get("meta_1W_start_date")
    ac_1w_start = ac_ret.get("start_date") if isinstance(ac_ret, dict) else None
    
    if ticker_1w_start is None or ac_1w_start is None:
        print(f"  WARNING: Could not extract start dates (Ticker: {ticker_1w_start}, AC: {ac_1w_start})")
    elif pd.Timestamp(ticker_1w_start) == pd.Timestamp(ac_1w_start):
        log_success(f"1W start dates match: Ticker={ticker_1w_start}, AC={ac_1w_start}")
    else:
        log_fail(f"1W start date MISMATCH: Ticker={ticker_1w_start} vs AC={ac_1w_start}")
    
    # Test SI alignment
    ticker_si_start = ticker_row.get("meta_SI_start_date")
    
    # For SI, asset class start should be the day before first trade
    first_trade = transactions["date"].min()
    expected_si_start = first_trade - pd.Timedelta(days=1)
    
    if ticker_si_start is not None:
        if pd.Timestamp(ticker_si_start) <= first_trade:
            log_success(f"SI start date {ticker_si_start} is on/before first trade {first_trade.date()}")
        else:
            log_fail(f"SI start date {ticker_si_start} is AFTER first trade {first_trade.date()}")


def test_mtd_ytd_backward_snap():
    """
    GIPS COMPLIANCE TEST: Verify MTD and YTD use backward snap for anchoring.
    
    MTD: Should snap to last trading day of prior month
    YTD: Should snap to last trading day of prior year
    """
    print("\n--- Testing MTD/YTD Backward Snap Logic ---")
    
    # Create data with typical month-end gaps
    # December 31, 2025 (Wed) -> January 2, 2026 (Fri after NYD)
    dates = pd.date_range("2025-11-01", "2026-01-08", freq="B")
    
    pv = pd.Series(10000.0, index=dates)
    inception = dates[0]
    as_of = pd.Timestamp("2026-01-08")
    
    # MTD Test
    # Jan MTD should anchor to Dec 31, 2025 (last trading day of Dec)
    mtd_start = get_portfolio_horizon_start(pv, inception, "MTD")
    
    # Find actual last trading day <= Dec 31
    dec_31 = pd.Timestamp("2025-12-31")
    expected_mtd = dates[dates <= dec_31].max()
    
    if mtd_start == expected_mtd:
        log_success(f"MTD correctly anchored to {mtd_start.date()} (last trading day of prior month)")
    elif mtd_start is None:
        log_fail("MTD returned None (unexpected)")
    else:
        log_fail(f"MTD anchor {mtd_start.date()} != expected {expected_mtd.date()}")
    
    # YTD Test  
    # YTD should anchor to last trading day <= Dec 31, 2025
    ytd_start = get_portfolio_horizon_start(pv, inception, "YTD")
    
    # Same expected date since YTD anchors to prior year end
    if ytd_start == expected_mtd:
        log_success(f"YTD correctly anchored to {ytd_start.date()} (last trading day of prior year)")
    elif ytd_start is None:
        # YTD could be None if inception is after Jan 1
        print(f"  YTD returned None (may be correct if inception > Jan 1)")
    else:
        # YTD may have different anchor logic, verify it's at least backward snapped
        prior_dates = dates[dates <= ytd_start]
        if len(prior_dates) > 0 and ytd_start in dates:
            log_success(f"YTD anchor {ytd_start.date()} is a valid trading day")
        else:
            log_fail(f"YTD anchor {ytd_start.date()} is not a valid trading day")


def test_pl_vs_return_start_date_alignment():
    """
    GIPS COMPLIANCE TEST: Verify P/L and Return calculations use identical start dates.
    
    This catches the bug where Horizon Return table showed different start date 
    than Horizon P/L table for the same horizon.
    """
    print("\n--- Testing P/L vs Return Start Date Alignment ---")
    
    # Create data with holiday gap to expose potential forward/backward snap mismatch
    dates = [
        pd.Timestamp("2025-12-29"),
        pd.Timestamp("2025-12-30"),
        pd.Timestamp("2025-12-31"),
        # Jan 1 holiday
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
        pd.Timestamp("2026-01-06"),
        pd.Timestamp("2026-01-07"),
        pd.Timestamp("2026-01-08"),
    ]
    
    pv = pd.Series([10000.0 + i*100 for i in range(len(dates))], index=dates)
    inception = dates[0]
    
    horizons = ["1D", "1W", "MTD"]
    
    for h in horizons:
        start = get_portfolio_horizon_start(pv, inception, h)
        
        if start is None:
            print(f"  {h}: N/A (insufficient data)")
            continue
        
        # The key test: start must be IN pv.index (a valid trading day)
        # If forward snap was used instead of backward snap, start might be
        # set to Jan 2 instead of Dec 31 for 1W
        
        if start in pv.index:
            log_success(f"{h}: Start date {start.date()} is valid trading day (consistent for P/L and Return)")
        else:
            # This is the bug we're catching
            pv_idx = pv.index.sort_values()
            backward_snap = pv_idx[pv_idx <= start].max() if len(pv_idx[pv_idx <= start]) > 0 else None
            forward_snap = pv_idx[pv_idx >= start].min() if len(pv_idx[pv_idx >= start]) > 0 else None
            
            log_fail(f"{h}: Start {start.date()} NOT in pv.index! Backward snap would be {backward_snap}, Forward snap would be {forward_snap}")


if __name__ == "__main__":
    try:
        test_portfolio_gating()
        test_security_gating()
        test_future_dates()
        
        # NEW: GIPS Compliance Tests for Start Date Consistency
        test_horizon_start_date_consistency()
        test_ticker_vs_asset_class_start_date_alignment()
        test_mtd_ytd_backward_snap()
        test_pl_vs_return_start_date_alignment()
        
        print("\n[SUCCESS] audit_02_horizon_gating.py passed all checks.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
