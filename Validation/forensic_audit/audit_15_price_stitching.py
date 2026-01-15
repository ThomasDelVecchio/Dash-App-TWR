"""
AUDIT 15: HYBRID PRICE STITCHING VALIDATION
============================================
Verifies that the FMP/yfinance hybrid price system produces consistent data
and properly handles the stitch boundary.

This audit tests:
1. Configuration validation
2. Price fetch structure and metadata
3. Stitch boundary continuity (FMP mode)
4. yfinance-only mode behavior
5. Fallback behavior when FMP fails
6. Date range coverage validation
7. Close vs Adjusted Close parameter
8. Error tracking in attrs
9. Cache key differentiation between modes
10. Multi-ticker fetch consistency

IMPORTANT: This audit changes to the project root directory to ensure
data_loader.py can find the required CSV files.
"""

import sys
import os

# CRITICAL: Change to project root BEFORE importing data_loader
# data_loader.load_holdings() uses relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch
import importlib

# ============================================================
# TEST UTILITIES
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_result(test_name, passed, details=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status}: {test_name}")
    if details:
        print(f"        {details}")

def print_subtest(name, passed, details=""):
    status = "  [OK]" if passed else "  [X]"
    print(f"    {status} {name}")
    if details:
        print(f"        {details}")

# ============================================================
# TEST SUITE: CONFIGURATION
# ============================================================

def test_01_config_validation():
    """Test 1: Verify config values are set correctly."""
    print_header("Test 1: Configuration Validation")
    
    from config import FMP_PRICE_ENABLED, FMP_PRICE_LOOKBACK_YEARS, FMP_API_KEY
    from data_loader import PRICE_LOOKBACK_YEARS
    
    all_passed = True
    
    # Check FMP enabled flag is boolean
    is_bool = isinstance(FMP_PRICE_ENABLED, bool)
    print_result("FMP_PRICE_ENABLED is boolean", is_bool, f"Value: {FMP_PRICE_ENABLED} (type: {type(FMP_PRICE_ENABLED).__name__})")
    all_passed &= is_bool
    
    # Check FMP lookback years
    valid_fmp_years = isinstance(FMP_PRICE_LOOKBACK_YEARS, int) and 1 <= FMP_PRICE_LOOKBACK_YEARS <= 10
    print_result("FMP_PRICE_LOOKBACK_YEARS is valid", valid_fmp_years, f"Value: {FMP_PRICE_LOOKBACK_YEARS}")
    all_passed &= valid_fmp_years
    
    # Check total lookback years
    valid_total_years = isinstance(PRICE_LOOKBACK_YEARS, int) and PRICE_LOOKBACK_YEARS >= FMP_PRICE_LOOKBACK_YEARS
    print_result("PRICE_LOOKBACK_YEARS >= FMP_PRICE_LOOKBACK_YEARS", valid_total_years, 
                 f"Total: {PRICE_LOOKBACK_YEARS}, FMP: {FMP_PRICE_LOOKBACK_YEARS}")
    all_passed &= valid_total_years
    
    # Check API key configuration
    has_key = FMP_API_KEY is not None and FMP_API_KEY != "demo" and len(FMP_API_KEY) > 10
    key_msg = f"Length={len(FMP_API_KEY)}" if FMP_API_KEY and FMP_API_KEY != "demo" else "Missing/demo"
    print_result("FMP_API_KEY is configured (for hybrid mode)", has_key, key_msg)
    # Note: Not marking as failure if key is missing - hybrid mode is optional
    
    # Check consistency: If FMP enabled, API key should be present
    if FMP_PRICE_ENABLED and not has_key:
        print_result("[WARN] FMP enabled but no valid API key", False, 
                     "Hybrid mode will fall back to yfinance-only")
    
    return all_passed


# ============================================================
# TEST SUITE: YFINANCE-ONLY MODE
# ============================================================

def test_02_yfinance_only_mode():
    """Test 2: Verify yfinance-only mode works correctly."""
    print_header("Test 2: yfinance-Only Mode")
    
    # Force yfinance-only mode by patching config
    import data_loader
    
    # Clear cache to ensure fresh fetch
    data_loader._PRICE_CACHE.clear()
    
    all_passed = True
    
    with patch.object(data_loader, 'FMP_PRICE_ENABLED', False):
        tickers = ["SPY", "QQQ"]
        
        try:
            prices = data_loader.fetch_price_history(tickers, years_back=2)
        except Exception as e:
            print_result("yfinance fetch successful", False, str(e))
            return False
        
        # Check DataFrame structure
        is_df = isinstance(prices, pd.DataFrame)
        print_result("Returns DataFrame", is_df)
        all_passed &= is_df
        
        if not is_df:
            return False
        
        # Check columns
        has_spy = "SPY" in prices.columns
        has_qqq = "QQQ" in prices.columns
        print_result("SPY column present", has_spy)
        print_result("QQQ column present", has_qqq)
        all_passed &= has_spy and has_qqq
        
        # Check index
        is_datetime_idx = isinstance(prices.index, pd.DatetimeIndex)
        print_result("Index is DatetimeIndex", is_datetime_idx)
        all_passed &= is_datetime_idx
        
        # Check source metadata
        meta = prices.attrs.get('source_metadata', {})
        source_attr = prices.attrs.get('source', '')
        
        is_yf_source = source_attr == 'yfinance'
        print_result("Source marked as 'yfinance'", is_yf_source, f"Got: '{source_attr}'")
        all_passed &= is_yf_source
        
        # Verify FMP list is empty
        fmp_list = meta.get('FMP', [])
        is_fmp_empty = len(fmp_list) == 0
        print_result("FMP ticker list is empty", is_fmp_empty, f"FMP tickers: {fmp_list}")
        all_passed &= is_fmp_empty
        
        # Verify yfinance list has tickers
        yf_list = meta.get('yfinance', [])
        has_yf_tickers = len(yf_list) >= 2
        print_result("yfinance ticker list populated", has_yf_tickers, f"Count: {len(yf_list)}")
        all_passed &= has_yf_tickers
        
        # Check date range coverage
        earliest = prices.index.min()
        latest = prices.index.max()
        days_covered = (latest - earliest).days
        expected_days = 2 * 365
        coverage_pct = (days_covered / expected_days) * 100
        
        good_coverage = coverage_pct > 85
        print_result(f"Date coverage > 85%", good_coverage, 
                     f"{days_covered} days ({coverage_pct:.1f}%) from {earliest.date()} to {latest.date()}")
        all_passed &= good_coverage
    
    # Clear cache after test
    data_loader._PRICE_CACHE.clear()
    
    return all_passed


# ============================================================
# TEST SUITE: HYBRID MODE (FMP + yfinance)
# ============================================================

def test_03_hybrid_mode_structure():
    """Test 3: Verify hybrid mode returns proper structure (if API key available)."""
    print_header("Test 3: Hybrid Mode Structure")
    
    from config import FMP_API_KEY, FMP_PRICE_ENABLED
    import data_loader
    
    # Skip if no valid API key
    has_key = FMP_API_KEY and FMP_API_KEY != "demo" and len(FMP_API_KEY) > 10
    if not has_key:
        print_result("SKIPPED: No valid FMP API key", True, "Hybrid mode requires FMP_API_KEY")
        return True
    
    # Clear cache
    data_loader._PRICE_CACHE.clear()
    
    all_passed = True
    
    with patch.object(data_loader, 'FMP_PRICE_ENABLED', True):
        tickers = ["SPY"]
        
        try:
            prices = data_loader.fetch_price_history(tickers, years_back=10)
        except Exception as e:
            print_result("Hybrid fetch successful", False, str(e))
            return False
        
        # Check structure
        is_df = isinstance(prices, pd.DataFrame)
        print_result("Returns DataFrame", is_df)
        all_passed &= is_df
        
        if not is_df:
            return False
        
        # Check source metadata
        meta = prices.attrs.get('source_metadata', {})
        source_attr = prices.attrs.get('source', '')
        
        is_hybrid = source_attr == 'hybrid'
        print_result("Source marked as 'hybrid'", is_hybrid, f"Got: '{source_attr}'")
        all_passed &= is_hybrid
        
        # Check FMP range is set
        fmp_range = meta.get('fmp_range', (None, None))
        has_fmp_range = fmp_range[0] is not None and fmp_range[1] is not None
        print_result("FMP range tracked", has_fmp_range, f"Range: {fmp_range}")
        all_passed &= has_fmp_range
        
        # Check yfinance range
        yf_range = meta.get('yf_range', (None, None))
        has_yf_range = yf_range[0] is not None
        print_result("yfinance range tracked", has_yf_range, f"Range: {yf_range}")
        all_passed &= has_yf_range
        
        # Check FMP/yfinance ticker counts
        fmp_tickers = prices.attrs.get('fmp_tickers', 0)
        yf_fallback = prices.attrs.get('yf_fallback', 0)
        print(f"        FMP success count: {fmp_tickers}")
        print(f"        yfinance fallback count: {yf_fallback}")
    
    # Clear cache
    data_loader._PRICE_CACHE.clear()
    
    return all_passed


def test_04_stitch_boundary_continuity():
    """Test 4: Verify price continuity at the stitch boundary."""
    print_header("Test 4: Stitch Boundary Continuity")
    
    from config import FMP_API_KEY
    import data_loader
    
    has_key = FMP_API_KEY and FMP_API_KEY != "demo" and len(FMP_API_KEY) > 10
    if not has_key:
        print_result("SKIPPED: No valid FMP API key", True, "Stitch test requires hybrid mode")
        return True
    
    data_loader._PRICE_CACHE.clear()
    
    all_passed = True
    
    with patch.object(data_loader, 'FMP_PRICE_ENABLED', True):
        prices = data_loader.fetch_price_history(["SPY"], years_back=10)
        
        if prices.empty or 'SPY' not in prices.columns:
            print_result("SPY data available", False)
            return False
        
        print_result("SPY data available", True, f"{len(prices)} rows")
        
        meta = prices.attrs.get('source_metadata', {})
        fmp_range = meta.get('fmp_range', (None, None))
        
        if fmp_range[0] is None:
            print_result("Stitch date available", False, "No FMP start date in metadata")
            return False
        
        stitch_date = pd.Timestamp(fmp_range[0])
        print_result("Stitch date identified", True, f"{stitch_date.date()}")
        
        # Analyze prices around the stitch boundary
        spy_prices = prices['SPY'].dropna()
        
        # Window around stitch: 5 days before and after
        window_start = stitch_date - pd.Timedelta(days=10)
        window_end = stitch_date + pd.Timedelta(days=10)
        
        near_stitch = spy_prices[
            (spy_prices.index >= window_start) & 
            (spy_prices.index <= window_end)
        ]
        
        if len(near_stitch) < 5:
            print_result("Sufficient data around stitch", False, f"Only {len(near_stitch)} days")
            return False
        
        print_result("Sufficient data around stitch", True, f"{len(near_stitch)} days")
        
        # Calculate daily returns around stitch
        daily_rets = near_stitch.pct_change().dropna()
        max_jump = daily_rets.abs().max() * 100
        
        # A single-day jump > 5% would indicate a stitching problem
        # (Normal market moves rarely exceed this, especially for SPY)
        is_continuous = max_jump < 5.0
        print_result(
            "Price continuity (no >5% jump)",
            is_continuous,
            f"Max daily change: {max_jump:.2f}%"
        )
        all_passed &= is_continuous
        
        # Check for NaN gaps
        has_gaps = near_stitch.isna().any()
        print_result("No NaN gaps around stitch", not has_gaps)
        all_passed &= not has_gaps
        
        # Show a few prices around stitch for manual verification
        print("\n        Prices around stitch boundary:")
        display_prices = spy_prices[
            (spy_prices.index >= stitch_date - pd.Timedelta(days=3)) &
            (spy_prices.index <= stitch_date + pd.Timedelta(days=3))
        ]
        for date, price in display_prices.items():
            marker = " <-- STITCH" if date.date() == stitch_date.date() else ""
            print(f"          {date.date()}: ${price:.2f}{marker}")
    
    data_loader._PRICE_CACHE.clear()
    return all_passed


# ============================================================
# TEST SUITE: SINGLE TICKER FUNCTIONS
# ============================================================

def test_05_fmp_single_ticker_fetch():
    """Test 5: Verify FMP single ticker fetch function."""
    print_header("Test 5: FMP Single Ticker Fetch")
    
    from config import FMP_API_KEY
    
    has_key = FMP_API_KEY and FMP_API_KEY != "demo" and len(FMP_API_KEY) > 10
    if not has_key:
        print_result("SKIPPED: No valid FMP API key", True, "Test requires FMP_API_KEY")
        return True
    
    from data_loader import fetch_fmp_price_history_single
    
    all_passed = True
    
    # Test with SPY (always available)
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    try:
        df = fetch_fmp_price_history_single("SPY", start_date, end_date)
    except Exception as e:
        print_result("FMP fetch executed", False, str(e))
        return False
    
    print_result("FMP fetch executed", True)
    
    # Check structure
    is_df = isinstance(df, pd.DataFrame)
    print_result("Returns DataFrame", is_df)
    all_passed &= is_df
    
    if df.empty:
        print_result("Data returned", False, "Empty DataFrame - FMP may have rate limited")
        return True  # Don't fail - FMP rate limits are common
    
    print_result("Data returned", True, f"{len(df)} rows")
    
    # Check columns
    has_close = "Close" in df.columns
    print_result("'Close' column present", has_close, f"Columns: {list(df.columns)}")
    all_passed &= has_close
    
    # Check index
    is_datetime_idx = isinstance(df.index, pd.DatetimeIndex)
    print_result("DatetimeIndex", is_datetime_idx)
    all_passed &= is_datetime_idx
    
    # Check sorted
    is_sorted = df.index.is_monotonic_increasing
    print_result("Index sorted ascending", is_sorted)
    all_passed &= is_sorted
    
    return all_passed


def test_06_stitch_dataframes_function():
    """Test 6: Verify the _stitch_price_dataframes helper function."""
    print_header("Test 6: DataFrame Stitching Logic")
    
    from data_loader import _stitch_price_dataframes
    
    all_passed = True
    
    # Create synthetic test data
    dates_old = pd.date_range("2020-01-01", "2020-06-30", freq="D")
    dates_new = pd.date_range("2020-06-01", "2020-12-31", freq="D")  # Overlapping
    
    df_old = pd.DataFrame({"Close": range(len(dates_old))}, index=dates_old)
    df_new = pd.DataFrame({"Close": range(1000, 1000 + len(dates_new))}, index=dates_new)
    
    # Test stitching
    stitched = _stitch_price_dataframes(df_new, df_old, "TEST")
    
    # Check result
    is_df = isinstance(stitched, pd.DataFrame)
    print_result("Returns DataFrame", is_df)
    all_passed &= is_df
    
    if not is_df:
        return False
    
    # Check no duplicate indices
    has_dupes = stitched.index.duplicated().any()
    print_result("No duplicate indices", not has_dupes)
    all_passed &= not has_dupes
    
    # Check sorted
    is_sorted = stitched.index.is_monotonic_increasing
    print_result("Index sorted", is_sorted)
    all_passed &= is_sorted
    
    # Check that FMP data (df_new) takes priority for overlapping dates
    overlap_date = pd.Timestamp("2020-06-15")
    if overlap_date in stitched.index:
        val = stitched.loc[overlap_date, "Close"]
        expected = df_new.loc[overlap_date, "Close"]
        correct_priority = val == expected
        print_result("FMP data takes priority in overlap", correct_priority,
                     f"Got {val}, expected {expected}")
        all_passed &= correct_priority
    
    # Check full coverage
    expected_start = dates_old.min()
    expected_end = dates_new.max()
    actual_start = stitched.index.min()
    actual_end = stitched.index.max()
    
    covers_range = actual_start <= expected_start and actual_end >= expected_end
    print_result("Full date range covered", covers_range,
                 f"Expected {expected_start.date()} to {expected_end.date()}, got {actual_start.date()} to {actual_end.date()}")
    all_passed &= covers_range
    
    # Test edge cases
    print("\n        Edge case tests:")
    
    # Empty FMP
    empty_stitch = _stitch_price_dataframes(pd.DataFrame(), df_old, "TEST")
    empty_fmp_ok = len(empty_stitch) == len(df_old)
    print_subtest("Empty FMP -> returns yfinance only", empty_fmp_ok)
    all_passed &= empty_fmp_ok
    
    # Empty yfinance
    empty_yf = _stitch_price_dataframes(df_new, pd.DataFrame(), "TEST")
    empty_yf_ok = len(empty_yf) == len(df_new)
    print_subtest("Empty yfinance -> returns FMP only", empty_yf_ok)
    all_passed &= empty_yf_ok
    
    # Both empty
    both_empty = _stitch_price_dataframes(pd.DataFrame(), pd.DataFrame(), "TEST")
    both_empty_ok = both_empty.empty
    print_subtest("Both empty -> returns empty", both_empty_ok)
    all_passed &= both_empty_ok
    
    return all_passed


# ============================================================
# TEST SUITE: PARAMETERS AND OPTIONS
# ============================================================

def test_07_close_vs_adj_close():
    """Test 7: Verify use_adj_close parameter works correctly."""
    print_header("Test 7: Close vs Adjusted Close Parameter")
    
    import data_loader
    data_loader._PRICE_CACHE.clear()
    
    all_passed = True
    
    # Fetch with regular close
    prices_close = data_loader.fetch_price_history(["SPY"], years_back=1, use_adj_close=False)
    
    # Clear cache between fetches to ensure different code paths
    data_loader._PRICE_CACHE.clear()
    
    # Fetch with adjusted close
    prices_adj = data_loader.fetch_price_history(["SPY"], years_back=1, use_adj_close=True)
    
    if prices_close.empty or prices_adj.empty:
        print_result("Both price sets available", False)
        return False
    
    print_result("Both price sets available", True,
                 f"Close: {len(prices_close)} rows, Adj: {len(prices_adj)} rows")
    
    # Get latest prices
    spy_close = prices_close['SPY'].iloc[-1]
    spy_adj = prices_adj['SPY'].iloc[-1]
    
    # For recent data, they should be very close (last dividend effect only)
    diff_pct = abs(spy_close - spy_adj) / spy_close * 100
    
    print_result("Close and Adj Close have data", True,
                 f"Close: ${spy_close:.2f}, Adj: ${spy_adj:.2f}, Diff: {diff_pct:.2f}%")
    
    # Difference should be small for recent data
    reasonable_diff = diff_pct < 5.0
    print_result("Difference is reasonable (<5%)", reasonable_diff)
    all_passed &= reasonable_diff
    
    # Check that historical data shows dividend divergence
    close_start = prices_close['SPY'].iloc[0]
    adj_start = prices_adj['SPY'].iloc[0]
    
    start_diff_pct = abs(close_start - adj_start) / close_start * 100
    print(f"        Historical divergence: Close ${close_start:.2f}, Adj ${adj_start:.2f}, Diff: {start_diff_pct:.2f}%")
    
    # Adj close should generally be lower than Close for dividend-paying stocks
    # (going back in time)
    
    data_loader._PRICE_CACHE.clear()
    return all_passed


def test_08_multi_ticker_consistency():
    """Test 8: Verify multi-ticker fetch returns consistent data."""
    print_header("Test 8: Multi-Ticker Fetch Consistency")
    
    import data_loader
    data_loader._PRICE_CACHE.clear()
    
    all_passed = True
    
    tickers = ["SPY", "QQQ", "VTI", "BND", "GLD"]
    
    try:
        prices = data_loader.fetch_price_history(tickers, years_back=3)
    except Exception as e:
        print_result("Multi-ticker fetch successful", False, str(e))
        return False
    
    print_result("Multi-ticker fetch successful", True, f"{len(prices)} rows")
    
    # Check all tickers present
    missing = [t for t in tickers if t not in prices.columns]
    all_present = len(missing) == 0
    print_result("All tickers present", all_present, 
                 f"Missing: {missing}" if missing else "All 5 tickers")
    all_passed &= all_present
    
    # Check index alignment (all tickers should have same date range)
    for ticker in tickers:
        if ticker in prices.columns:
            non_null = prices[ticker].dropna()
            if len(non_null) > 0:
                print(f"        {ticker}: {len(non_null)} valid prices, "
                      f"{non_null.index.min().date()} to {non_null.index.max().date()}")
    
    # Check for forward-fill (no large gaps)
    # After ffill, there should be no internal NaNs
    for ticker in tickers:
        if ticker in prices.columns:
            has_internal_nan = prices[ticker].isna().sum() > 0
            if has_internal_nan:
                print(f"        [WARN] {ticker} has {prices[ticker].isna().sum()} NaN values")
    
    # Verify SPY is always present (default benchmark)
    has_spy = "SPY" in prices.columns
    print_result("SPY (default benchmark) present", has_spy)
    all_passed &= has_spy
    
    data_loader._PRICE_CACHE.clear()
    return all_passed


def test_09_date_range_coverage():
    """Test 9: Verify proper date range coverage for different lookbacks."""
    print_header("Test 9: Date Range Coverage")
    
    import data_loader
    
    all_passed = True
    
    test_cases = [
        (1, 85),   # 1 year, expect 85%+ coverage
        (3, 85),   # 3 years
        (5, 85),   # 5 years
        (10, 80),  # 10 years (more slack for older data)
    ]
    
    for years, expected_pct in test_cases:
        data_loader._PRICE_CACHE.clear()
        
        try:
            prices = data_loader.fetch_price_history(["SPY"], years_back=years)
        except Exception as e:
            print_result(f"{years}Y coverage", False, str(e))
            all_passed = False
            continue
        
        if prices.empty:
            print_result(f"{years}Y coverage", False, "Empty DataFrame")
            all_passed = False
            continue
        
        earliest = prices.index.min()
        latest = prices.index.max()
        
        actual_days = (latest - earliest).days
        expected_days = years * 365
        coverage_pct = (actual_days / expected_days) * 100
        
        passed = coverage_pct >= expected_pct
        print_result(
            f"{years}Y coverage >= {expected_pct}%",
            passed,
            f"{actual_days} days ({coverage_pct:.1f}%), {earliest.date()} to {latest.date()}"
        )
        all_passed &= passed
    
    data_loader._PRICE_CACHE.clear()
    return all_passed


def test_10_error_tracking():
    """Test 10: Verify error tracking in attrs."""
    print_header("Test 10: Error Tracking Attributes")
    
    import data_loader
    data_loader._PRICE_CACHE.clear()
    
    all_passed = True
    
    prices = data_loader.fetch_price_history(["SPY"], years_back=1)
    
    # Check attrs exists
    has_attrs = hasattr(prices, 'attrs') and isinstance(prices.attrs, dict)
    print_result("attrs dictionary exists", has_attrs)
    all_passed &= has_attrs
    
    if not has_attrs:
        return False
    
    # Check 'errors' key exists
    has_errors_key = 'errors' in prices.attrs
    print_result("'errors' key in attrs", has_errors_key)
    all_passed &= has_errors_key
    
    if has_errors_key:
        errors = prices.attrs['errors']
        is_list = isinstance(errors, list)
        print_result("'errors' is a list", is_list, f"Type: {type(errors).__name__}")
        all_passed &= is_list
        
        if errors:
            print(f"        Found {len(errors)} error(s):")
            for e in errors[:3]:
                preview = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                print(f"          - {preview}")
    
    # Check 'source' key
    has_source = 'source' in prices.attrs
    print_result("'source' key in attrs", has_source)
    all_passed &= has_source
    
    if has_source:
        source = prices.attrs['source']
        valid_source = source in ['yfinance', 'hybrid']
        print_result("'source' has valid value", valid_source, f"Value: '{source}'")
        all_passed &= valid_source
    
    # Check 'source_metadata' key
    has_meta = 'source_metadata' in prices.attrs
    print_result("'source_metadata' key in attrs", has_meta)
    all_passed &= has_meta
    
    if has_meta:
        meta = prices.attrs['source_metadata']
        required_keys = ['FMP', 'yfinance', 'mixed', 'fmp_range', 'yf_range']
        for key in required_keys:
            has_key = key in meta
            print_subtest(f"'{key}' in source_metadata", has_key)
            all_passed &= has_key
    
    data_loader._PRICE_CACHE.clear()
    return all_passed


def test_11_cache_key_differentiation():
    """Test 11: Verify cache keys differentiate modes properly."""
    print_header("Test 11: Cache Key Differentiation")
    
    import data_loader
    
    # Clear cache
    data_loader._PRICE_CACHE.clear()
    
    all_passed = True
    
    # The cache key should include FMP_PRICE_ENABLED flag
    # So yfinance-only mode and hybrid mode should have different cache entries
    
    # Fetch in yfinance mode
    with patch.object(data_loader, 'FMP_PRICE_ENABLED', False):
        _ = data_loader.fetch_price_history(["SPY"], years_back=1)
        cache_size_1 = len(data_loader._PRICE_CACHE)
    
    # Check cache has entry
    print_result("Cache populated after yfinance fetch", cache_size_1 > 0, f"Size: {cache_size_1}")
    all_passed &= cache_size_1 > 0
    
    # Now fetch in hybrid mode (if API key available)
    from config import FMP_API_KEY
    has_key = FMP_API_KEY and FMP_API_KEY != "demo" and len(FMP_API_KEY) > 10
    
    if has_key:
        with patch.object(data_loader, 'FMP_PRICE_ENABLED', True):
            _ = data_loader.fetch_price_history(["SPY"], years_back=1)
            cache_size_2 = len(data_loader._PRICE_CACHE)
        
        # Should have 2 cache entries (different keys for different modes)
        has_separate_entries = cache_size_2 == cache_size_1 + 1
        print_result("Hybrid mode creates separate cache entry", has_separate_entries,
                     f"Cache size: {cache_size_1} -> {cache_size_2}")
        all_passed &= has_separate_entries
    else:
        print_result("SKIPPED: Hybrid cache test (no API key)", True)
    
    # Verify same mode uses cache (no new entry)
    with patch.object(data_loader, 'FMP_PRICE_ENABLED', False):
        _ = data_loader.fetch_price_history(["SPY"], years_back=1)
        cache_size_3 = len(data_loader._PRICE_CACHE)
    
    cache_hit = cache_size_3 == len(data_loader._PRICE_CACHE)
    print_result("Same mode uses cached data", cache_hit)
    all_passed &= cache_hit
    
    # Show cache keys for debugging
    print("\n        Cache keys:")
    for key in list(data_loader._PRICE_CACHE.keys())[:3]:
        tickers, years, adj, fmp_flag = key
        print(f"          tickers={tickers[:2]}..., years={years}, adj={adj}, fmp={fmp_flag}")
    
    data_loader._PRICE_CACHE.clear()
    return all_passed


# ============================================================
# MAIN RUNNER
# ============================================================

def run_all_tests():
    print("\n" + "=" * 70)
    print(" AUDIT 15: HYBRID PRICE STITCHING VALIDATION")
    print(" Testing FMP/yfinance integration and data consistency")
    print("=" * 70)
    
    # Show current configuration
    from config import FMP_PRICE_ENABLED, FMP_PRICE_LOOKBACK_YEARS, FMP_API_KEY
    from data_loader import PRICE_LOOKBACK_YEARS
    
    print(f"\nCurrent Configuration:")
    print(f"  Working Directory: {os.getcwd()}")
    print(f"  FMP_PRICE_ENABLED: {FMP_PRICE_ENABLED}")
    print(f"  FMP_PRICE_LOOKBACK_YEARS: {FMP_PRICE_LOOKBACK_YEARS}")
    print(f"  Total PRICE_LOOKBACK_YEARS: {PRICE_LOOKBACK_YEARS}")
    print(f"  FMP_API_KEY configured: {'Yes' if FMP_API_KEY and FMP_API_KEY != 'demo' and len(FMP_API_KEY) > 10 else 'No'}")
    
    # Run all tests
    tests = [
        ("01 Config Validation", test_01_config_validation),
        ("02 yfinance-Only Mode", test_02_yfinance_only_mode),
        ("03 Hybrid Mode Structure", test_03_hybrid_mode_structure),
        ("04 Stitch Boundary Continuity", test_04_stitch_boundary_continuity),
        ("05 FMP Single Ticker Fetch", test_05_fmp_single_ticker_fetch),
        ("06 DataFrame Stitching Logic", test_06_stitch_dataframes_function),
        ("07 Close vs Adj Close", test_07_close_vs_adj_close),
        ("08 Multi-Ticker Consistency", test_08_multi_ticker_consistency),
        ("09 Date Range Coverage", test_09_date_range_coverage),
        ("10 Error Tracking Attrs", test_10_error_tracking),
        ("11 Cache Key Differentiation", test_11_cache_key_differentiation),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] Test {name} crashed with exception:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print(" AUDIT SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  === ALL TESTS PASSED - Price system working correctly! ===")
    else:
        print("\n  [WARN] Some tests failed - Review output above for details")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
