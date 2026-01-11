import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dash_wrappers import (
    run_analytics_engine,
    calculate_active_metrics, 
    compute_drawdown_series,
    _get_daily_twr_curve
)

def log_success(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def test_risk_intelligence():
    print("\n--- Testing Risk Intelligence (Institutional Upgrade) ---")
    
    # Load Data
    print("Loading live data...")
    data = run_analytics_engine()
    
    # 1. Correlation Matrix Integrity
    print("Checking Correlation Matrix...")
    corr_matrix = data.get("correlation_matrix", {})
    
    # Convert dict of dicts to DataFrame
    df_corr = pd.DataFrame(corr_matrix)
    
    if df_corr.empty:
        print("[SKIP] Correlation matrix empty (insufficient data).")
    else:
        # Check Diagonal
        diag = np.diag(df_corr)
        if np.allclose(diag, 1.0, atol=1e-6):
            log_success("Correlation Matrix Diagonal is all 1.0")
        else:
            log_fail(f"Correlation Matrix Diagonal mismatch: {diag}")
            
        # Check Symmetry
        if np.allclose(df_corr, df_corr.T, atol=1e-6):
            log_success("Correlation Matrix is Symmetric")
        else:
            log_fail("Correlation Matrix is NOT Symmetric")
            
        # Check Values
        if df_corr.min().min() >= -1.0 and df_corr.max().max() <= 1.0:
            log_success("Correlation Values within [-1, 1]")
        else:
            log_fail("Correlation Values out of bounds")

        # Check Specific Relation (if data exists)
        if "US Large Cap" in df_corr.columns and "Fixed Income" in df_corr.columns:
            val = df_corr.loc["US Large Cap", "Fixed Income"]
            if abs(val) > 1e-6:
                log_success(f"US Large Cap vs Fixed Income Correlation is non-zero: {val:.4f}")
            else:
                print("[WARN] US Large Cap vs Fixed Income Correlation is 0.0 (Could be coincidence)")

    # 2. Active Risk Metrics
    print("Checking Active Metrics (Beta/TE)...")
    # Need SPY price history. run_analytics_engine caches prices.
    # calculate_active_metrics fetches prices again?
    # It uses fetch_price_history which caches.
    
    metrics = calculate_active_metrics(data, "SPY")
    beta = metrics.get("beta")
    te = metrics.get("te")
    
    if beta == "N/A":
        print("[SKIP] Active Metrics N/A (Insufficient history).")
    else:
        if isinstance(beta, (int, float)):
            log_success(f"Beta vs SPY calculated: {beta:.4f}")
            # Beta should be reasonable (0.5 to 1.5 usually, but could be anything)
        else:
            log_fail(f"Beta returned non-numeric: {beta}")
            
        if isinstance(te, (int, float)) and te >= 0:
            log_success(f"Tracking Error calculated: {te:.4f}%")
        else:
            log_fail(f"Tracking Error invalid: {te}")

    # 3. Drawdown Analysis
    print("Checking Drawdown Logic...")
    twr_curve = _get_daily_twr_curve(data)
    if twr_curve.empty:
        print("[SKIP] TWR Curve empty.")
    else:
        dd_series, max_dd, recovery = compute_drawdown_series(twr_curve)
        
        # Max Drawdown should be min of series
        calc_min = dd_series.min()
        if abs(calc_min - max_dd) < 1e-6:
            log_success(f"Max Drawdown ({max_dd:.2f}%) matches Minimum of Series")
        else:
            log_fail(f"Max Drawdown Mismatch: MaxDD={max_dd}, MinSeries={calc_min}")
            
        # Drawdown should never be > 0 (it's percentage drop from peak)
        if dd_series.max() <= 1e-6:
            log_success("Drawdown Series is always <= 0.0")
        else:
            log_fail(f"Drawdown Series has positive values: Max={dd_series.max()}")

if __name__ == "__main__":
    try:
        test_risk_intelligence()
        print("\n[SUCCESS] audit_04_risk_intelligence.py passed.")
        sys.exit(0)
    except SystemExit as e:
        sys.exit(e.code)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
