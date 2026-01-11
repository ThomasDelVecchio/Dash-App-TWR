import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from data_loader import load_holdings, load_cashflows_external, load_transactions_raw, load_dividends
except ImportError:
    # Fallback if running from a different context
    sys.path.append(os.getcwd())
    from data_loader import load_holdings, load_cashflows_external, load_transactions_raw, load_dividends

def log_success(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def log_warn(msg):
    print(f"[WARN] {msg}")

def test_data_integrity():
    print("\n--- Testing Data Integrity (Input Validation) ---")
    
    # 1. Holdings Load
    print("\nChecking Holdings...")
    try:
        holdings = load_holdings()
        if not holdings.empty:
            required_cols = {"ticker", "shares"}
            if required_cols.issubset(holdings.columns):
                log_success(f"Holdings loaded: {len(holdings)} positions. Columns valid.")
            else:
                log_fail(f"Holdings missing columns. Found: {holdings.columns}, Expected: {required_cols}")
        else:
            log_fail("Holdings DataFrame is empty.")
    except Exception as e:
        log_fail(f"Holdings Load Crash: {e}")

    # 2. Cashflows Load
    print("\nChecking External Cashflows...")
    try:
        cf_ext = load_cashflows_external()
        if not cf_ext.empty:
            required_cols = {"date", "amount"}
            if required_cols.issubset(cf_ext.columns):
                log_success(f"Cashflows loaded: {len(cf_ext)} entries.")
            else:
                log_fail(f"Cashflows missing columns. Found: {cf_ext.columns}")
        else:
            log_warn("External Cashflows empty (Check if intentional).")
    except Exception as e:
        log_fail(f"Cashflows Load Crash: {e}")

    # 3. Transactions Load
    print("\nChecking Transactions...")
    try:
        tx_raw = load_transactions_raw()
        if not tx_raw.empty:
            required_cols = {"date", "ticker", "shares", "amount"} 
            # Note: columns might vary slightly in implementation, adjusting based on usage
            if {"date", "ticker", "shares"}.issubset(tx_raw.columns):
                log_success(f"Transactions loaded: {len(tx_raw)} trades.")
            else:
                log_fail(f"Transactions missing core columns. Found: {tx_raw.columns}")
        else:
            log_warn("Transactions empty.")
    except Exception as e:
        log_fail(f"Transactions Load Crash: {e}")

    # 4. Dividends Load
    print("\nChecking Dividends...")
    try:
        dividends = load_dividends()
        if not dividends.empty:
             log_success(f"Dividends loaded: {len(dividends)} entries.")
        else:
            log_warn("Dividends empty.")
    except Exception as e:
        log_fail(f"Dividends Load Crash: {e}")

    # 5. Shares Reconciliation (Holdings vs Tx)
    print("\nReconciling Shares (Holdings vs Transactions)...")
    try:
        if not tx_raw.empty and not holdings.empty:
            tx_shares = tx_raw.groupby("ticker")["shares"].sum()
            
            mismatches = []
            for _, row in holdings.iterrows():
                t = row["ticker"]
                if t == "CASH": continue
                
                h_shares = row["shares"]
                tx_s = tx_shares.get(t, 0.0)
                
                # Tolerance for float comparison
                if abs(h_shares - tx_s) > 1e-4:
                    mismatches.append(f"{t}: Holdings={h_shares}, TxSum={tx_s}")
            
            if not mismatches:
                log_success("All share counts match between Holdings and Transaction history.")
            else:
                log_fail(f"Share Count Mismatches: {mismatches}")
        else:
            log_warn("Skipping reconciliation (Missing data).")
    except Exception as e:
        log_fail(f"Reconciliation Crash: {e}")

if __name__ == "__main__":
    test_data_integrity()
    print("\n[SUCCESS] audit_00_data_integrity.py passed.")
