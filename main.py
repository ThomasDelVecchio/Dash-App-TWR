#!/usr/bin/env python3
import pandas as pd
from generate_report import build_report
from portfolio_engine import run_engine

def run_console_report():
    """
    Runs the engine and prints the summary tables to the console (like the original main1.py).
    """
    # Run engine
    twr_df, sec_table, class_df, pv, twr_since_inception, twr_since_inception_annualized, pl_since_inception = run_engine()

    # ---------- PRINT PORTFOLIO TWR ----------
    print("\n========== PORTFOLIO TWR (Time-Weighted Return) ==========\n")
    for _, row in twr_df.iterrows():
        h = row["Horizon"]
        v = row["Return"]
        if pd.isna(v):
            print(f"{h:>3}: insufficient data")
        else:
            print(f"{h:>3}: {v:>8.4%}")
    
    # Add Since Inception to TWR display
    si_v = twr_since_inception_annualized if pd.notna(twr_since_inception_annualized) else twr_since_inception
    if pd.isna(si_v):
        print(f" SI: insufficient data")
    else:
        print(f" SI: {si_v:>8.4%}")
    print("\n==========================================================\n")

    # ---------- PRINT P/L SUMMARY ----------
    print("========== PORTFOLIO P/L SUMMARY ==========\n")
    print(f"P/L Since Inception: {pl_since_inception:,.2f}")
    print("\n===========================================\n")

    # ---------- PRINT SECURITY-LEVEL TABLE ----------
    if not sec_table.empty:
        # Hide meta columns for console output
        clean_sec = sec_table[[c for c in sec_table.columns if not str(c).startswith("meta_")]]
        
        print("========== SECURITY-LEVEL MODIFIED DIETZ RETURNS (Money-Weighted) ==========\n")
        with pd.option_context("display.float_format", lambda x: f"{x:0.4f}"):
            print(clean_sec.to_string(index=False))
        print("\n==========================================================================\n")
    else:
        print("No valid security-level Modified Dietz returns could be computed.\n")
        return

    # ---------- PRINT ASSET-CLASS TABLE ----------
    if not class_df.empty:
        # Hide meta columns for console output
        clean_class = class_df[[c for c in class_df.columns if not str(c).startswith("meta_")]]
        
        print("========== ASSET CLASS MODIFIED DIETZ RETURNS (Money-Weighted) ==========\n")
        with pd.option_context("display.float_format", lambda x: f"{x:0.4%}"):
            print(clean_class.to_string(index=False))
        print("\n==========================================================================\n")
    else:
        print("No valid asset-class Modified Dietz returns could be computed.\n")


if __name__ == "__main__":
    import sys
    
    # Simple CLI: no args = build report. "console" = print tables.
    if len(sys.argv) > 1 and sys.argv[1] == "console":
        run_console_report()
    else:
        print("Generating DOCX/PDF report...")
        build_report()
