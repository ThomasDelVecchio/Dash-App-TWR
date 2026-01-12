import subprocess
import os
import sys
import json
from datetime import datetime

# Audit Scripts in order with descriptions
SCRIPTS = [
    {
        "file": "audit_00_data_integrity.py",
        "name": "Data Integrity & Validation",
        "objective": "Ensure all input data sources are clean, structured, and consistent before processing.",
        "methodology": [
            "Validate Holdings, Cashflows, Transactions, and Dividends columns.",
            "Reconcile Share Counts (Holdings vs Transaction Sums)."
        ]
    },
    {
        "file": "audit_01_math_core.py",
        "name": "Math Core Physics",
        "objective": "Verify the fundamental financial mathematics engine accuracy.",
        "methodology": [
            "Stress test Time-Weighted Return (TWR) calculation (Standard, Fundings, Withdrawals).",
            "Verify Geometric Chaining Invariant ((1+r1)*(1+r2)-1).",
            "Verify Modified Dietz logic and Component Identity.",
            "Confirm Dividend Capture in Money-Weighted Returns.",
            "Test Asset Class aggregation logic.",
            "Confirm P/L 'Sum of Parts' equality."
        ]
    },
    {
        "file": "audit_02_horizon_gating.py",
        "name": "Horizon Gating & Logic",
        "objective": "Ensure returns are only calculated for valid time periods.",
        "methodology": [
            "Test 'Too New' assets to ensure returns are suppressed (NaN).",
            "Verify MTD/1M/1W availability for new assets.",
            "Verify security-level gating rules.",
            "Verify all horizons: 6M, YTD, 3Y, 5Y, SI.",
            "Boundary Testing (364 vs 365 days) for Annualization."
        ]
    },
    {
        "file": "audit_03_pl_attribution.py",
        "name": "P/L Attribution Ledger",
        "objective": "Prove that no money is disappearing from the portfolio.",
        "methodology": [
            "Calculate Portfolio P/L (Top-Down) using Net External Flows.",
            "Calculate Sum of Ticker P/Ls (Bottom-Up) using Net Internal Flows + Income.",
            "Assert the Gap between Portfolio P/L and Sum(Tickers) is negligible (< $15.00)."
        ]
    },
    {
        "file": "audit_04_risk_intelligence.py",
        "name": "Risk Intelligence",
        "objective": "Validate advanced risk metrics and correlation data.",
        "methodology": [
            "Check Correlation Matrix integrity (Diagonal=1.0, Symmetric).",
            "Verify Active Risk Metrics (Beta, Tracking Error) vs SPY.",
            "Verify Drawdown Series logic (Max Drawdown matches series minimum)."
        ]
    },
    {
        "file": "audit_05_consistency_matrix.py",
        "name": "UI Consistency Matrix",
        "objective": "Ensure the Dashboard tells a consistent story across all views.",
        "methodology": [
            "Compare 'Snapshot' TWR vs 'Cumulative Return Chart' final value.",
            "Compare 'Snapshot' P/L vs 'Horizon Analysis Table' P/L.",
            "Compare 'Growth of Capital Table' Total vs 'Snapshot' P/L.",
            "Compare 'Attribution Summary' Sum vs 'Snapshot' TWR."
        ]
    },
    {
        "file": "audit_06_gips_scorecard.py",
        "name": "GIPS Compliance Scorecard",
        "objective": "Rate the system against key GIPS requirements.",
        "methodology": [
            "Check 1: External Flows distinct from Trade Flows.",
            "Check 2: Portfolio Return calculated using TWR.",
            "Check 3: Segment Returns calculated using Money-Weighted (Modified Dietz).",
            "Check 4: Daily Valuation enabled.",
            "Check 5: No Annualization for periods < 1 Year."
        ]
    },
    {
        "file": "audit_07_stress_test.py",
        "name": "System Stress Tests",
        "objective": "Verify system stability under extreme conditions.",
        "methodology": [
            "Test Gapped Data (Missing Prices).",
            "Test Negative Equity (Short/Leverage) handling.",
            "Test Trillion Dollar Scale.",
            "Test Day 1 Inception (Deposit+Buy).",
            "Test Weekend Flow Snapping."
        ]
    },
    {
        "file": "audit_08_time_machine.py",
        "name": "Time Machine (Point-in-Time)",
        "objective": "Verify historical accuracy and absence of look-ahead bias.",
        "methodology": [
            "Run Engine with historical end_date (e.g., T-14).",
            "Compare Historical PV vs Live PV at that date.",
            "Ensure no future tickers leak into historical holdings."
        ]
    },
    {
        "file": "audit_09_custom_report.py",
        "name": "Custom Report Logic",
        "objective": "Validate 'Curve Slicing' methodology for arbitrary time periods.",
        "methodology": [
            "Load normalized TWR Curve.",
            "Calculate return for random windows using Slice Method.",
            "Calculate return for same windows using Rigorous Engine.",
            "Assert they match within 1bps."
        ]
    },
    {
        "file": "audit_10_cash.py",
        "name": "Cash Yield & GIPS Logic",
        "objective": "Verify Cash is treated as an Asset Class with performance yield, not just a holding tank.",
        "methodology": [
            "Validate 'Yield = Interest / Time-Weighted Balance' math.",
            "Verify Variable Balance weighting logic.",
            "Confirm Interest is treated as Return (Growth) in TWR, not External Flow."
        ]
    },
    {
        "file": "audit_11_single_ticker_consistency.py",
        "name": "Single-Ticker Consistency",
        "objective": "Verify that Asset Class returns strictly match Ticker returns when the class contains only that ticker.",
        "methodology": [
            "Scan for single-ticker asset classes (active or exited).",
            "Compare Asset Class Return vs Ticker Return for all horizons (1W, 1M, YTD, etc.).",
            "Assert mathematical equality to ensure start-date alignment."
        ]
    },
    {
        "file": "audit_12_holiday_lookback.py",
        "name": "Holiday Lookback Logic",
        "objective": "Verify Backward Snap logic for rolling horizons to handle holidays correctly.",
        "methodology": [
            "Simulate PV series with holiday gap exactly on lookback target date.",
            "Compare Forward Snap (Bug) vs Backward Snap (Fix) returns.",
            "Verify 1W and 3M rolling horizons."
        ]
    },
    {
        "file": "audit_13_annualization.py",
        "name": "Universal Annualization Logic",
        "objective": "Verify that returns are only annualized when duration strictly exceeds 1.0 years (365.25 days).",
        "methodology": [
            "Verify 180 Days -> Cumulative (No Annualization).",
            "Verify 365 Days -> Cumulative (Boundary Condition).",
            "Verify 366 Days -> Annualized (Leap Year Edge Case).",
            "Verify 3 Years -> Annualized correctly."
        ]
    },
    {
        "file": "audit_14_wrapper_dates.py",
        "name": "Wrapper Horizon Analysis",
        "objective": "Verify proper date handling in the Dash Wrapper layer.",
        "methodology": [
            "Verify Holiday Snap logic propagates to UI horizon analysis.",
            "Verify week/month boundaries."
        ]
    },
    {
        "file": "audit_15_consistency_sweep.py",
        "name": "Consistency Sweep",
        "objective": "Deep verification of Date Logic across all layers.",
        "methodology": [
            "Verify get_horizon_target_date logic.",
            "Verify get_effective_anchor_date logic.",
            "Verify is_market_holiday."
        ]
    }
]

def generate_html_report(results, report_path):
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["passed"])
    failed_tests = total_tests - passed_tests
    health_score = int((passed_tests / total_tests) * 100) if total_tests > 0 else 0
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forensic Audit Report</title>
    <style>
        :root {{
            --primary-color: #00ff9d;
            --background-color: #0a0a0a;
            --card-bg: #1a1a1a;
            --text-color: #e0e0e0;
            --danger-color: #ff4757;
            --success-color: #2ed573;
            --border-color: #333;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            margin: 0;
            padding: 2rem;
            line-height: 1.6;
        }}
        .container {{
            max_width: 1200px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
        }}
        h1, h2, h3 {{ margin: 0; }}
        h1 {{ color: var(--primary-color); }}
        .meta {{ text-align: right; color: #888; }}
        
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        .status-card {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            text-align: center;
        }}
        .status-card.pass {{ border-color: var(--success-color); }}
        .status-card.fail {{ border-color: var(--danger-color); }}
        
        .audit-module {{
            background: var(--card-bg);
            border-radius: 8px;
            margin-bottom: 1.5rem;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        .module-header {{
            padding: 1rem 1.5rem;
            background: rgba(255, 255, 255, 0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }}
        .status-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
        }}
        .status-badge.pass {{ background: rgba(46, 213, 115, 0.2); color: var(--success-color); }}
        .status-badge.fail {{ background: rgba(255, 71, 87, 0.2); color: var(--danger-color); }}
        
        .module-content {{
            padding: 1.5rem;
            border-top: 1px solid var(--border-color);
        }}
        .methodology-list {{
            background: rgba(0, 0, 0, 0.2);
            padding: 1rem 2rem;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #aaa;
        }}
        .log-output {{
            background: #000;
            padding: 1rem;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            color: #ccc;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #333;
            margin-top: 1rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Forensic Audit Report</h1>
                <div style="opacity: 0.8;">Executive Integrity Dashboard</div>
            </div>
            <div class="meta">
                <div>Date: {timestamp}</div>
                <div>Version: 3.0 (Modular Opus Suite)</div>
            </div>
        </header>

        <section class="executive-summary">
            <h2>Executive Summary</h2>
            <p>
                This report certifies the mathematical integrity, regulatory compliance, and data consistency 
                of the Portfolio Analytics Engine. All tests are performed against live and synthetic datasets 
                to ensure maximum reliability.
            </p>
            
            <div class="status-grid">
                <div class="status-card {'pass' if health_score == 100 else 'fail'}">
                    <h3>System Health</h3>
                    <div style="font-size: 2.5rem; font-weight: bold; color: var(--primary-color);">
                        {health_score}%
                    </div>
                </div>
                <div class="status-card pass">
                    <h3>Modules Passed</h3>
                    <div style="font-size: 2.5rem; font-weight: bold; color: var(--success-color);">
                        {passed_tests}
                    </div>
                </div>
                <div class="status-card {'pass' if failed_tests == 0 else 'fail'}">
                    <h3>Modules Failed</h3>
                    <div style="font-size: 2.5rem; font-weight: bold; color: {'var(--border-color)' if failed_tests == 0 else 'var(--danger-color)'};">
                        {failed_tests}
                    </div>
                </div>
            </div>
        </section>

        <section class="audit-section">
            <h2>Detailed Audit Logs</h2>
    """
    
    for res in results:
        status_class = "pass" if res["passed"] else "fail"
        status_text = "PASSED" if res["passed"] else "FAILED"
        name = res["meta"]["name"]
        obj = res["meta"]["objective"]
        file_name = res["meta"]["file"]
        
        # Methodology List
        method_html = "<ul class='methodology-list'>"
        for m in res["meta"]["methodology"]:
            method_html += f"<li>• {m}</li>"
        method_html += "</ul>"
        
        # Output Logs
        log_content = res["stdout"]
        if res["stderr"]:
            log_content += "\n\n[STDERR]\n" + res["stderr"]
            
        html += f"""
            <div class="audit-module">
                <div class="module-header">
                    <h3>{name} <span style="font-size: 0.8rem; font-weight: normal; opacity: 0.7;">({file_name})</span></h3>
                    <span class="status-badge {status_class}">{status_text}</span>
                </div>
                <div class="module-content">
                    <p><strong>Objective:</strong> {obj}</p>
                    {method_html}
                    <div class="log-output">{log_content}</div>
                </div>
            </div>
        """
        
    html += """
        </section>

        <footer>
            Generated by Automated Forensic Audit Suite v3.0
        </footer>
    </div>
</body>
</html>
    """
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    return report_path

def run_audits():
    results = []
    audit_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Starting Forensic Audit Suite at {datetime.now()}")
    print("-" * 50)

    all_passed = True

    for item in SCRIPTS:
        script_file = item["file"]
        script_name = item["name"]
        print(f"Running {script_file} ({script_name})...")
        script_path = os.path.join(audit_dir, script_file)
        
        try:
            # Run script and capture output
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                check=False 
            )
            
            passed = (result.returncode == 0)
            if not passed:
                all_passed = False
                
            results.append({
                "meta": item,
                "passed": passed,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
            
            status = "PASS" if passed else "FAIL"
            print(f"  -> {status}")
            
        except Exception as e:
            all_passed = False
            results.append({
                "meta": item,
                "passed": False,
                "stdout": "",
                "stderr": str(e)
            })
            print(f"  -> CRASH: {e}")

    # Generate Report
    report_path = os.path.join(audit_dir, "FORENSIC_AUDIT_REPORT.html")
    generate_html_report(results, report_path)
            
    print("-" * 50)
    print(f"Audit Complete. Report saved to: {report_path}")
    
    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    run_audits()
