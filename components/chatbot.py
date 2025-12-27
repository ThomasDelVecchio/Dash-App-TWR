import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import re
import difflib
import pandas as pd
import numpy as np
from datetime import datetime
import dash_wrappers as dw
import config
from data_loader import fetch_price_history
from pages.help_content import HELP_TOPICS

# ============================================================
# CONFIG & KNOWLEDGE BASE
# ============================================================

COMPONENT_REGISTRY = {
    "cumulative_return": {
        "canonical_name": "Cumulative Return Chart",
        "type": "chart",
        "page": "performance",
        "description": "Shows the percentage growth of the portfolio over time compared to selected benchmarks (e.g., S&P 500).",
        "interpretation": "A rising line indicates positive growth. If the Portfolio line is above the Benchmark lines, you are outperforming the market. It uses Time-Weighted Return (TWR) to filter out the noise of deposits/withdrawals.",
        "common_questions": ["cumulative return", "growth chart", "performance chart", "vs spy", "line chart"]
    },
    "excess_return": {
        "canonical_name": "Excess Return Chart",
        "type": "chart",
        "page": "performance",
        "description": "Displays the difference (spread) between the portfolio's return and benchmark returns over various time horizons.",
        "interpretation": "Positive bars (Green/Blue) mean you beat the benchmark (Alpha). Negative bars mean you underperformed. This helps isolate skill from market movement.",
        "common_questions": ["excess return", "alpha", "outperformance", "underperformance", "bar chart"]
    },
    "horizon_return": {
        "canonical_name": "Horizon Returns Table",
        "type": "table",
        "page": "performance",
        "description": "Lists Time-Weighted Returns (TWR) for the total portfolio, asset classes, and individual tickers over standard periods (1 Month, YTD, 1 Year, etc.).",
        "interpretation": "Green values are positive returns. Compare 'YTD' (Year-to-Date) to see this year's performance. Compare 'SI' (Since Inception) for long-term results.",
        "common_questions": ["horizon return", "return table", "performance table", "returns", "performance highlights"]
    },
    "growth_of_capital": {
        "canonical_name": "Growth of Invested Capital",
        "type": "chart",
        "page": "performance",
        "description": "Visualizes the total value of the portfolio (Stacked Area) versus the net cash invested (Dashed Line).",
        "interpretation": "The colored area is your money working for you. The dashed line is your own contributions. The gap between the Line and the Top of the Area represents your cumulative investment profit.",
        "common_questions": ["growth of invested capital", "invested capital", "cash vs value", "money weighted", "investment summary"]
    },
    "portfolio_value": {
        "canonical_name": "Portfolio Value (Mountain)",
        "type": "chart",
        "page": "overview",
        "description": "Shows the total market value of the portfolio in dollars over time.",
        "interpretation": "This is your account balance history. Steep drops may indicate market crashes or withdrawals. Steep rises indicate rallies or deposits.",
        "common_questions": ["portfolio value", "mountain chart", "total value", "account balance"]
    },
    "asset_allocation": {
        "canonical_name": "Asset Allocation",
        "type": "chart",
        "page": "allocations",
        "description": "Breakdown of the portfolio by major asset class (Equity, Fixed Income, etc.).",
        "interpretation": "Checks diversification. If one slice is too large, you may be over-exposed to that risk. The 'Bar Chart' next to it compares this Actual allocation to your Target.",
        "common_questions": ["asset allocation", "pie chart", "holdings breakdown"]
    },
    "sector_allocation": {
        "canonical_name": "Sector Allocation",
        "type": "chart",
        "page": "allocations",
        "description": "Look-through analysis that maps ETFs to their underlying economic sectors (e.g., Technology, Healthcare).",
        "interpretation": "Reveals hidden concentration. You might own 5 different ETFs, but if they are all Tech-heavy, this chart will show a large 'Technology' bar.",
        "common_questions": ["sector allocation", "sectors", "industries", "exposure", "look-through", "look through"]
    },
    "allocation_history": {
        "canonical_name": "Allocation History",
        "type": "chart",
        "page": "allocations",
        "description": "Visualizes how your portfolio's asset allocation has changed over time as a percentage of total value.",
        "interpretation": "The stacked areas show the weight of each asset class. Changes occur due to market performance (drift) or your trading activity (rebalancing).",
        "common_questions": ["allocation history", "historical allocation", "allocation over time", "weight history"]
    },
    "attribution": {
        "canonical_name": "Daily Attribution",
        "type": "chart",
        "page": "attribution",
        "description": "Breaks down daily changes in portfolio value into 'External Flows' (Deposits/Withdrawals) and 'Market Effect' (Performance). Includes drill-down capability.",
        "interpretation": "Blue bars are market gains/losses. Click a bar to see the 'Daily Breakdown' by Asset Class below. Red dotted line is cumulative P/L.",
        "common_questions": ["attribution", "daily change", "market effect", "p/l chart", "drill down"]
    },
    "risk_return": {
        "canonical_name": "Risk vs Return",
        "type": "chart",
        "page": "risk",
        "description": "Scatter plot comparing the Volatility (Risk) vs Expected Return of different asset classes.",
        "interpretation": "Items in the top-left are ideal (High Return, Low Risk). Items in the bottom-right are poor (Low Return, High Risk).",
        "common_questions": ["risk chart", "volatility", "scatter plot", "risk return"]
    },
    "risk_diversification": {
        "canonical_name": "Risk & Diversification Table",
        "type": "table",
        "page": "risk",
        "description": "Summary table highlighting concentration risks, such as your top 3 holdings and largest asset class exposure.",
        "interpretation": "Use this to spot if you are too concentrated. 'Top 3 holdings %' should generally be monitored if it exceeds 15-20%.",
        "common_questions": ["risk table", "diversification table", "concentration", "top 3"]
    },
    "flows_summary": {
        "canonical_name": "YTD Flows Summary",
        "type": "table",
        "page": "flows",
        "description": "Summary of deposits, withdrawals, buys, sells, and dividends for the current year.",
        "interpretation": "Tracks money moving in and out of the portfolio (External) and trades within the portfolio (Internal).",
        "common_questions": ["flows summary", "flows", "deposits", "withdrawals", "dividends"]
    },
    "flows_chart": {
        "canonical_name": "Internal Flows Chart",
        "type": "chart",
        "page": "flows",
        "description": "Bar chart showing the net internal flows (Buys vs Sells) for each asset class.",
        "interpretation": "Positive bars (Green) indicate you are a net buyer of that asset class. Negative bars (Red) indicate net selling.",
        "common_questions": ["flows chart", "internal flows", "buying vs selling", "activity"]
    },
    "contribution_schedule": {
        "canonical_name": "Monthly Contribution Schedule",
        "type": "table",
        "page": "allocations",
        "description": "Calculates recommended monthly contributions to rebalance the portfolio over time.",
        "interpretation": "Shows which assets are 'Underweight' and how much of your monthly deposit should go to each to close the gap.",
        "common_questions": ["contribution schedule", "monthly contribution", "rebalance", "what to buy"]
    },
    "portfolio_simulator": {
        "canonical_name": "20-Year Projections",
        "type": "chart",
        "page": "risk", 
        "description": "Projects future portfolio value based on different return assumptions and monthly contributions using interactive sliders.",
        "interpretation": "Adjust the 'Expected Return' and 'Monthly Contribution' sliders to see how your portfolio value could grow over time. Solid lines are lump sum only; Dashed lines include contributions.",
        "common_questions": ["simulator", "projection", "future value", "forecast", "interactive projections"]
    },
    "trade_lab": {
        "canonical_name": "What If Trade Lab",
        "type": "tool",
        "page": "trade_lab",
        "description": "A sandbox environment to simulate the impact of hypothetical trades on your portfolio's future performance using Monte Carlo simulations.",
        "interpretation": "Enter a ticker, side (Buy/Sell), and amount to see how it changes your probability curves. The 'Current' line shows your path without the trade, and the 'Hypothetical' line shows the path with the trade.",
        "common_questions": ["trade lab", "what if", "hypothetical trade", "scenario"]
    },
    "correlation_matrix": {
        "canonical_name": "Rolling Correlation Matrix",
        "type": "chart",
        "page": "risk",
        "description": "A heatmap showing the 90-day rolling correlation between your top 10 holdings.",
        "interpretation": "Red (close to +1) means assets move together (high risk). Blue (close to -1) means they move inversely (hedge). Faint colors (close to 0) mean they are uncorrelated (diversified).",
        "common_questions": ["correlation", "heatmap", "matrix", "diversification check", "rolling correlation"]
    },
    "asset_allocation_simulator": {
        "canonical_name": "Asset Allocation Simulator",
        "type": "tool",
        "page": "risk",
        "description": "Interactive sliders to adjust target weights of asset classes and see the impact on Portfolio Risk/Return profile.",
        "interpretation": "Move the sliders to change weights. The gauges show how the Expected Return and Volatility would change compared to your current portfolio.",
        "common_questions": ["asset allocation simulator", "weight simulator", "rebalance simulator", "allocation sliders"]
    },
    "morning_brief": {
        "canonical_name": "Morning Brief",
        "type": "card",
        "page": "overview",
        "description": "An AI-generated summary of your portfolio's recent performance, key movers, and market context.",
        "interpretation": "Provides a quick narrative update so you don't have to analyze every chart manually. It highlights daily returns and significant changes.",
        "common_questions": ["morning brief", "ai summary", "summary", "brief", "what happened"]
    },
    "holdings_table": {
        "canonical_name": "Holdings Table",
        "type": "table",
        "page": "holdings",
        "description": "Detailed list of all current positions, including shares, market value, weight, and performance.",
        "interpretation": "Use this to drill down into specific tickers. Check 'Weight' for concentration and 'Unrealized P/L' for tax planning.",
        "common_questions": ["holdings", "positions", "stocks", "etfs", "current portfolio"]
    },
    "ticker_allocation": {
        "canonical_name": "Ticker Allocation",
        "type": "chart",
        "page": "holdings",
        "description": "Pie and Bar charts showing the breakdown of your portfolio by individual tickers.",
        "interpretation": "Visualizes your largest single positions. The Bar chart compares each ticker's actual weight to its target weight (if defined).",
        "common_questions": ["ticker allocation", "ticker breakdown", "largest positions", "top holdings"]
    },
    "performance_highlights": {
        "canonical_name": "Performance Highlights",
        "type": "table",
        "page": "overview",
        "description": "Table showing the best and worst performing tickers for the last Month and Day.",
        "interpretation": "Quickly identifies what is moving your portfolio. High returns here drive the daily performance.",
        "common_questions": ["performance highlights", "highlights", "best performers", "worst performers", "movers"]
    },
    "asset_class_table": {
        "canonical_name": "Asset Class Allocation Table",
        "type": "table",
        "page": "allocations",
        "description": "Detailed table comparing Actual vs Target allocation for each asset class, including the exact dollar gap.",
        "interpretation": "Use the 'Delta %' column to see how far you are from your targets. Positive Delta means Overweight, Negative means Underweight.",
        "common_questions": ["asset class table", "allocation table", "target gaps", "delta"]
    },
    "efficiency_scores": {
        "canonical_name": "Efficiency Scores (Sharpe/Sortino)",
        "type": "card",
        "page": "performance",
        "description": "Risk-adjusted performance metrics. Sharpe measures return per unit of total risk. Sortino measures return per unit of downside risk.",
        "interpretation": "Higher is better. A Sharpe > 1.0 is considered good. Sortino > 2.0 implies high returns with minimal downside crashes.",
        "common_questions": ["efficiency scores", "sharpe", "sortino", "risk adjusted"]
    },
    "drawdown_chart": {
        "canonical_name": "Drawdown Analysis (Underwater)",
        "type": "chart",
        "page": "risk",
        "description": "Visualizes the percentage decline from the portfolio's historical peak (High-Water Mark) over time.",
        "interpretation": "Shows how deep your losses go during market corrections and how long it takes to recover to new highs.",
        "common_questions": ["drawdown", "underwater", "max drawdown", "recovery", "losses"]
    },
    "active_strategy": {
        "canonical_name": "Active Strategy vs Benchmarks",
        "type": "table",
        "page": "attribution",
        "description": "Comparison of portfolio sensitivity (Beta) and active deviation (Tracking Error) against major benchmarks like SPY.",
        "interpretation": "Beta > 1 means higher volatility than the market. High Tracking Error means you are deviating significantly from the benchmark (Active Management).",
        "common_questions": ["active strategy", "beta", "tracking error", "active risk", "benchmark comparison"]
    }
}

EXPLANATIONS = {
    "twr": (
        "**Time-Weighted Return (TWR)** measures the compound rate of growth of the portfolio, "
        "eliminating the distorting effects of cash inflows and outflows. \n\n"
        "It is calculated by chaining daily returns: `(1 + r1) * (1 + r2) * ... - 1`. \n\n"
        "External flows (deposits/withdrawals) are treated as occurring at the **start of the day** "
        "for GIPS compliance."
    ),
    "dietz": (
        "**Modified Dietz** is a money-weighted return method used for individual securities "
        "and asset classes. \n\n"
        "Formula: `R = (V1 - V0 - C) / (V0 + W*C)` \n"
        "Where: \n"
        "- `V1`: End Value\n"
        "- `V0`: Start Value\n"
        "- `C`: Net External Flows\n"
        "- `W`: Time-weighting factor for flows"
    ),
    "contribution": (
        "The **Contribution Schedule** calculates how much capital is needed to bring underweight positions "
        "up to their target allocation. \n\n"
        "It allocates the monthly contribution amount proportionally to the 'gap' (Target $ - Current $) "
        "of each underweight asset."
    ),
    "attribution": (
        "**Attribution** breaks down the change in Portfolio Value (ΔPV) into two components: \n"
        "1. **External Flows**: Deposits or withdrawals.\n"
        "2. **Market Effect**: Investment performance (Price changes + Dividends)."
    ),
    "profit": "Profit/Loss (P/L) is calculated as `Mark-to-Market Value - Net Invested Capital`. It represents the actual economic gain or loss in dollars.",
    "return": "See 'TWR' for Portfolio Return or 'Modified Dietz' for Asset Class/Ticker Return.",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

SYNONYMS = {
    # Ranking
    "biggest gainer": "highest return ticker",
    "top gainer": "highest return ticker",
    "best gainer": "highest return ticker",
    "biggest loser": "lowest return ticker",
    "top loser": "lowest return ticker",
    "worst loser": "lowest return ticker",
    "up the most": "highest return ticker",
    "down the most": "lowest return ticker",
    "best performing": "highest return",
    "worst performing": "lowest return",
    "performing best": "highest return",
    "performing worst": "lowest return",
    "highest performing": "highest return",
    
    # Metrics
    "most allocated": "highest allocation",
    "least allocated": "lowest allocation",
    "biggest position": "highest allocation",
    "largest position": "highest allocation",
    "smallest position": "lowest allocation",
    "most profit": "highest pl",
    "least profit": "lowest pl",
    "most money": "highest pl",
    "highest contribution": "highest pl",
    "grew the most": "highest return",
    "grew the least": "lowest return",
    "over allocated": "overweight", # Added
    "under allocated": "underweight", # Added
    "allocations": "allocation",
    "holding": "allocation",
    "exposure": "allocation",
    "money made": "pl",
    "profit": "pl",
    "loss": "pl",
    "gain": "pl",
    "p/l": "pl", # Added
    "growth": "return",
    "value": "market_value",
    "worth": "market_value",
    "balance": "market_value",
    "dividends": "income",
    "dividend": "income",
    "deposits": "flows",
    "withdrawals": "flows",
    "contributions": "flows",
    "net flows": "flows",
    "added": "flows",
    "invested": "net_invested",
    "buys": "buys",
    "sells": "sells",
    "purchases": "buys",
    "sales": "sells",
    "transactions": "transactions",
    "trades": "transactions",

    # Entities
    "total portfolio": "portfolio",
    "whole portfolio": "portfolio",
    "my portfolio": "portfolio",
    "total account": "portfolio",
    "account": "portfolio",
    "cash": "CASH", # Special case

    # Comparisons
    "vs": "benchmark",
    "compared to": "benchmark",
    "against": "benchmark",
    "beating": "excess return",
    "losing to": "excess return",
    "alpha": "excess return",
}

def normalize_text(text):
    """Normalize synonyms and casual phrasing to canonical terms."""
    text = text.lower()
    # Sort by length descending to match longest phrases first
    sorted_syns = sorted(SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True)
    
    for syn, canonical in sorted_syns:
        if syn in text:
            text = text.replace(syn, canonical)
            
    # Cleanup redundant phrases
    text = text.replace("performing performing", "performing")
    text = text.replace("return return", "return")
    text = text.replace("allocation allocation", "allocation")
    
    return text

def parse_horizon(text):
    """
    Parses natural language text to find a valid horizon code.
    Defaults to '1D' if no specific horizon is found, but returns None if text implies 'current'/snapshot 
    without a specific period intent (handled by caller).
    """
    text = text.lower()
    
    # Map phrases to canonical codes (Order matters: longer phrases first)
    horizon_map = [
        ("today", "1D"), ("1d", "1D"), ("daily", "1D"),
        ("week to date", "1W"), ("week", "1W"), ("1w", "1W"),
        ("month to date", "MTD"), ("mtd", "MTD"), ("this month", "MTD"),
        ("one month", "1M"), ("1 month", "1M"), ("1m", "1M"), ("last month", "1M"),
        ("quarter", "3M"), ("3 months", "3M"), ("3m", "3M"),
        ("half year", "6M"), ("6 months", "6M"), ("6m", "6M"),
        ("year to date", "YTD"), ("ytd", "YTD"), ("this year", "YTD"),
        ("one year", "1Y"), ("1 year", "1Y"), ("1y", "1Y"), ("last year", "1Y"),
        ("3 years", "3Y"), ("3y", "3Y"),
        ("5 years", "5Y"), ("5y", "5Y"),
        ("inception", "SI"), ("since inception", "SI"), ("all time", "SI"), ("si", "SI")
    ]
    
    for phrase, code in horizon_map:
        if phrase in text:
            return code
            
    return None 

def extract_metric(text):
    """
    Identifies the metric the user is asking about.
    Returns: 'return', 'pl', 'allocation', 'value', 'flows', 'transaction', 'net_invested', 'sharpe', 'sortino', 'drawdown', 'beta', 'te', or None.
    """
    text = text.lower()
    if any(x in text for x in ["sharpe", "sharp ratio"]): return "sharpe"
    if any(x in text for x in ["sortino"]): return "sortino"
    if any(x in text for x in ["drawdown", "max dd", "underwater"]): return "drawdown"
    if any(x in text for x in ["beta", "sensitivity"]): return "beta"
    if any(x in text for x in ["tracking error", "active risk", "te"]): return "te"
    
    if any(x in text for x in ["return", "performance", "growth"]): return "return"
    if any(x in text for x in ["pl", "profit", "loss", "gain", "money made", "p/l"]): return "pl"
    if any(x in text for x in ["allocation", "weight", "portfolio share", "exposure", "position"]): return "allocation"
    if any(x in text for x in ["value", "worth", "balance", "amount"]): return "value"
    if any(x in text for x in ["buys", "sells", "transactions", "trades"]): return "transaction"
    if any(x in text for x in ["net invested", "net amount", "net in", "total in"]): return "net_invested"
    if any(x in text for x in ["flows", "deposit", "withdrawal", "net flow"]): return "flows"
    return None

def extract_entity(text, data):
    """
    Identifies if the user is asking about a specific Ticker, Asset Class, or Sector.
    Returns (entity_name, entity_type).
    """
    text = text.lower()
    sec_current = data.get("sec_table_current")
    if sec_current is None: return None, None
    
    # 1. Tickers (Exact Match)
    tickers = sec_current["ticker"].unique().tolist()
    # Add common benchmark tickers just in case
    tickers += ["SPY", "QQQ", "IWM", "ACWI", "AGG", "GLD", "BTC"]
    
    for t in tickers:
        # Check for word boundary to avoid partial matches
        if re.search(r"\\b" + re.escape(t.lower()) + r"\\b", text):
            return t, "ticker"
            
    # 2. Asset Classes
    # Normalize map (display name -> matchable key)
    # The keys in sec_current are formal.
    ac_list = sec_current["asset_class"].unique().tolist()
    # Common variations
    ac_map = {
        "gold": "Gold / Precious Metals",
        "precious metals": "Gold / Precious Metals",
        "crypto": "Digital Assets",
        "digital": "Digital Assets",
        "bitcoin": "Digital Assets",
        "bonds": "US Bonds",
        "fixed income": "Fixed Income",
        "us equity": "US Large Cap", # Approximation
        "stocks": "US Large Cap", # Approximation
        "international": "International Equity",
        "intl": "International Equity",
        "cash": "CASH"
    }
    
    for ac in ac_list:
        if ac.lower() in text:
            return ac, "asset_class"
            
    for keyword, formal in ac_map.items():
        if keyword in text:
            return formal, "asset_class"
            
    # 3. Sectors
    sector_df = data.get("sector_df")
    if sector_df is not None and not sector_df.empty:
        sectors = sector_df["Sector"].unique().tolist()
        for s in sectors:
            if s.lower() in text:
                return s, "sector"
    
    # --- FUZZY MATCHING (Improvement 1) ---
    # If no exact match found, look for close typos.
    # We scan the text for words that might be close to our entities.
    
    # Build Candidate Map: {lowercase_candidate: (canonical_name, type)}
    candidate_map = {}
    
    # Tickers
    for t in tickers:
        candidate_map[t.lower()] = (t, "ticker")
        
    # Asset Classes (Formal)
    for ac in ac_list:
        candidate_map[ac.lower()] = (ac, "asset_class")
        
    # Asset Classes (Synonyms)
    for kw, formal in ac_map.items():
        candidate_map[kw] = (formal, "asset_class")
        
    # Sectors
    if sector_df is not None and not sector_df.empty:
        for s in sector_df["Sector"].unique():
            candidate_map[s.lower()] = (s, "sector")
            
    # Check each word in text (simplistic tokenization)
    words = text.split()
    all_candidates = list(candidate_map.keys())
    
    for word in words:
        # Skip common short words to avoid noise
        if len(word) < 3: continue
        
        matches = difflib.get_close_matches(word, all_candidates, n=1, cutoff=0.6)
        if matches:
            best_match = matches[0]
            canonical, etype = candidate_map[best_match]
            return canonical, etype
                
    return None, None

def analyze_portfolio():
    """
    Analyzes the current portfolio state using dash_wrappers data.
    Returns a markdown-formatted summary of observations.
    """
    try:
        data = dw.get_data()
        if not data:
            return "Portfolio data is currently unavailable."
            
        sec_current = data.get("sec_table_current")
        holdings = data.get("holdings")
        
        if sec_current is None or sec_current.empty:
            return "No current holdings data found."

        observations = []
        
        # 1. CONCENTRATION CHECK
        total_mv = sec_current["market_value"].sum()
        if total_mv > 0:
            sec_current = sec_current.copy()
            sec_current["weight_calc"] = sec_current["market_value"] / total_mv * 100
            
            # Sort
            sorted_pos = sec_current[sec_current["ticker"] != "CASH"].sort_values("weight_calc", ascending=False)
            
            # Top 1
            if not sorted_pos.empty:
                top_1 = sorted_pos.iloc[0]
                if top_1["weight_calc"] > 15.0:
                    observations.append(f"🔴 **High Single-Stock Risk**: **{top_1['ticker']}** makes up **{top_1['weight_calc']:.1f}%** of the portfolio.")
                elif top_1["weight_calc"] > 10.0:
                    observations.append(f"⚠️ **Concentration Note**: **{top_1['ticker']}** is your largest holding at **{top_1['weight_calc']:.1f}%**.")
                    
            # Top 3
            top_3_pct = sorted_pos.head(3)["weight_calc"].sum()
            if top_3_pct > 40.0:
                 observations.append(f"⚠️ **Top Heavy**: Your top 3 holdings account for **{top_3_pct:.1f}%** of total value.")

        # 2. ASSET CLASS BALANCE
        ac_alloc = sec_current.groupby("asset_class")["market_value"].sum() / total_mv * 100
        targets = holdings.groupby("asset_class")["target_pct"].sum()
        
        for ac in ac_alloc.index:
            actual = ac_alloc.get(ac, 0)
            target = targets.get(ac, 0)
            diff = actual - target
            
            if diff > 5.0:
                observations.append(f"🔵 **Overweight**: {ac} is **{actual:.1f}%** (Target: {target:.1f}%).")
            elif diff < -5.0:
                observations.append(f"⚪ **Underweight**: {ac} is **{actual:.1f}%** (Target: {target:.1f}%).")

        # 3. DIVERSIFICATION / RISK PROFILE
        EQUITY_ACS = ["US Large Cap", "US Growth", "US Small Cap", "International Equity", "Emerging Markets"]
        FI_ACS = ["US Bonds", "Fixed Income", "Treasuries"]
        
        equity_pct = ac_alloc[ac_alloc.index.isin(EQUITY_ACS)].sum()
        cash_pct = ac_alloc.get("CASH", 0)
        
        if equity_pct > 80.0:
            observations.append(f"🔥 **Aggressive Profile**: **{equity_pct:.1f}%** Equity exposure implies higher expected volatility.")
        elif equity_pct < 40.0:
            observations.append(f"🛡️ **Conservative Profile**: Low Equity exposure (**{equity_pct:.1f}%**) prioritizes preservation.")
            
        if cash_pct > 15.0:
            observations.append(f"💵 **High Cash Drag**: **{cash_pct:.1f}%** Cash is uninvested.")

        if not observations:
            return "Your portfolio looks **balanced** according to standard checks. No major deviations flagged."
        
        return "### Portfolio Analysis\n\n" + "\n".join([f"- {obs}" for obs in observations])
    except Exception as e:
        return f"Could not complete portfolio analysis: {str(e)}"

# ============================================================
# QUERY HANDLERS
# ============================================================

def handle_ranking_query(text, data, horizon, metric=None):
    """
    Handles questions like:
    - "What is my second best ticker?"
    - "Top 3 asset classes by return"
    - "Worst performing sector"
    """
    text = text.lower()
    
    # 1. Determine Target (Ticker vs Asset Class vs Sector)
    target_type = "ticker" # Default
    if "asset class" in text or "category" in text: target_type = "asset_class"
    elif "sector" in text or "industry" in text: target_type = "sector"
    
    # 2. Determine Metric (Return vs P/L vs Allocation)
    # Use context metric if available and applicable, otherwise default
    if not metric:
        metric = "return"
        if any(x in text for x in ["allocation", "weight", "biggest", "largest", "smallest", "position"]): metric = "allocation"
        if any(x in text for x in ["pl", "profit", "loss", "gain", "money made"]): metric = "pl"
    
    # 3. Determine Direction (Top/Best vs Bottom/Worst)
    ascending = False
    if any(x in text for x in ["lowest", "worst", "bottom", "least", "smallest", "loser"]): 
        ascending = True
        
    # 4. Determine Count/Offset (Top 3, 2nd best)
    # Parse N: "top 3", "5 worst"
    count = 1
    count_match = re.search(r" (?:top|bottom|best|worst) (\d+)", text)
    if count_match:
        count = int(count_match.group(1))
    
    # Parse Offset: "second best", "3rd highest"
    offset = 0
    if "second" in text or "2nd" in text: offset = 1
    elif "third" in text or "3rd" in text: offset = 2
    elif "fourth" in text or "4th" in text: offset = 3
    elif "fifth" in text or "5th" in text: offset = 4
    
    if offset > 0: count = 1 # Specific rank implies single result usually
    
    # --- FETCH DATA ---
    df = pd.DataFrame()
    name_col = ""
    val_col = ""
    fmt_str = ""
    
    sec_current = data.get("sec_table_current")
    
    if target_type == "ticker":
        name_col = "ticker"
        if metric == "allocation":
            df = sec_current[sec_current["ticker"] != "CASH"].copy()
            val_col = "market_value"
            fmt_str = "${:,.2f}"
        elif metric == "pl":
            df = dw.get_ticker_pl_df(data, horizon).reset_index()
            val_col = "pl"
            fmt_str = "${:,.2f}"
        else: # Return
            if horizon in sec_current.columns:
                df = sec_current[sec_current["ticker"] != "CASH"].copy()
                val_col = horizon
                fmt_str = "{:+.2f}%"
    
    elif target_type == "asset_class":
        name_col = "asset_class"
        if metric == "allocation":
             df = sec_current.groupby("asset_class")["market_value"].sum().reset_index()
             val_col = "market_value"
             fmt_str = "${:,.2f}"
        elif metric == "pl":
            # Compute PL for all classes
            results = []
            for ac in sec_current["asset_class"].unique():
                if ac == "CASH": continue
                pl = dw.get_asset_class_pl(data, ac, horizon)
                if pl is not None: results.append({"asset_class": ac, "pl": pl})
            df = pd.DataFrame(results)
            val_col = "pl"
            fmt_str = "${:,.2f}"
        else: # Return
            class_df = data.get("class_df")
            if class_df is not None and horizon in class_df.columns:
                df = class_df.copy()
                val_col = horizon
                fmt_str = "{:+.2f}%"

    elif target_type == "sector":
        sector_df = data.get("sector_df") # Allocation only usually
        name_col = "Sector"
        if metric == "allocation" and not sector_df.empty:
             df = sector_df.copy()
             val_col = "Exposure" # Note: This is %, not $ usually in sector_df
             # dw._prepare_sector_df returns "Exposure" as raw sum of (weight * pct). 
             # Actually dw._prepare_sector_df calculates "Exposure" as the weighted sum of % allocation?
             # Let's check dash_wrappers: "Exposure" is sum(weight_pct * sector_pct/100). So it is % of portfolio.
             fmt_str = "{:.1f}%"
        
        # We don't easily have Sector Return/PL pre-calc. 
        # For this exercise, let's restrict sector queries to allocation or say "not available"
        if metric != "allocation":
            return f"Calculated {metric} metrics for **Sectors** are not currently available in the report."

    # --- PROCESS ---
    if df.empty or val_col not in df.columns:
        return f"Data for {target_type} {metric} ({horizon}) is currently unavailable."
        
    # Sort
    df = df.sort_values(val_col, ascending=ascending)
    
    # Handle Offset (e.g. 2nd best)
    if offset >= len(df):
        return f"There are only {len(df)} {target_type}s available, cannot show rank {offset+1}."
        
    # Slice
    result_df = df.iloc[offset : offset + count]
    
    # Format Output
    metric_name = metric.upper() if metric == "pl" else metric.title()
    dir_name = "Bottom" if ascending else "Top"
    if offset > 0: dir_name = f"#{offset+1}"
    
    lines = [f"**{dir_name} {target_type.title()} by {metric_name} ({horizon})**:"]
    
    for _, row in result_df.iterrows():
        val = row[val_col]
        # Adjust formatting if it's a return (convert decimal to %)
        disp_val = val
        if metric == "return": disp_val = val * 100
        
        val_formatted = fmt_str.format(disp_val)
        
        # Add context if allocation (add % if not already)
        if metric == "allocation" and target_type != "sector":
             total = df[val_col].sum()
             pct = (val / total) * 100
             val_formatted += f" ({pct:.1f}%)"
             
        lines.append(f"- **{row[name_col]}**: {val_formatted}")
        
    return "\n".join(lines)

def handle_entity_query(entity_name, entity_type, text, data, horizon, metric=None):
    """
    Handles specific lookup:
    - "What is my allocation to Gold?"
    - "How much PL in AAPL?"
    - "Return of Tech sector?"
    """
    text = text.lower()
    
    # Determine Metric
    # If metric passed from context/extraction, use it. Else infer.
    if not metric:
        metric = "market_value" # Default to value/allocation
        if "return" in text or "performance" in text: metric = "return"
        elif any(x in text for x in ["pl", "profit", "loss", "gain", "money made"]): metric = "pl"
        elif any(x in text for x in ["allocation", "weight", "portfolio share"]): metric = "allocation"
        elif any(x in text for x in ["value", "worth", "balance", "amount"]): metric = "market_value"
    
    # Normalize 'value' to 'market_value'
    if metric == "value": metric = "market_value"
    
    sec_current = data.get("sec_table_current")
    
    val = None
    fmt = ""
    
    # --- FETCH ---
    if entity_type == "ticker":
        if metric == "return":
            row = sec_current[sec_current["ticker"] == entity_name]
            if not row.empty and horizon in row.columns:
                val = row[horizon].iloc[0] * 100
                fmt = "{:+.2f}%"
        elif metric == "pl":
            pl_df = dw.get_ticker_pl_df(data, horizon)
            if entity_name in pl_df.index:
                val = pl_df.loc[entity_name, "pl"]
                fmt = "${:,.2f}"
        else: # Allocation/Value
            row = sec_current[sec_current["ticker"] == entity_name]
            if not row.empty:
                val = row["market_value"].iloc[0]
                weight = row["weight"].iloc[0] * 100
                fmt = "${:,.2f} (" + f"{weight:.1f}%)"

    elif entity_type == "asset_class":
        if metric == "return":
            class_df = data.get("class_df")
            row = class_df[class_df["asset_class"] == entity_name]
            if not row.empty and horizon in row.columns:
                val = row[horizon].iloc[0] * 100
                fmt = "{:+.2f}%"
        elif metric == "pl":
            val = dw.get_asset_class_pl(data, entity_name, horizon)
            fmt = "${:,.2f}"
        else:
            grp = sec_current[sec_current["asset_class"] == entity_name]
            if not grp.empty:
                val = grp["market_value"].sum()
                total = sec_current["market_value"].sum()
                pct = (val / total) * 100
                fmt = "${:,.2f} (" + f"{pct:.1f}%)"

    elif entity_type == "sector":
        sector_df = data.get("sector_df")
        # Map back to standardized sector name if needed
        # We assume extract_entity returned the canonical name
        row = sector_df[sector_df["Sector"] == entity_name]
        if not row.empty:
             # Sector df only has exposure %
             pct = row["Exposure"].iloc[0]
             val = pct # Just show %
             fmt = "{:.1f}%"
             if metric != "allocation":
                 return f"Only **allocation** data is available for Sectors in this report."
        else:
            val = 0.0
            fmt = "{:.1f}%"

    if val is None:
        return f"Data for **{entity_name}** {metric} ({horizon}) is not available."
        
    return f"**{entity_name}** {metric.title()} ({horizon}): **{fmt.format(val)}**"

def handle_transaction_query(entity_name, entity_type, text, data, horizon):
    """
    Handles transaction-based queries for a specific entity.
    - "What are my buys in VOO"
    - "Show my sells of AAPL"
    - "Net invested in GOOG"
    """
    if entity_type != "ticker":
        return f"Transaction data is only available for specific tickers, not for '{entity_name}'."

    tx_raw = data.get("tx_raw")
    if tx_raw is None or tx_raw.empty:
        return "No transaction data is available."

    # Filter for the specific ticker
    ticker_tx = tx_raw[tx_raw["ticker"].str.upper() == entity_name.upper()].copy()
    if ticker_tx.empty:
        return f"No transactions found for **{entity_name}**."

    # Determine transaction type from the query
    query_type = "all"
    text_lower = text.lower()
    if "buys" in text_lower or "purchases" in text_lower:
        query_type = "buy"
    elif "sells" in text_lower or "sales" in text_lower:
        query_type = "sell"
    elif "net invested" in text_lower or "net amount" in text_lower:
        query_type = "net"

    response_lines = []
    
    if query_type == "buy":
        buys = ticker_tx[ticker_tx["amount"] < 0]
        if buys.empty:
            return f"No buy transactions found for **{entity_name}**."
        response_lines.append(f"**Buy Transactions for {entity_name}**:")
        for _, row in buys.iterrows():
            response_lines.append(f"- {row['date'].strftime('%Y-%m-%d')}: **{row['shares']:,.2f} shares** for **${-row['amount']:,.2f}**")
        total_spent = -buys['amount'].sum()
        response_lines.append(f"**Total Spent:** **${total_spent:,.2f}**")

    elif query_type == "sell":
        sells = ticker_tx[ticker_tx["amount"] > 0]
        if sells.empty:
            return f"No sell transactions found for **{entity_name}**."
        response_lines.append(f"**Sell Transactions for {entity_name}**:")
        for _, row in sells.iterrows():
            response_lines.append(f"- {row['date'].strftime('%Y-%m-%d')}: **{row['shares']:,.2f} shares** for **${row['amount']:,.2f}**")
        total_proceeds = sells['amount'].sum()
        response_lines.append(f"**Total Proceeds:** **${total_proceeds:,.2f}**")

    elif query_type == "net":
        net_invested = -ticker_tx["amount"].sum()
        return f"The net amount invested in **{entity_name}** is **${net_invested:,.2f}**."

    else: # "all" transactions
        response_lines.append(f"**All Transactions for {entity_name}**:")
        for _, row in ticker_tx.iterrows():
            tx_type = "Buy" if row['amount'] < 0 else "Sell"
            abs_amount = abs(row['amount'])
            response_lines.append(f"- {row['date'].strftime('%Y-%m-%d')} ({tx_type}): **{row['shares']:,.2f} shares** for **${abs_amount:,.2f}**")
        net_invested = -ticker_tx["amount"].sum()
        response_lines.append(f"**Net Amount Invested:** **${net_invested:,.2f}**")

    return "\n".join(response_lines)


def handle_portfolio_query(text, data, horizon):
    """
    Handles top-level questions:
    - "Total portfolio value"
    - "Total return SI"
    - "Total deposits"
    """
    text = text.lower()
    
    # Metric
    if any(x in text for x in ["return", "performance", "growth"]):
        # TWR
        twr_df = data.get("twr_df")
        row = twr_df[twr_df["Horizon"] == horizon]
        if not row.empty:
            val = row["Return"].iloc[0] * 100
            return f"**Portfolio Return ({horizon})**: **{val:+.2f}%**"
        # Special case for SI Ann
        if horizon == "SI" and "twr_si" in data:
            val = (data["twr_si_ann"] if pd.notna(data["twr_si_ann"]) else data["twr_si"]) * 100
            return f"**Portfolio Return (SI)**: **{val:+.2f}%**"
            
    elif any(x in text for x in ["pl", "profit", "loss", "gain", "money made"]):
        # PL
        if horizon == "SI":
            val = data.get("pl_si", 0)
        else:
            val = dw.calculate_horizon_pl(data["pv"], data["inception_date"], data["cf_ext"], horizon)
        
        if val is None: return f"Portfolio P/L for {horizon} is unavailable."
        return f"**Portfolio P/L ({horizon})**: **${val:,.2f}**"
        
    elif any(x in text for x in ["flows", "deposit", "withdrawal", "net flow", "invested"]):
        # Flows
        # If horizon is SI, sum all. If YTD, filter.
        cf_ext = data.get("cf_ext")
        if cf_ext is None or cf_ext.empty: return "No external flows recorded."
        
        subset = cf_ext
        if horizon == "YTD":
            start = pd.Timestamp.now().replace(month=1, day=1, hour=0, minute=0, second=0)
            subset = cf_ext[cf_ext["date"] >= start]
        elif horizon != "SI":
            # Approximation for other horizons
            return "Flows are summarized for **SI** or **YTD**."
            
        deposits = subset.loc[subset["amount"] > 0, "amount"].sum()
        withdrawals = subset.loc[subset["amount"] < 0, "amount"].sum()
        net = subset["amount"].sum()
        
        return (f"**Flows Summary ({horizon})**:\n"
                f"- Net Invested: **${net:,.2f}**\n"
                f"- Deposits: ${deposits:,.2f}\n"
                f"- Withdrawals: ${withdrawals:,.2f}")
                
    elif "cash" in text:
        # Cash Balance
        sec_current = data.get("sec_table_current")
        cash_row = sec_current[sec_current["ticker"] == "CASH"]
        val = cash_row["market_value"].iloc[0] if not cash_row.empty else 0.0
        return f"**Current Cash Balance**: **${val:,.2f}**"
        
    elif any(x in text for x in ["sharpe", "sortino"]):
        # Risk Efficiency
        twr_curve = dw._get_daily_twr_curve(data)
        eff = dw.calculate_efficiency_metrics(twr_curve)
        
        sharpe = f"{eff['sharpe']:.2f}" if isinstance(eff['sharpe'], (int, float)) else "N/A"
        sortino = f"{eff['sortino']:.2f}" if isinstance(eff['sortino'], (int, float)) else "N/A"
        
        return (f"**Portfolio Efficiency Scores**:\n"
                f"- **Sharpe Ratio**: {sharpe}\n"
                f"- **Sortino Ratio**: {sortino}")

    elif "drawdown" in text or "max dd" in text:
        # Drawdown
        twr_curve = dw._get_daily_twr_curve(data)
        _, max_dd, recovery = dw.compute_drawdown_series(twr_curve)
        
        return (f"**Drawdown Analysis**:\n"
                f"- **Max Drawdown**: {max_dd:.2f}%\n"
                f"- **Days to Recover**: {recovery} days")

    elif "beta" in text or "tracking error" in text:
        # Active Risk
        metrics = dw.calculate_active_metrics(data, "SPY") # Default to SPY
        beta = metrics.get("beta", "N/A")
        te = metrics.get("te", "N/A")
        
        if isinstance(beta, (int, float)): beta = f"{beta:.2f}"
        if isinstance(te, (int, float)): te = f"{te:.2f}%"
        
        return (f"**Active Risk vs SPY**:\n"
                f"- **Beta**: {beta}\n"
                f"- **Tracking Error**: {te}")

    else:
        # Default to Total Value
        pv = data.get("pv")
        if not pv.empty:
            val = pv.iloc[-1]
            return f"**Current Portfolio Value**: **${val:,.2f}**"
            
    return "Could not determine portfolio metric."

def handle_benchmark_query(text, data, horizon):
    """
    Handles "excess return vs [TICKER]"
    """
    # Note: "vs" is normalized to "benchmark"
    match = re.search(r"(?:vs|benchmark)\s+(\w+)", text, re.IGNORECASE)
    if not match: return "Please specify a benchmark ticker (e.g. 'vs SPY')."
    
    bm_ticker = match.group(1).upper()
    
    # Get Portfolio Return
    twr_df = data.get("twr_df")
    port_ret = 0.0
    
    if horizon == "SI":
         port_ret = data["twr_si_ann"] if pd.notna(data["twr_si_ann"]) else data["twr_si"]
    else:
        row = twr_df[twr_df["Horizon"] == horizon]
        if row.empty: return f"Portfolio return for {horizon} unavailable."
        port_ret = row["Return"].iloc[0]
        
    # Get Benchmark Return
    try:
        # Need start date
        pv = data["pv"]
        if horizon == "SI":
            start = data["inception_date"]
        else:
            start = dw.get_portfolio_horizon_start(pv, data["inception_date"], horizon)
            
        if start is None: return "Horizon start date invalid."
        
        # Fetch
        hist = fetch_price_history([bm_ticker])
        if bm_ticker not in hist: return f"Could not fetch data for {bm_ticker}."
        
        ser = hist[bm_ticker]
        ser = ser[ser.index >= start]
        # Clip to portfolio end
        ser = ser[ser.index <= pv.index.max()]
        
        if len(ser) < 2: return f"Not enough history for {bm_ticker} over {horizon}."
        
        bm_ret = ser.iloc[-1] / ser.iloc[0] - 1.0
        
        excess = (port_ret - bm_ret) * 100
        
        return (f"**Excess Return vs {bm_ticker} ({horizon})**:\n"
                f"- Portfolio: {port_ret*100:+.2f}%\n"
                f"- {bm_ticker}: {bm_ret*100:+.2f}%\n"
                f"- **Alpha**: **{excess:+.2f}%**")
                
    except Exception as e:
        return f"Error calculating benchmark comparison: {str(e)}"

def process_data_query(text, context=None):
    """
    Master Dispatcher for Data Queries.
    """
    try:
        data = dw.get_data()
        if not data: return None, "Data is currently unavailable.", context
            
        text_norm = normalize_text(text)
        
        # --- CONTEXT MANAGEMENT (Improvement 2) ---
        context = context or {}
        last_horizon = context.get("last_horizon")
        last_entity = context.get("last_entity")
        last_entity_type = context.get("last_entity_type")
        last_metric = context.get("last_metric")
        
        # 1. Resolve Horizon
        new_horizon = parse_horizon(text)
        # If explicit horizon, use it. Else use context. Else default to SI.
        resolved_horizon = new_horizon if new_horizon else (last_horizon or "SI")
        
        # 2. Resolve Entity
        new_entity, new_entity_type = extract_entity(text_norm, data)
        
        # Priority Rule: Explicit > Context
        # Only use context if NO new entity found AND query implies specific entity intent
        resolved_entity = new_entity if new_entity else last_entity
        resolved_entity_type = new_entity_type if new_entity else last_entity_type
        
        # 3. Resolve Metric
        new_metric = extract_metric(text_norm)
        resolved_metric = new_metric if new_metric else last_metric
        
        # Special Case: If resolving to default/context, ensure we don't accidentally
        # apply an entity context to a portfolio-wide query (e.g. "Total value").
        is_portfolio_query = any(x in text_norm for x in ["portfolio", "total", "cash", "account", "flows", "benchmark", "excess return"])
        if is_portfolio_query:
            # If explicitly asking for portfolio, ignore entity context for this query,
            # BUT we might want to keep the context for later? 
            # Requirement says "updates this memory every time I ask a new valid question".
            # If I ask "Total Value", should I forget "Apple"? 
            # Usually yes, context shifts. But if I ask "Total Value" then "Return", maybe return to Portfolio?
            # Let's say explicit portfolio query clears entity context.
            if not new_entity: 
                resolved_entity = None
                resolved_entity_type = None

        # Prepare Updated Context
        updated_context = {
            "last_horizon": resolved_horizon,
            "last_entity": resolved_entity,
            "last_entity_type": resolved_entity_type,
            "last_metric": resolved_metric
        }

        # --- DISPATCH WITH RESOLVED VALUES ---
        
        # 1. Benchmark Query
        if "benchmark" in text_norm or "excess return" in text_norm:
            return None, handle_benchmark_query(text_norm, data, resolved_horizon), updated_context
            
        # 2. Ranking Query (Must come before Entity check to avoid "best ticker" being treated as entity lookup)
        if any(x in text_norm for x in ["highest", "lowest", "best", "worst", "top", "bottom", "rank", "list"]):
            return None, handle_ranking_query(text_norm, data, resolved_horizon, resolved_metric), updated_context

        # 3. Entity Query (Specific OR Contextual)
        if resolved_entity:
            # If the metric is about transactions, route to the new handler
            if resolved_metric in ["transaction", "net_invested"]:
                return None, handle_transaction_query(resolved_entity, resolved_entity_type, text_norm, data, resolved_horizon), updated_context
            # Otherwise, use the existing entity query handler
            return None, handle_entity_query(resolved_entity, resolved_entity_type, text_norm, data, resolved_horizon, resolved_metric), updated_context

        # 4. Portfolio Level Query (Default fallthrough)
        # If no entity and not a ranking, assume portfolio
        return None, handle_portfolio_query(text_norm, data, resolved_horizon), updated_context
        
    except Exception as e:
        return None, f"An error occurred while processing your data query: {str(e)}", context

# ============================================================
# COMPONENT LAYOUT
# ============================================================

layout = html.Div([
    # Floating Toggle Button
    dbc.Button(
        html.I(className="bi bi-chat-dots-fill"), 
        id="btn-chatbot-toggle",
        color="primary",
        style={
            "position": "fixed",
            "bottom": "20px",
            "right": "20px",
            "zIndex": 2000,
            "borderRadius": "50%",
            "width": "60px",
            "height": "60px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "fontSize": "1.5rem",
            "boxShadow": "0 4px 8px rgba(0,0,0,0.3)"
        }
    ),

    # Offcanvas Panel
    dbc.Offcanvas(
        html.Div([
            # Chat History Area
            html.Div(
                id="chat-history-display",
                style={
                    "flex": "1",
                    "overflowY": "auto",
                    "padding": "1rem",
                    "marginBottom": "1rem",
                    "border": "1px solid rgba(255,255,255,0.1)",
                    "borderRadius": "5px",
                    "backgroundColor": "rgba(0,0,0,0.2)",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "10px"
                }
            ),
            
            # Input Area
            html.Div([
                dbc.Input(
                    id="chat-input", 
                    placeholder="Ask about returns, allocations, or rankings...", 
                    type="text",
                    autoComplete="off",
                    n_submit=0
                ),
                dbc.Button("Send", id="btn-chat-send", color="primary", className="mt-2 w-100"),
            ])
        ], style={"display": "flex", "flexDirection": "column", "height": "100%"}),
        id="chatbot-offcanvas",
        title="Portfolio Assistant",
        placement="end",
        is_open=False,
        style={"width": "400px", "maxWidth": "80vw"}, 
    ),
    
    # Internal Stores
    dcc.Store(id="chat-history-store", data=[], storage_type="session"),
    dcc.Store(id="chatbot-context", data={}, storage_type="session"),
    # Global Command Store (Main App listens to this)
    dcc.Store(id="chatbot-command", data={}),
    # Signal to close the panel
    dcc.Store(id="close-chatbot-store", data=False),
    
    # Success Toast
    dbc.Toast(
        id="chatbot-toast",
        header="Success",
        is_open=False,
        dismissable=True,
        duration=4000,
        icon="success",
        style={"position": "fixed", "top": 80, "right": 20, "width": 350, "zIndex": 2100},
    ),
])

# ============================================================
# LOGIC & PARSING
# ============================================================

def parse_intent(text, pathname=None, context=None):
    """
    Parses user intent from text.
    Returns (command_dict, response_text, updated_context).
    """
    try:
        # Pre-process
        raw_text = text.lower().strip()
        text_norm = normalize_text(raw_text)
        context = context or {}
        
        current_page = pathname.strip("/").lower() if pathname else "overview"
        if not current_page: current_page = "overview"
        
        # 1. SORT (Strict Regex on Normalized)
        sort_match = re.search(r"sort\s+(?P<target>.*?)\s+by\s+(?P<col>.*)", text_norm)
        if not sort_match and text_norm.startswith("sort"):
             sort_match = re.search(r"sort\s+(?P<col>.*)", text_norm)
             target = "table"
        else:
             target = sort_match.group("target").strip() if sort_match else None
             
        if sort_match:
            col = sort_match.group("col").strip()
            direction = "desc"
            if "ascending" in col or " lowest" in col: 
                direction = "asc"
                col = col.replace("ascending", "").replace("lowest", "")
            elif "descending" in col or "highest" in col:
                direction = "desc"
                col = col.replace("descending", "").replace("highest", "")
            col = col.replace("by ", "").strip()
            return {
                "action": "SORT",
                "params": {"column": col, "direction": direction, "target": target}
            }, f"Sorting **{target or 'table'}** by **{col}** ({direction})...", context

        # 2. FILTER
        filter_match = re.search(r"filter\s+(?:the\s+)?(?:table|grid)?\s*(?:by)?\s+(?P<val>[\w\s]+)", text_norm)
        if filter_match:
            val = filter_match.group("val").strip()
            if val in ["reset", "clear", "all", "everything"]:
                return {"action": "RESET", "params": {}}, "Clearing all filters.", context
            return {
                "action": "FILTER", 
                "params": {"value": val}
            }, f"Filtering table for **{val}**...", context

        # 3. ANALYZE
        analysis_keywords = ["analyze", "analysis", "risk", "diversified", "diversification", "overweight", "underweight", "profile", "concentration"]
        if any(k in text_norm for k in analysis_keywords):
            return None, analyze_portfolio(), context

        # --- INTENT BIFURCATION ---
        # Check if user explicitly wants an explanation (report, chart, definition)
        # Check RAW text for phrases like "what is the" or "what is" to avoid normalization issues
        is_explanation = any(x in raw_text for x in ["report", "chart", "graph", "table", "explain", "describe", "definition", "meaning", "what is the", "what is"])
        
        # Helper to check KB
        def check_knowledge_base():
            # Check component registry
            for key, data in COMPONENT_REGISTRY.items():
                # Check both normalized and raw text to catch "Portfolio Value" (which norm converts to market_value)
                name_match = data["canonical_name"].lower() in raw_text or data["canonical_name"].lower() in text_norm
                question_match = any(q in raw_text for q in data["common_questions"]) or any(q in text_norm for q in data["common_questions"])
                
                if name_match or question_match:
                    return (
                        f"### {data['canonical_name']}\n"
                        f"{data['description']}\n\n"
                        f"**Interpretation**: {data['interpretation']}"
                    )
            # Check explanations
            for key, val in EXPLANATIONS.items():
                if key in text_norm or key in raw_text:
                    return val
            
            # Check HELP_TOPICS (New Knowledge Base)
            for key, topic in HELP_TOPICS.items():
                # Check key match or title match
                if key in text_norm or topic["title"].lower() in text_norm:
                    return f"### {topic['title']}\n{topic['content']}"
            
            return None

        # 4. EXPLANATION PRIORITY
        if is_explanation:
            kb_response = check_knowledge_base()
            if kb_response:
                return None, kb_response, context

        # 5. DATA QUERY (The core enhancement)
        # We try to process as a data query. If it returns a response, use it.
        # If not, fall through to explanation/help.
        _, data_response, updated_context = process_data_query(text, context)
        if data_response:
            return None, data_response, updated_context

        # 6. EXPLANATION FALLBACK
        # If not explicit explanation but data query failed, try KB now
        if not is_explanation:
            kb_response = check_knowledge_base()
            if kb_response:
                return None, kb_response, context

        # 6. RESET
        if text_norm in ["reset", "clear", "show all"]:
            return {"action": "RESET", "params": {}}, "Resetting view.", context

        # 7. GREETING / DEFAULT
        if text_norm in ["hi", "hello", "help", "menu", "start"]:
            return None, (
                "Hello! I can answer questions about your portfolio.\n\n"
                "**Try asking:**\n"
                "- 'What is my second best ticker?'\n"
                "- 'Total portfolio value'\n"
                "- 'Excess return vs SPY YTD'\n"
                "- 'What is my Gold allocation?'\n"
                "- 'Total deposits'\n"
                "- 'Sort by return'"
            ), context

        return None, "I'm not sure how to answer that. Try asking about **returns**, **P/L**, **allocations**, or **rankings**.", context
        
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}", context

# ============================================================
# CALLBACKS
# ============================================================

def register_callbacks(app):
    
    # 1. Toggle Panel
    @app.callback(
        Output("chatbot-offcanvas", "is_open"),
        [Input("btn-chatbot-toggle", "n_clicks"),
         Input("close-chatbot-store", "data")],
        State("chatbot-offcanvas", "is_open"),
    )
    def toggle_chatbot(n, close_signal, is_open):
        ctx = dash.callback_context
        if not ctx.triggered: return is_open
        id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if id == "btn-chatbot-toggle":
            return not is_open
        elif id == "close-chatbot-store":
            if close_signal:
                return False
        return is_open

    # 2. Process User Message (Logic)
    @app.callback(
        [Output("chat-history-store", "data"),
         Output("chat-input", "value"),
         Output("chatbot-command", "data"),
         Output("close-chatbot-store", "data"),
         Output("chatbot-toast", "children"),
         Output("chatbot-toast", "is_open"),
         Output("chatbot-context", "data")],
        [Input("btn-chat-send", "n_clicks"),
         Input("chat-input", "n_submit")],
        [State("chat-input", "value"),
         State("chat-history-store", "data"),
         State("url", "pathname"),
         State("chatbot-context", "data")]
    )
    def process_user_message(n_clicks, n_submit, text, history, pathname, context):
        if not text:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        history = history or []
        context = context or {}
        
        # Add User Message
        history.append({"role": "user", "content": text})
        
        # Process Logic
        command, response_text, updated_context = parse_intent(text, pathname, context)
        
        # Add Bot Message
        history.append({"role": "bot", "content": response_text})
            
        # Prepare Command (if any)
        cmd_data = no_update
        should_close = False
        toast_msg = ""
        toast_open = False
        
        if command:
            cmd_data = {
                "action": command["action"],
                "params": command["params"],
                "timestamp": datetime.now().isoformat()
            }
            # If actionable command, close and toast
            if command["action"] in ["SORT", "FILTER", "RESET"]:
                should_close = True
                toast_msg = response_text
                toast_open = True
            
        return history, "", cmd_data, should_close, toast_msg, toast_open, updated_context

    # 3. Render Chat History (Display)
    @app.callback(
        Output("chat-history-display", "children"),
        Input("chat-history-store", "data")
    )
    def render_chat_history(history):
        if not history:
            return []
            
        display = []
        for msg in history:
            is_user = msg["role"] == "user"
            style = {
                "alignSelf": "flex-end" if is_user else "flex-start",
                "backgroundColor": "#4C6A92" if is_user else "#444",
                "color": "white",
                "padding": "8px 12px",
                "borderRadius": "12px",
                "maxWidth": "85%",
                "boxShadow": "0 1px 2px rgba(0,0,0,0.2)"
            }
            display.append(html.Div(dcc.Markdown(msg["content"], mathjax=True), style=style))
            
        return display
