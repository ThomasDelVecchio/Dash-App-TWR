import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
import re
import difflib
import pandas as pd
import numpy as np
from datetime import datetime
import dash_wrappers as dw
from data_loader import fetch_price_history
from financial_math import annualize_return
from pages.help_content import HELP_TOPICS
from tax_engine import build_tax_lots, simulate_sell
from report_formatting import fmt_dollar_clean

# ============================================================
# CONFIG & KNOWLEDGE BASE
# ============================================================

STRATEGIC_KEYWORDS = [
    "should", "sell", "buy", "wait", "time to", "analysis", 
    "advisory", "strategy", "opinion", "recommend", "hold"
]

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
    },
    "tax_sunburst": {
        "canonical_name": "Tax Liability Composition",
        "type": "chart",
        "page": "taxes",
        "description": "Sunburst chart showing the breakdown of your estimated tax liability by Term (Short/Long) and then by Asset Class.",
        "interpretation": "Identify which parts of your portfolio are driving your potential tax bill. Inner ring is Term, outer ring is Asset Class.",
        "common_questions": ["sunburst", "liability chart", "tax composition", "breakdown of taxes"]
    },
    "tax_tactical_radar": {
        "canonical_name": "Tactical Decision Radar",
        "type": "chart",
        "page": "taxes",
        "description": "A scatter plot designed to help with 'Harvest vs Hold' decisions. Maps Unrealized P/L (X-axis) against Days Held (Y-axis).",
        "interpretation": "Top-Left (High Loss, Short Term) = Prime Harvesting candidates. Top-Right (High Gain, Short Term) = Avoid selling if near 365 days (Cliff).",
        "common_questions": ["radar", "tactical radar", "harvest vs hold", "decision chart"]
    },
    "cliff_watch": {
        "canonical_name": "The Cliff Watch",
        "type": "table",
        "page": "taxes",
        "description": "List of tax lots that will turn Long-Term (held > 1 year) within the next 30 days.",
        "interpretation": "If you are planning to sell these, WAIT. Crossing the 1-year mark typically reduces the tax rate on gains significantly.",
        "common_questions": ["cliff watch", "cliff", "turning long term", "wait to sell"]
    },
    "harvest_radar": {
        "canonical_name": "Harvesting Radar",
        "type": "table",
        "page": "taxes",
        "description": "List of tax lots currently at a loss, ranked by the size of the loss.",
        "interpretation": "These are your best opportunities to 'harvest' losses to offset other realized gains and lower your tax bill.",
        "common_questions": ["harvesting radar", "harvest list", "losers", "what to harvest"]
    },
    "tax_lot_explorer": {
        "canonical_name": "Tax Lot Explorer",
        "type": "table",
        "page": "taxes",
        "description": "Comprehensive grid of all open tax lots and realized tax events.",
        "interpretation": "Use this to see exactly when you bought each share and its individual cost basis. Essential for granular tax planning.",
        "common_questions": ["lot explorer", "tax lots", "lots", "basis"]
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
    
    # --- TAX DEFINITIONS ---
    "tax harvest": (
        "**Harvestable Losses** (Tax Loss Harvesting) is the practice of selling an asset that has experienced a loss. \n\n"
        "By realizing this loss, you can use it to **offset realized capital gains** and up to $3,000 of ordinary income "
        "on your tax return, directly lowering your tax bill."
    ),
    "tax efficiency": (
        "**Tax Efficiency** refers to the percentage of your portfolio held in **Long-Term** tax lots (held > 1 year). \n\n"
        "Long-term gains are taxed at preferential rates (typically 15% or 20%), whereas Short-Term gains are taxed "
        "as ordinary income (up to 37%)."
    ),
    "tax cliff": (
        "The **Tax Cliff** is the specific date when a tax lot crosses from Short-Term to Long-Term (366 days). \n\n"
        "Crossing this cliff usually drops the tax rate on gains from ~35% to ~15%."
    ),
    "wash sale": (
        "A **Wash Sale** occurs if you sell a security at a loss and buy a 'substantially identical' security within 30 days before or after the sale. \n\n"
        "If this happens, the IRS **disallows the loss deduction**."
    ),
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
    "net invested": "net_invested",
    "invested": "net_invested",
    "buys": "buys",
    "sells": "sells",
    "purchases": "buys",
    "sales": "sells",
    "transactions": "transactions",
    "trades": "transactions",
    
    # Sector Synonyms
    "tech": "Technology",
    "technology": "Technology",
    "comm services": "Communication Services",
    "financials": "Financials",
    "healthcare": "Health Care",
    "consumer": "Consumer Discretionary",

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

    # Tax Synonyms
    "owe": "tax_realized",
    "bill": "tax_realized",
    "realized": "tax_realized",
    "irs": "tax_realized",
    "uncle sam": "tax_realized",
    "paid taxes": "tax_realized",
    "tax ytd": "tax_realized",
    "pay": "tax_realized",
    
    "liability": "tax_liability",
    "unrealized tax": "tax_liability",
    "unrealized": "tax_liability",
    "shadow bill": "tax_liability",
    "deferred": "tax_liability",
    "potential tax": "tax_liability",
    "if i sold everything": "tax_liability",

    "harvestable losses": "tax_harvest",
    "harvestable loss": "tax_harvest",
    "harvestable": "tax_harvest",
    "tax harvest": "tax_harvest",
    "harvest": "tax_harvest",
    "tax efficiency": "tax_efficiency",
    "efficiency": "tax_efficiency",
    "long term percent": "tax_efficiency",
    "cliff": "tax_cliff",
    "long term": "tax_cliff",
    "short term": "tax_cliff",
    "holding period": "tax_cliff",
    "1 year": "tax_cliff",
    "365": "tax_cliff",

    "write off": "tax_harvest",
    "save tax": "tax_harvest",
    "offset": "tax_harvest",
    
    # Natural Language Phrases
    "how did i do": "return",
    "how am i doing": "return",
    "how is my portfolio": "return",
}

def normalize_text(text, protected_tickers=None):
    """
    Normalize synonyms and casual phrasing to canonical terms.
    
    Args:
        text: The text to normalize
        protected_tickers: Set of ticker symbols that should NOT be replaced by synonyms
                          (e.g., 'IT', 'ALL', 'CAN' are valid tickers that are also common words)
    """
    text = text.lower()
    protected_tickers = protected_tickers or set()
    
    # Convert to uppercase set for case-insensitive comparison
    protected_upper = {t.upper() for t in protected_tickers}
    
    # Sort by length descending to match longest phrases first
    sorted_syns = sorted(SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True)
    
    for syn, canonical in sorted_syns:
        # TICKER PROTECTION: Skip replacement if the synonym is a protected ticker
        # Check if this synonym (uppercase) matches a valid ticker
        if syn.upper() in protected_upper:
            continue
            
        # Use regex for whole-word replacement to avoid partial matches
        text = re.sub(r"\b" + re.escape(syn) + r"\b", canonical, text)
            
    # Cleanup redundant phrases
    text = text.replace("performing performing", "performing")
    text = text.replace("return return", "return")
    text = text.replace("allocation allocation", "allocation")
    
    return text


def _get_all_known_tickers(data):
    """
    Returns a set of all known tickers from both current holdings AND transaction history.
    This enables lookups for closed positions.
    """
    tickers = set()
    
    # 1. Current holdings
    sec_current = data.get("sec_table_current") if data else None
    if sec_current is not None and not sec_current.empty:
        tickers.update(sec_current["ticker"].unique().tolist())
    
    # 2. Transaction history (includes closed positions)
    tx_raw = data.get("tx_raw") if data else None
    if tx_raw is not None and not tx_raw.empty:
        tickers.update(tx_raw["ticker"].str.upper().unique().tolist())
    
    # 3. Common benchmark tickers
    tickers.update(["SPY", "QQQ", "IWM", "ACWI", "AGG", "GLD", "BTC"])
    
    # 4. Add common tickers that are also English words (protect these from synonym replacement)
    # These are real tickers that could be mistakenly replaced by synonym logic
    common_word_tickers = ["IT", "ALL", "CAN", "A", "C", "V", "F", "T", "K", "X", "LOW", "BIG", "CAT", "FAST", "NOW", "WELL"]
    tickers.update(common_word_tickers)
    
    return tickers


def _resolve_pronouns(text, context):
    """
    Resolves pronouns like 'it', 'that', 'this' to the last discussed entity.
    
    Example: If context has last_entity='AAPL', "Should I sell it?" becomes "Should I sell AAPL?"
    """
    if not context:
        return text
        
    last_entity = context.get("last_entity")
    if not last_entity:
        return text
    
    # Only resolve for ticker entities (not asset classes or sectors)
    last_entity_type = context.get("last_entity_type")
    if last_entity_type != "ticker":
        return text
    
    # Pronouns that could refer to a ticker
    pronouns = [r"\bit\b", r"\bthat\b", r"\bthis one\b", r"\bthis\b", r"\bthe stock\b", r"\bthe ticker\b"]
    
    for pronoun in pronouns:
        # Only replace if it appears in a context suggesting it refers to a security
        # e.g., "sell it" but not "is it good"
        action_context = re.search(rf"(sell|buy|hold|keep|dump|add to|reduce)\s+{pronoun}", text)
        if action_context:
            text = re.sub(pronoun, last_entity.lower(), text)
            break
            
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
    if re.search(r"\bte\b", text) or any(x in text for x in ["tracking error", "active risk"]): return "te"
    
    if any(x in text for x in ["return", "performance", "growth"]): return "return"
    if re.search(r"\bpl\b", text) or any(x in text for x in ["profit", "loss", "gain", "money made", "p/l"]): return "pl"
    if any(x in text for x in ["allocation", "weight", "portfolio share", "exposure", "position"]): return "allocation"
    if any(x in text for x in ["value", "worth", "balance", "amount"]): return "value"
    if any(x in text for x in ["buys", "sells", "transactions", "trades"]): return "transaction"
    if any(x in text for x in ["net invested", "net amount", "net in", "total in", "net_invested"]): return "net_invested"
    if any(x in text for x in ["flows", "deposit", "withdrawal", "net flow"]): return "flows"
    
    # Tax Metrics
    if "tax_liability" in text: return "tax_liability"
    if "tax_realized" in text: return "tax_realized"
    if "tax_harvest" in text: return "tax_harvest"
    if "tax_efficiency" in text: return "tax_efficiency"
    if "tax_cliff" in text: return "tax_cliff"
    
    return None

def extract_entity(text, data):
    """
    Identifies if the user is asking about a specific Ticker, Asset Class, or Sector.
    Returns (entity_name, entity_type).
    
    IMPORTANT: Searches both current holdings AND transaction history to support
    queries about closed positions (assets no longer owned).
    """
    text = text.lower()
    sec_current = data.get("sec_table_current")
    if sec_current is None: return None, None
    
    # 1. Tickers (Exact Match) - Search BOTH current holdings AND transaction history
    tickers = set(sec_current["ticker"].unique().tolist())
    
    # Add tickers from transaction history (closed positions)
    tx_raw = data.get("tx_raw")
    if tx_raw is not None and not tx_raw.empty:
        historical_tickers = tx_raw["ticker"].str.upper().unique().tolist()
        tickers.update(historical_tickers)
    
    # Add common benchmark tickers
    tickers.update(["SPY", "QQQ", "IWM", "ACWI", "AGG", "GLD", "BTC"])
    
    for t in tickers:
        # Check for word boundary to avoid partial matches
        if re.search(r"\b" + re.escape(t.lower()) + r"\b", text):
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

    # Only check for asset classes if a ticker hasn't been found or if the text strongly implies AC
    # But wait, the ticker loop already returned if it found one.
    # The issue is "gold" is in "should i wait to sell gld".
    # We should use word boundaries for AC keywords too.
    
    for ac in ac_list:
        if re.search(r"\b" + re.escape(ac.lower()) + r"\b", text):
            return ac, "asset_class"
            
    for keyword, formal in ac_map.items():
        if re.search(r"\b" + re.escape(keyword) + r"\b", text):
            return formal, "asset_class"
            
    # 3. Sectors
    sector_df = data.get("sector_df")
    if sector_df is not None and not sector_df.empty:
        sectors = sector_df["Sector"].unique().tolist()
        
        # Explicit Sector Mappings (Abbreviations)
        sector_map = {
            "tech": "Technology",
            "comm services": "Communication Services",
            "finance": "Financials",
            "healthcare": "Health Care",
            "consumer": "Consumer Discretionary" # or Staples, but usually Discretionary is implied
        }
        
        for s in sectors:
            if s.lower() in text:
                return s, "sector"
                
        for abbr, full in sector_map.items():
            if re.search(r"\b" + re.escape(abbr) + r"\b", text):
                # Verify the full name exists in our data
                if full in sectors:
                    return full, "sector"
    
    # --- FUZZY MATCHING (Improvement 1) ---
    # If no exact match found, look for close typos.
    # We scan the text for words that might be close to our entities.
    
    # Build Candidate Map: {lowercase_candidate: (canonical_name, type)}
    candidate_map = {}
    
    # Tickers (both current AND historical for closed position support)
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
        
        matches = difflib.get_close_matches(word, all_candidates, n=1, cutoff=0.8)
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

def handle_tax_metric_query(entity_name, entity_type, metric, data, horizon=None):
    """
    Handles specific tax metric queries for Portfolio or Entities.
    Metrics: tax_liability, tax_realized, tax_harvest, tax_efficiency, tax_cliff
    """
    open_lots, realized_events = build_tax_lots()
    
    # Filter by Entity if provided
    filtered_lots = open_lots
    filtered_realized = realized_events
    
    entity_label = "Portfolio"
    
    if entity_name:
        entity_label = entity_name
        # Normalize for matching
        norm_entity = entity_name.upper()
        
        if entity_type == "ticker":
            if not open_lots.empty:
                filtered_lots = open_lots[open_lots["Ticker"] == norm_entity]
            if not realized_events.empty:
                filtered_realized = realized_events[realized_events["Ticker"] == norm_entity]
                
        elif entity_type == "asset_class":
            # We need to map lots to asset class. This requires joining with sec_table_current.
            # This is expensive, so we do it only if needed.
            sec_current = data.get("sec_table_current")
            if sec_current is not None:
                # Create Ticker -> Asset Class map
                # Ensure unique
                ac_map = sec_current.set_index("ticker")["asset_class"].to_dict()
                
                if not open_lots.empty:
                    # Map
                    filtered_lots = open_lots.copy()
                    filtered_lots["Asset Class"] = filtered_lots["Ticker"].map(ac_map)
                    filtered_lots = filtered_lots[filtered_lots["Asset Class"] == entity_name]
                    
                if not realized_events.empty:
                    filtered_realized = realized_events.copy()
                    filtered_realized["Asset Class"] = filtered_realized["Ticker"].map(ac_map)
                    filtered_realized = filtered_realized[filtered_realized["Asset Class"] == entity_name]
            else:
                return "Asset class tax data unavailable (missing holdings data)."

    # 1. Tax Liability (Unrealized)
    if metric == "tax_liability":
        val = filtered_lots["Est Tax Liability"].sum() if not filtered_lots.empty else 0.0
        return f"**{entity_label}** Estimated Tax Liability: **{fmt_dollar_clean(val)}**"

    # 2. Realized Tax (YTD Bill)
    elif metric == "tax_realized":
        val = filtered_realized["Tax Impact"].sum() if not filtered_realized.empty else 0.0
        gains = filtered_realized["Realized P/L"].sum() if not filtered_realized.empty else 0.0
        return f"**{entity_label}** Realized Tax Bill (YTD): **{fmt_dollar_clean(val)}** (on {fmt_dollar_clean(gains)} gains)."

    # 3. Harvestable Losses
    elif metric == "tax_harvest":
        if filtered_lots.empty: return f"No open lots found for **{entity_label}**."
        losers = filtered_lots[filtered_lots["Unrealized P/L"] < 0]
        val = abs(losers["Unrealized P/L"].sum())
        
        count = len(losers)
        return f"**{entity_label}** Harvestable Losses: **{fmt_dollar_clean(val)}** across {count} lots."

    # 4. Tax Efficiency (% Long Term)
    elif metric == "tax_efficiency":
        if filtered_lots.empty: return f"No open positions for **{entity_label}**."
        total_mv = filtered_lots["Market Value"].sum()
        lt_mv = filtered_lots[filtered_lots["Term"] == "Long-Term"]["Market Value"].sum()
        pct = (lt_mv / total_mv * 100) if total_mv > 0 else 0.0
        return f"**{entity_label}** Tax Efficiency: **{pct:.1f}%** Long-Term."

    # 5. Tax Cliff
    elif metric == "tax_cliff":
        if filtered_lots.empty or "Is Near Cliff" not in filtered_lots.columns:
            return f"No lots found for **{entity_label}**."
        near = filtered_lots[filtered_lots["Is Near Cliff"] == True]
        count = len(near)
        return f"**{entity_label}**: **{count}** lots are approaching the Long-Term cliff (< 30 days)."

    return f"Tax metric '{metric}' not implemented for {entity_label}."

def handle_tax_query(text):
    """
    Handles robust tax queries using the Tax Engine.
    """
    text = text.lower()
    print(f"DEBUG Chatbot: handle_tax_query received: '{text}'")
    
    # 1. Simulation (Regex)
    # Pattern A: sell 10 AAPL
    match_a = re.search(r"(?:sell|simulate|dump)\s+(\d+)\s+(?:shares\s+of\s+)?([a-z0-9\-\.]+)", text)
    # Pattern B: sell AAPL 10
    match_b = re.search(r"(?:sell|simulate|dump)\s+([a-z0-9\-\.]+)\s+(\d+)", text)
    
    if match_a or match_b:
        if match_a:
            shares, ticker = float(match_a.group(1)), match_a.group(2)
        else:
            ticker, shares = match_b.group(1), float(match_b.group(2))
            
        result = simulate_sell(ticker, shares)
        return result["summary_text"]

    # Load Data (Only if needed)
    open_lots, realized_events = build_tax_lots()
    
    # 2. Harvesting (Must check for "tax harvest" due to SYNONYMS)
    if "tax_harvest" in text or "harvest" in text:
        print("DEBUG Chatbot: Tax query matched Harvesting")
        if open_lots.empty: return "No open lots to harvest."
        
        losers = open_lots[open_lots["Unrealized P/L"] < 0].sort_values("Unrealized P/L", ascending=True)
        if losers.empty:
            return "You have no unrealized losses available to harvest right now."
            
        total_harvest = abs(losers["Unrealized P/L"].sum())
        top = losers.iloc[0]
        
        return (f"**Harvestable Losses** represent the total dollar amount of unrealized losses currently in your portfolio that can be sold to offset realized gains, thereby reducing your tax bill.\n\n"
                f"You have **{fmt_dollar_clean(total_harvest)}** in unrealized losses available to harvest.\n\n"
                f"Top target: **{top['Ticker']}** (Down **{fmt_dollar_clean(abs(top['Unrealized P/L']))}**).")

    # 3. Tax Efficiency
    if "tax_efficiency" in text:
        if open_lots.empty: return "No open lots to calculate efficiency."
        total_mv = open_lots["Market Value"].sum()
        lt_mv = open_lots[open_lots["Term"] == "Long-Term"]["Market Value"].sum()
        efficiency_pct = (lt_mv / total_mv * 100) if total_mv > 0 else 0.0
        
        return (f"**Tax Efficiency** measures the percentage of your portfolio held in 'Long-Term' lots (assets held for > 1 year).\n\n"
                f"Current Efficiency: **{efficiency_pct:.1f}%**.\n"
                f"Higher efficiency is better, as long-term gains are taxed at a lower rate (typically 0-20%) compared to short-term gains (up to 37%).")

    # 4. Realized Bill
    if "tax_realized" in text:
        total = realized_events["Tax Impact"].sum() if not realized_events.empty else 0.0
        gains = realized_events["Realized P/L"].sum() if not realized_events.empty else 0.0
        return f"You currently have a **Realized Tax Bill** of **{fmt_dollar_clean(total)}**. This comes from **{fmt_dollar_clean(gains)}** in realized gains YTD."

    # 5. Unrealized Liability
    if "tax_liability" in text:
        total = open_lots["Est Tax Liability"].sum() if not open_lots.empty else 0.0
        return f"Your **Unrealized Tax Liability** (Shadow Bill) is **{fmt_dollar_clean(total)}**. This is what you would owe if you liquidated everything today."

    # 6. Cliff Watch
    if "tax_cliff" in text:
        if open_lots.empty or "Is Near Cliff" not in open_lots.columns:
            return "No open lots found."
            
        near_cliff = open_lots[open_lots["Is Near Cliff"] == True]
        if near_cliff.empty:
            return "No lots are currently within 30 days of the Long-Term cliff."
            
        count = len(near_cliff)
        # Pick most significant
        near_cliff = near_cliff.sort_values("Unrealized P/L", ascending=False)
        top = near_cliff.iloc[0]
        
        return (f"⚠️ **YES**. You have **{count}** lots approaching the 1-year mark.\n\n"
                f"Notable: **{top['Ticker']}** ({top['Shares']} shares) turns Long-Term in **{top['Days to LT']}** days. "
                f"Hold to save ~20% on tax.")

    # 7. Catch-All Summary (If text is just "tax" or "taxes" or "status")
    if "tax" in text or "status" in text:
        realized = realized_events["Tax Impact"].sum() if not realized_events.empty else 0.0
        unrealized = open_lots["Est Tax Liability"].sum() if not open_lots.empty else 0.0
        harvest_opps = len(open_lots[open_lots["Unrealized P/L"] < 0]) if not open_lots.empty else 0
        
        return (f"**Tax Status**:\n"
                f"- You currently owe **{fmt_dollar_clean(realized)}** (Realized).\n"
                f"- You have **{fmt_dollar_clean(unrealized)}** in potential Unrealized Liability.\n"
                f"- You have **{harvest_opps}** harvesting opportunities available.")

    return None

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
    Supports BOTH current positions AND closed positions (assets no longer owned).
    
    Examples:
    - "What are my buys in VOO"
    - "Show my sells of AAPL"
    - "Net invested in GOOG"
    - "What did I buy?" (closed position)
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

    # Check if this is a closed position (not in current holdings)
    sec_current = data.get("sec_table_current")
    is_closed_position = True
    if sec_current is not None and not sec_current.empty:
        current_tickers = sec_current["ticker"].unique().tolist()
        is_closed_position = entity_name.upper() not in [t.upper() for t in current_tickers]
    
    closed_note = " *(Closed Position)*" if is_closed_position else ""

    # Determine transaction type from the query
    query_type = "all"
    text_lower = text.lower()
    if "buys" in text_lower or "purchases" in text_lower or "bought" in text_lower:
        query_type = "buy"
    elif "sells" in text_lower or "sales" in text_lower or "sold" in text_lower:
        query_type = "sell"
    elif "net invested" in text_lower or "net amount" in text_lower:
        query_type = "net"

    response_lines = []
    
    if query_type == "buy":
        buys = ticker_tx[ticker_tx["amount"] < 0]
        if buys.empty:
            return f"No buy transactions found for **{entity_name}**{closed_note}."
        response_lines.append(f"**Buy Transactions for {entity_name}**{closed_note}:")
        for _, row in buys.iterrows():
            response_lines.append(f"- {row['date'].strftime('%Y-%m-%d')}: **{row['shares']:,.2f} shares** for **${-row['amount']:,.2f}**")
        total_spent = -buys['amount'].sum()
        response_lines.append(f"**Total Spent:** **${total_spent:,.2f}**")

    elif query_type == "sell":
        sells = ticker_tx[ticker_tx["amount"] > 0]
        if sells.empty:
            return f"No sell transactions found for **{entity_name}**{closed_note}."
        response_lines.append(f"**Sell Transactions for {entity_name}**{closed_note}:")
        for _, row in sells.iterrows():
            response_lines.append(f"- {row['date'].strftime('%Y-%m-%d')}: **{row['shares']:,.2f} shares** for **${row['amount']:,.2f}**")
        total_proceeds = sells['amount'].sum()
        response_lines.append(f"**Total Proceeds:** **${total_proceeds:,.2f}**")

    elif query_type == "net":
        net_invested = -ticker_tx["amount"].sum()
        return f"The net amount invested in **{entity_name}**{closed_note} is **${net_invested:,.2f}**."

    else: # "all" transactions
        response_lines.append(f"**All Transactions for {entity_name}**{closed_note}:")
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

    elif any(x in text for x in ["buys", "sells", "transactions", "trades"]):
        # Portfolio-level transactions
        tx_raw = data.get("tx_raw")
        if tx_raw is None or tx_raw.empty: return "No transactions found."
        
        # Filter by Horizon
        # We need a start date. Use PV index or similar logic.
        start_date = None
        if horizon == "SI":
            start_date = data["inception_date"]
        else:
            # We can use the helper from dw if available, or approximate
            # dw.get_portfolio_horizon_start is used in handle_benchmark_query
            try:
                start_date = dw.get_portfolio_horizon_start(data["pv"], data["inception_date"], horizon)
            except:
                pass
        
        if start_date:
            tx_raw = tx_raw[tx_raw["date"] >= start_date]
            
        if tx_raw.empty: return f"No transactions found for **{horizon}**."
        
        # Filter by Type
        query_type = "all"
        if "buys" in text or "purchases" in text: query_type = "buy"
        elif "sells" in text or "sales" in text: query_type = "sell"
        
        if query_type == "buy":
            subset = tx_raw[tx_raw["amount"] < 0]
            total = -subset["amount"].sum()
            count = len(subset)
            # Top 3 by amount
            top_3 = subset.sort_values("amount", ascending=True).head(3)
            top_str = ", ".join([f"{r['ticker']} (${-r['amount']:,.0f})" for _, r in top_3.iterrows()])
            return f"**Buys ({horizon})**: {count} trades totaling **${total:,.2f}**.\nTop: {top_str}"
            
        elif query_type == "sell":
            subset = tx_raw[tx_raw["amount"] > 0]
            total = subset["amount"].sum()
            count = len(subset)
            top_3 = subset.sort_values("amount", ascending=False).head(3)
            top_str = ", ".join([f"{r['ticker']} (${r['amount']:,.0f})" for _, r in top_3.iterrows()])
            return f"**Sells ({horizon})**: {count} trades totaling **${total:,.2f}**.\nTop: {top_str}"
            
        else:
            count = len(tx_raw)
            return f"**Transactions ({horizon})**: {count} total trades."

    else:
        # Default to Total Value
        pv = data.get("pv")
        if not pv.empty:
            val = pv.iloc[-1]
            return f"**Current Portfolio Value**: **${val:,.2f}**"
            
    return "Could not determine portfolio metric."

def handle_strategic_ticker_query(ticker, data):
    """
    Retrieves a broader set of context for a ticker (YTD, Vol, Benchmark Spread, Tax)
    to generate a text-based strategic response.
    """
    try:
        sec_current = data.get("sec_table_current")
        risk_return = data.get("risk_return", {})
        pv = data.get("pv")
        
        # 1. Performance Context (YTD)
        ytd_ret = 0.0
        asset_class = "Unknown"
        if sec_current is not None and not sec_current.empty:
            row = sec_current[sec_current["ticker"] == ticker]
            if not row.empty:
                ytd_ret = row["YTD"].iloc[0] if "YTD" in row.columns else 0.0
                asset_class = row["asset_class"].iloc[0]
        
        # 2. Volatility Context (Asset Class Level)
        vol = 0.0
        if asset_class in risk_return:
            vol = risk_return[asset_class].get("vol", 0.0)
            
        # 3. Benchmark Comparison (YTD vs SPY)
        spy_ytd = 0.0
        try:
            if sec_current is not None and not sec_current.empty:
                spy_row = sec_current[sec_current["ticker"] == "SPY"]
                if not spy_row.empty:
                    val = spy_row["YTD"].iloc[0]
                    spy_ytd = val if pd.notna(val) else 0.0
        except:
            pass
            
        alpha_ytd = (ytd_ret - spy_ytd) * 100
        
        # 4. Tax Context
        tax_notes = []
        is_near_cliff = False
        cliff_days = 0
        unrealized_pl = 0.0
        has_lots = False
        
        try:
            open_lots, _ = build_tax_lots()
            if not open_lots.empty:
                ticker_lots = open_lots[open_lots["Ticker"] == ticker]
                if not ticker_lots.empty:
                    has_lots = True
                    unrealized_pl = ticker_lots["Unrealized P/L"].sum()
                    tax_notes.append(f"- **Unrealized P/L**: {fmt_dollar_clean(unrealized_pl)}")
                    
                    # Cliff Watch
                    near_cliff = ticker_lots[ticker_lots["Is Near Cliff"] == True]
                    if not near_cliff.empty:
                        is_near_cliff = True
                        cliff_days = int(near_cliff["Days to LT"].min())
                        tax_notes.append(f"- ⚠️ **Tax Cliff**: Lots turn Long-Term in **{cliff_days} days**.")
                    
                    # Harvesting
                    if unrealized_pl < 0:
                        tax_notes.append(f"- 💡 **Candidate for Tax Loss Harvesting**.")
        except Exception as e:
            tax_notes.append(f"*(Tax data error: {str(e)})*")

        # 5. Strategic Synthesis (The Final Answer)
        recommendation = ""
        if has_lots:
            if is_near_cliff:
                recommendation = f"**Strategic View**: Based on the upcoming tax cliff in {cliff_days} days, **waiting** to sell is likely the most tax-efficient move to lock in long-term rates."
            elif unrealized_pl < 0:
                recommendation = "**Strategic View**: This position is at a loss. Selling now could be used for **tax-loss harvesting** to offset other gains."
            elif alpha_ytd > 5:
                recommendation = "**Strategic View**: This ticker is significantly outperforming the benchmark YTD. If your target allocation allows, **holding** to let winners run may be appropriate."
            else:
                recommendation = "**Strategic View**: No immediate tax or performance red flags. Review your target allocation to decide if rebalancing is needed."
        else:
            recommendation = "**Strategic View**: No active position found. Consider the volatility and benchmark spread before entering a new position."

        # 6. Synthesize Response (Format Cleanup)
        perf_val = ytd_ret * 100
        perf_str = f"{perf_val:+.2f}%" if pd.notna(perf_val) else "N/A"
        alpha_str = f"{alpha_ytd:+.2f}%" if pd.notna(alpha_ytd) else "N/A"
        
        perf_color = "🟢" if (pd.notna(perf_val) and perf_val >= 0) else "🔴"
        
        response = [
            f"**Strategic Analysis: {ticker}**",
            f"{perf_color} **YTD**: {perf_str} | **vs SPY**: {alpha_str}",
            f"🎲 **Volatility**: {vol:.1f}% ({asset_class})",
            "\n**Tax Context**:",
        ]
        
        if tax_notes:
            response.extend(tax_notes)
        else:
            response.append("- No active tax lots found.")
            
        response.append(f"\n{recommendation}")
        
        return "\n".join(response)
        
    except Exception as e:
        return f"Error generating strategic analysis for {ticker}: {str(e)}"

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
        ser = hist[bm_ticker]
        
        # Robust Start Price (Snapback logic matching dash_wrappers)
        base_price = None
        if start <= pv.index.min():
             history_before = ser[ser.index < start]
             if not history_before.empty:
                 base_price = float(history_before.iloc[-1])
        
        if base_price is None:
             if not ser[ser.index <= start].empty:
                 base_price = float(ser.asof(start))
             elif not ser[ser.index >= start].empty:
                 base_price = float(ser[ser.index >= start].iloc[0])
                 
        # Robust End Price
        end_date = pv.index.max()
        end_price = float(ser.asof(end_date)) if not ser[ser.index <= end_date].empty else None
        
        if base_price is None or end_price is None:
            return f"Not enough aligned history for {bm_ticker}."
            
        bm_cum = end_price / base_price - 1.0
        
        # Annualize if needed (Consistency with Portfolio Return)
        bm_ret = annualize_return(bm_cum, start, end_date)
        
        excess = (port_ret - bm_ret) * 100
        
        return (f"**Excess Return vs {bm_ticker} ({horizon})**:\n"
                f"- Portfolio: {port_ret*100:+.2f}%\n"
                f"- {bm_ticker}: {bm_ret*100:+.2f}%\n"
                f"- **Alpha**: **{excess:+.2f}%**")
                
    except Exception as e:
        return f"Error calculating benchmark comparison: {str(e)}"

def handle_rebalancing_query(text, data, context=None):
    """
    Handles rebalancing intents like "what should i buy with $500" or "what can i sell".
    
    Supports contextual pronouns (e.g., "Should I sell it?" where "it" refers to
    the previously discussed ticker from conversation context).
    """
    # Resolve pronouns if context is provided
    if context:
        text = _resolve_pronouns(text, context)
    
    # 1. Extract Amount
    # Look for $X or X dollars or just numbers if context implies
    amount = 0
    # Regex for $500, $500.00, 500 dollars
    match = re.search(r"\$?\s?(\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        try:
            amount = float(match.group(1).replace(",", ""))
        except:
            pass
            
    # 2. Determine Intent (Buy vs Sell/Rebalance)
    allow_sales = False
    # If user explicitly asks to sell or rebalance/fix, allow sales
    if any(x in text.lower() for x in ["sell", "sales", "selling", "rebalance", "fix"]):
        allow_sales = True
        
    # 3. Get Recommendations
    recs = dw.get_rebalancing_recommendations(cash_to_deploy=amount, allow_sales=allow_sales)
    
    if not recs:
        return "Your portfolio is perfectly balanced! No trades are recommended at this time."
        
    # 4. Format Response
    response = [f"**Rebalancing Plan** (Cash: ${amount:,.2f} | Sales: {'Yes' if allow_sales else 'No'})"]
    
    buys = [r for r in recs if r['action'] == 'Buy']
    sells = [r for r in recs if r['action'] == 'Sell']
    
    if sells:
        response.append("\n**Sell Recommendations** (Overweight):")
        for r in sells:
            response.append(f"- Sell **{int(r['shares'])}** shares of **{r['ticker']}** (~${r['amount']:,.2f})")
            
    if buys:
        response.append("\n**Buy Recommendations** (Underweight):")
        for r in buys:
            response.append(f"- Buy **{int(r['shares'])}** shares of **{r['ticker']}** (~${r['amount']:,.2f})")
            
    return "\n".join(response)

def process_data_query(text, context=None, force_history=False):
    """
    Master Dispatcher for Data Queries.
    """
    try:
        data = dw.get_data()
        if not data: return None, "Data is currently unavailable.", context
        
        # Get all known tickers for synonym protection
        all_tickers = _get_all_known_tickers(data)
        text_norm = normalize_text(text, all_tickers)
        
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
        
        # Force Portfolio Context for specific metrics that are usually portfolio-level
        # even if an entity is mentioned (e.g. "Portfolio Beta vs SPY" -> SPY is benchmark, not subject)
        if resolved_metric in ["beta", "sharpe", "sortino", "drawdown", "te"]:
             resolved_entity = None
             resolved_entity_type = None
        
        # --- STRATEGIC INTENT BRANCH (NEW) ---
        # Check if this is an advisory/strategic query about a ticker
        is_strategic = any(k in text_norm for k in STRATEGIC_KEYWORDS)
        
        # EXCEPTION: If the user is asking for transactions (buys/sells), it's NOT strategic
        if resolved_metric in ["transaction", "net_invested"]:
            is_strategic = False
            
        # EXCEPTION: If force_history is True (past tense), disable strategic analysis
        if force_history:
            is_strategic = False
            
        if is_strategic and new_entity and new_entity_type == "ticker":
            return None, handle_strategic_ticker_query(new_entity, data), {
                "last_horizon": "YTD", # Strategic analysis defaults to YTD context
                "last_entity": new_entity,
                "last_entity_type": "ticker",
                "last_metric": "strategy"
            }

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

        # 3. Tax Metric Query (Intercept)
        # Check if the resolved metric is tax-related
        if resolved_metric and resolved_metric.startswith("tax_"):
            return None, handle_tax_metric_query(resolved_entity, resolved_entity_type, resolved_metric, data, resolved_horizon), updated_context

        # 4. Entity Query (Specific OR Contextual)
        if resolved_entity:
            # If the metric is about transactions, route to the new handler
            if resolved_metric in ["transaction", "net_invested"]:
                return None, handle_transaction_query(resolved_entity, resolved_entity_type, text_norm, data, resolved_horizon), updated_context
            # Otherwise, use the existing entity query handler
            return None, handle_entity_query(resolved_entity, resolved_entity_type, text_norm, data, resolved_horizon, resolved_metric), updated_context

        # 5. Portfolio Level Query (Default fallthrough)
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
        context = context or {}
        
        current_page = pathname.strip("/").lower() if pathname else "overview"
        if not current_page: current_page = "overview"

        # --- GET ALL VALID TICKERS (for synonym protection) ---
        data = dw.get_data()
        all_tickers = _get_all_known_tickers(data)
        
        # Normalize text AFTER getting tickers (for synonym protection)
        text_norm = normalize_text(raw_text, all_tickers)
        
        # --- PRIORITY 0A: SPECIFIC TAX SIMULATION (Sell 10 AAPL) ---
        # Check for specific sell commands BEFORE generic rebalancing triggers
        # Pattern: "sell X TICKER" or "sell TICKER X" with explicit share count
        specific_sell_match = re.search(
            r"(?:sell|simulate|dump)\s+(\d+)\s+(?:shares?\s+(?:of\s+)?)?([a-z0-9\-\.]+)", 
            raw_text
        ) or re.search(
            r"(?:sell|simulate|dump)\s+([a-z0-9\-\.]+)\s+(\d+)", 
            raw_text
        )
        if specific_sell_match:
            # Route to tax simulation, not rebalancing
            tax_response = handle_tax_query(text_norm)
            if tax_response:
                return None, tax_response, context

        # --- PRIORITY 0B: HISTORY vs. REBALANCING INTENT DISAMBIGUATION ---
        # Past-tense indicators signal HISTORY queries (what did I do?)
        # Future/advisory indicators signal REBALANCING queries (what should I do?)
        
        past_tense_markers = [
            "did i", "have i", "bought", "sold", "purchased", "history", 
            "transactions", "trades", "was", "were", "had", "when did",
            "what did", "how much did", "previously", "last", "ago", "invested"
        ]
        
        future_advisory_markers = [
            "should i", "what should", "what to", "what can i", "recommend", 
            "invest", "deploy", "rebalance", "allocate", "advice", "suggestion"
        ]
        
        is_past_tense = any(marker in raw_text for marker in past_tense_markers)
        is_future_advisory = any(marker in raw_text for marker in future_advisory_markers)
        
        horizon_found = parse_horizon(text)
        
        # RULE: Past-tense queries ALWAYS take priority over rebalancing
        # Even if "buy"/"sell" keywords are present (e.g., "what did I buy?")
        if is_past_tense and not is_future_advisory:
            # This is a history/transaction query - let it flow to data handlers
            pass  # Skip rebalancing check
        elif is_future_advisory and not is_past_tense:
            # This is a rebalancing/advisory query
            # Resolve pronoun context for "it" (e.g., "Should I sell it?")
            resolved_text = _resolve_pronouns(text_norm, context)
            
            # EXCEPTION: If a specific ticker is mentioned, prefer Strategic Analysis (process_data_query)
            # unless it's a clear "what to buy" (rebalancing) query without a ticker.
            # "Should I sell AAPL?" -> Strategic
            # "What should I buy?" -> Rebalancing
            # We check if any known ticker is in the text.
            has_ticker = False
            for t in all_tickers:
                if re.search(r"\b" + re.escape(t.lower()) + r"\b", resolved_text):
                    has_ticker = True
                    break
            
            if has_ticker:
                # Let it fall through to process_data_query which handles strategic intent
                pass
            else:
                return None, handle_rebalancing_query(resolved_text, data, context), context
        elif not is_past_tense and not horizon_found:
            # Ambiguous but no past markers and no horizon - check for rebal triggers
            rebal_triggers = ["buy", "sell", "rebalance", "allocate"]
            if any(x in text_norm for x in rebal_triggers):
                # Additional check: If there's a specific ticker mentioned without past tense,
                # and strategic keywords, treat as rebalancing
                if any(k in text_norm for k in STRATEGIC_KEYWORDS):
                    resolved_text = _resolve_pronouns(text_norm, context)
                    return None, handle_rebalancing_query(resolved_text, data, context), context

        # --- PRIORITY 1: DATA QUERIES (Numeric Answers) ---
        # We process this first so questions like "portfolio return" get numbers, not definitions.
        cmd, data_response, updated_context = process_data_query(text, context, force_history=is_past_tense)
        if data_response:
            # We must be careful: process_data_query fallthrough is "Portfolio Value".
            # Only return if it actually matched something specific or if it's a clear data query.
            # But wait, the user wants "SI return" which IS a data query.
            # If the response isn't the "could not determine" fallback, use it.
            if "could not determine" not in data_response.lower():
                return cmd, data_response, updated_context

        # --- PRIORITY 2: UI COMMANDS (SORT/FILTER) ---
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
        is_explanation = any(x in raw_text for x in ["report", "chart", "graph", "table", "explain", "describe", "definition", "meaning", "what is the", "what is", "define", "how does", "how do"])
        
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

        # 5. TAX QUERY (Priority Intercept)
        # Check if it's a tax question first
        tax_response = handle_tax_query(text_norm)
        if tax_response:
             # Check for raw simulation command too (in case normalization stripped numbers? No, norm keeps numbers)
             return None, tax_response, context

        # 6. DATA QUERY (Fallback)
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
