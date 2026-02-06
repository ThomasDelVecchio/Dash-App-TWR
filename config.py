import os
import importlib

from dotenv import load_dotenv  # <--- NEW: Import the library

# ============================================================
# API KEYS (SECURE LOAD)
# ============================================================

# 1. NEW: Load the .env file immediately
load_dotenv() 

# 2. Try loading from Environment (Now populated by .env)
FMP_API_KEY = os.environ.get("FMP_API_KEY")

# 3. If missing, try Google Colab Secrets (Keeps it working for iPad)
if not FMP_API_KEY:
    try:
        colab = importlib.import_module("google.colab")
        userdata = getattr(colab, "userdata", None)
        if userdata is not None:
            FMP_API_KEY = userdata.get("FMP_API_KEY")
    except Exception:
        pass

# 4. Safety Check
if not FMP_API_KEY:
    print("⚠️ WARNING: FMP_API_KEY not found. Using 'demo' mode.")
    FMP_API_KEY = "demo"

# ============================================================
# E*TRADE API CONFIGURATION
# ============================================================
# Add these to your .env file:
#   ETRADE_CONSUMER_KEY=your_oauth_consumer_key
#   ETRADE_CONSUMER_SECRET=your_consumer_secret
#   ETRADE_ACCOUNT_ID=your_account_id_key (optional, auto-detected if omitted)
#   ETRADE_SANDBOX=false (set to true for testing)

ETRADE_CONSUMER_KEY = os.environ.get("ETRADE_CONSUMER_KEY")
ETRADE_CONSUMER_SECRET = os.environ.get("ETRADE_CONSUMER_SECRET")
ETRADE_ACCOUNT_ID = os.environ.get("ETRADE_ACCOUNT_ID")  # Optional
ETRADE_SANDBOX = os.environ.get("ETRADE_SANDBOX", "false").lower() == "true"

# Enable/disable automatic E*TRADE sync on dashboard startup
ETRADE_AUTO_SYNC = os.environ.get("ETRADE_AUTO_SYNC", "true").lower() == "true"

# Headless mode (for Colab/iPad) - prevents browser OAuth prompts
# Auto-detects Colab environment, or can be manually set via env var
def _detect_colab():
    """Detect if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

ETRADE_HEADLESS = (
    os.environ.get("ETRADE_HEADLESS", "").lower() == "true" or 
    _detect_colab()
)

# Skip transaction sync entirely (use when E*TRADE transactions API is down/hanging)
# Set to True to only sync holdings (which works) and use existing cashflows.csv
ETRADE_SKIP_TRANSACTIONS = os.environ.get("ETRADE_SKIP_TRANSACTIONS", "false").lower() == "true"

# Sync timeout in seconds (how long to wait before giving up)
ETRADE_SYNC_TIMEOUT = int(os.environ.get("ETRADE_SYNC_TIMEOUT", "45"))  # Increased from 15s for slow E*TRADE responses

def is_etrade_configured() -> bool:
    """Check if E*TRADE credentials are properly configured."""
    return bool(ETRADE_CONSUMER_KEY and ETRADE_CONSUMER_SECRET)

# ============================================================
# FMP PRICE DATA CONFIGURATION (Hybrid Mode)
# ============================================================
# FMP Starter Plan provides 5 years of historical data.
# Set FMP_PRICE_ENABLED=True to use FMP for last 5 years + yfinance for older data.
# Set FMP_PRICE_ENABLED=False to use yfinance only (free, unlimited history).
FMP_PRICE_ENABLED = True
FMP_PRICE_LOOKBACK_YEARS = 5  # FMP covers last 5 years (Starter plan limit)

TARGET_PORTFOLIO_VALUE = 75000.0

TARGET_MONTHLY_CONTRIBUTION = 400  # or whatever value you want

# ============================================================
# TAX RATES
# ============================================================
TAX_RATE_ST = 0.35  # Short-Term Capital Gains Rate (35%)
TAX_RATE_LT = 0.15  # Long-Term Capital Gains Rate (15%)

# ============================================================
# RISK PARAMETERS
# ============================================================
RISK_FREE_RATE = 0.04  # 4% annual risk-free rate for Sharpe/Sortino ratios

# ============================================================
# GLOBAL COLOR PALETTE
# ============================================================
GLOBAL_PALETTE = [
    "#4C6A92",  # steel blue
    "#8C9CB1",  # soft gray-blue
    "#C0504D",  # muted red
    "#D79E9C",  # soft red-gray
    "#9BBB59",  # olive green
    "#C5D6A4",  # light olive
    "#8064A2",  # muted purple
    "#B1A0C7",  # lavender gray
    "#4F81BD",  # corporate blue
    "#A5B5CF",  # cool gray-blue
    "#F2C200",  # muted gold (accent)
    "#D6B656",  # soft gold-gray
]

# ============================================================
# STRATEGY BACKTESTING PRESETS
# ============================================================
TARGET_WEIGHT_PRESET_NAME = "Target Weights (Holdings)"

BENCHMARK_PRESETS = [
    {"name": "S&P 500 (SPY)", "weights": {"SPY": 100}},
    {"name": "Classic 60/40", "weights": {"VTI": 60, "BND": 40}},
    {"name": "Bogleheads 3-Fund", "weights": {"VTI": 60, "VXUS": 20, "BND": 20}},
    {"name": "Ray Dalio All Weather", "weights": {"VTI": 30, "TLT": 40, "IEF": 15, "GLD": 7.5, "DBC": 7.5}},
    {"name": "Golden Butterfly", "weights": {"VTI": 20, "VBR": 20, "TLT": 20, "SHY": 20, "GLD": 20}},
    {"name": "Permanent Portfolio", "weights": {"VTI": 25, "TLT": 25, "GLD": 25, "SHY": 25}},
]

# ============================================================
# MODULE CONFIGURATION
# ============================================================
NAV_MODULES = [
    {"label": "Overview", "href": "/", "id": "overview", "icon": "bi-house-door", "can_toggle": False, "group": "OVERVIEW"},
    {"label": "Performance", "href": "/performance", "id": "performance", "icon": "bi-graph-up-arrow", "can_toggle": True, "group": "ANALYSIS"},
    {"label": "Allocation", "href": "/allocations", "id": "allocations", "icon": "bi-pie-chart", "can_toggle": True, "group": "ANALYSIS"},
    {"label": "Attribution", "href": "/attribution", "id": "attribution", "icon": "bi-bar-chart-line", "can_toggle": True, "group": "ANALYSIS"},
    {"label": "Flows", "href": "/flows", "id": "flows", "icon": "bi-arrow-left-right", "can_toggle": True, "group": "ANALYSIS"},
    {"label": "Holdings", "href": "/holdings", "id": "holdings", "icon": "bi-wallet2", "can_toggle": True, "group": "ANALYSIS"},
    {"label": "Risk & Projections", "href": "/risk", "id": "risk", "icon": "bi-shield-exclamation", "can_toggle": True, "group": "ANALYSIS"},
    {"label": "Trade Lab", "href": "/trade-lab", "id": "trade_lab", "icon": "bi-lightning", "can_toggle": True, "group": "TOOLS"},
    {"label": "Strategy Backtesting", "href": "/strategy-backtesting", "id": "strategy_backtesting", "icon": "bi-activity", "can_toggle": True, "group": "TOOLS"},
    {"label": "Tax Authority", "href": "/taxes", "id": "taxes", "icon": "bi-receipt", "can_toggle": True, "group": "TOOLS"},
    {"label": "Rebalancing", "href": "/rebalancing", "id": "rebalancing", "icon": "bi-sliders", "can_toggle": True, "group": "TOOLS"},
    {"label": "Trade Execution", "href": "/trade", "id": "trade", "icon": "bi-cart-check", "can_toggle": True, "group": "TOOLS"},
    {"label": "Custom Report", "href": "/custom-report", "id": "custom_report", "icon": "bi-file-earmark-text", "can_toggle": True, "group": "TOOLS"},
    {"label": "Settings", "href": "/settings", "id": "settings", "icon": "bi-gear", "can_toggle": False, "group": "CONFIG"},
    {"label": "Help", "href": "/help", "id": "help", "icon": "bi-question-circle", "can_toggle": False, "group": "CONFIG"},
]

