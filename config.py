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
FMP_PRICE_ENABLED = False
FMP_PRICE_LOOKBACK_YEARS = 5  # FMP covers last 5 years (Starter plan limit)

TARGET_PORTFOLIO_VALUE = 50000.0

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
# MODULE CONFIGURATION
# ============================================================
NAV_MODULES = [
    {"label": "Overview", "href": "/", "id": "overview", "can_toggle": False},
    {"label": "Performance", "href": "/performance", "id": "performance", "can_toggle": True},
    {"label": "Allocations", "href": "/allocations", "id": "allocations", "can_toggle": True},
    {"label": "Attribution", "href": "/attribution", "id": "attribution", "can_toggle": True},
    {"label": "Flows", "href": "/flows", "id": "flows", "can_toggle": True},
    {"label": "Holdings", "href": "/holdings", "id": "holdings", "can_toggle": True},
    {"label": "Rebalancing", "href": "/rebalancing", "id": "rebalancing", "can_toggle": True},
    {"label": "Trade", "href": "/trade", "id": "trade", "can_toggle": True},
    {"label": "Risk & Proj", "href": "/risk", "id": "risk", "can_toggle": True},
    {"label": "Trade Lab", "href": "/trade-lab", "id": "trade_lab", "can_toggle": True},
    {"label": "Tax Authority", "href": "/taxes", "id": "taxes", "can_toggle": True},
    {"label": "Custom Report", "href": "/custom-report", "id": "custom_report", "can_toggle": True},
    {"label": "Settings", "href": "/settings", "id": "settings", "can_toggle": False},
    {"label": "Help Index", "href": "/help", "id": "help", "can_toggle": False},
]
