import os

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
        from google.colab import userdata
        FMP_API_KEY = userdata.get('FMP_API_KEY')
    except ImportError:
        pass

# 4. Safety Check
if not FMP_API_KEY:
    print("⚠️ WARNING: FMP_API_KEY not found. Using 'demo' mode.")
    FMP_API_KEY = "demo"

TARGET_PORTFOLIO_VALUE = 50000.0

TARGET_MONTHLY_CONTRIBUTION = 400  # or whatever value you want

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
