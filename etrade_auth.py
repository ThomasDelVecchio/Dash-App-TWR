"""
E*TRADE OAuth 1.0a Authentication Module

Handles token acquisition, caching, and refresh for E*TRADE API access.
E*TRADE uses OAuth 1.0a (not 2.0), so we use requests_oauthlib.

Usage:
    from etrade_auth import get_etrade_session
    session = get_etrade_session()
    response = session.get("https://api.etrade.com/v1/accounts/list")
"""

import os
import json
import webbrowser
from datetime import datetime, timedelta
from requests_oauthlib import OAuth1Session
from config import (
    ETRADE_CONSUMER_KEY,
    ETRADE_CONSUMER_SECRET,
    ETRADE_SANDBOX,
)

# ============================================================
# CONFIGURATION
# ============================================================
TOKEN_FILE = "etrade_token.json"

# E*TRADE API Endpoints
if ETRADE_SANDBOX:
    BASE_URL = "https://apisb.etrade.com"
    print("⚠️  E*TRADE running in SANDBOX mode")
else:
    BASE_URL = "https://api.etrade.com"

REQUEST_TOKEN_URL = f"{BASE_URL}/oauth/request_token"
AUTHORIZE_URL = "https://us.etrade.com/e/t/etws/authorize"
ACCESS_TOKEN_URL = f"{BASE_URL}/oauth/access_token"
RENEW_TOKEN_URL = f"{BASE_URL}/oauth/renew_access_token"


# ============================================================
# TOKEN MANAGEMENT
# ============================================================

def load_cached_token():
    """Load cached OAuth tokens from disk."""
    if not os.path.exists(TOKEN_FILE):
        return None
    
    try:
        with open(TOKEN_FILE, "r") as f:
            data = json.load(f)
        
        # Check if token was created today (E*TRADE tokens expire at midnight ET)
        created = datetime.fromisoformat(data.get("created", "2000-01-01"))
        now = datetime.now()
        
        # Check if token matches current environment (sandbox vs production)
        # Prevents using sandbox token for production or vice versa
        cached_sandbox = data.get("sandbox", True)  # Default to True for old tokens
        if cached_sandbox != ETRADE_SANDBOX:
            print(f"🔄 E*TRADE token environment mismatch (cached={'sandbox' if cached_sandbox else 'production'}, current={'sandbox' if ETRADE_SANDBOX else 'production'}). Re-authenticating...")
            return None
        
        # Tokens expire at midnight ET, so check if same calendar day
        if created.date() == now.date():
            return data
        else:
            print("🔄 E*TRADE token expired (new day). Re-authenticating...")
            return None
            
    except Exception as e:
        print(f"⚠️  Error loading E*TRADE token: {e}")
        return None


def save_token(oauth_token, oauth_token_secret):
    """Save OAuth tokens to disk with timestamp and environment."""
    data = {
        "oauth_token": oauth_token,
        "oauth_token_secret": oauth_token_secret,
        "created": datetime.now().isoformat(),
        "sandbox": ETRADE_SANDBOX,  # Track which environment this token is for
    }
    
    with open(TOKEN_FILE, "w") as f:
        json.dump(data, f, indent=2)
    
    env_label = "SANDBOX" if ETRADE_SANDBOX else "PRODUCTION"
    print(f"✅ E*TRADE token cached to etrade_token.json ({env_label})")


def delete_cached_token():
    """Remove cached token (forces re-auth on next call)."""
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)
        print("🗑️  E*TRADE token cache cleared")


# ============================================================
# OAUTH 1.0a FLOW
# ============================================================

def authenticate_etrade():
    """
    Complete OAuth 1.0a flow to get access tokens.
    
    Returns:
        tuple: (oauth_token, oauth_token_secret)
    """
    if not ETRADE_CONSUMER_KEY or not ETRADE_CONSUMER_SECRET:
        raise ValueError(
            "E*TRADE credentials not configured. "
            "Set ETRADE_CONSUMER_KEY and ETRADE_CONSUMER_SECRET in .env file."
        )
    
    print("\n" + "="*60)
    print("E*TRADE AUTHENTICATION")
    print("="*60)
    
    # Step 1: Get Request Token
    print("\n📡 Step 1: Requesting temporary token...")
    
    oauth = OAuth1Session(
        ETRADE_CONSUMER_KEY,
        client_secret=ETRADE_CONSUMER_SECRET,
        callback_uri="oob"  # Out-of-band (manual PIN entry)
    )
    
    try:
        response = oauth.fetch_request_token(REQUEST_TOKEN_URL)
    except Exception as e:
        raise RuntimeError(f"Failed to get request token: {e}")
    
    request_token = response.get("oauth_token")
    request_token_secret = response.get("oauth_token_secret")
    
    # Step 2: User Authorization
    print("\n🔑 Step 2: User authorization required...")
    
    auth_url = f"{AUTHORIZE_URL}?key={ETRADE_CONSUMER_KEY}&token={request_token}"
    
    print(f"\n👉 Opening browser to authorize. If it doesn't open, visit:\n")
    print(f"   {auth_url}\n")
    
    try:
        webbrowser.open(auth_url)
    except:
        pass  # Browser may not be available
    
    # Get verification code from user
    verifier = input("📝 Enter the verification code from E*TRADE: ").strip()
    
    if not verifier:
        raise ValueError("Verification code is required")
    
    # Step 3: Exchange for Access Token
    print("\n🔄 Step 3: Exchanging for access token...")
    
    oauth = OAuth1Session(
        ETRADE_CONSUMER_KEY,
        client_secret=ETRADE_CONSUMER_SECRET,
        resource_owner_key=request_token,
        resource_owner_secret=request_token_secret,
        verifier=verifier
    )
    
    try:
        response = oauth.fetch_access_token(ACCESS_TOKEN_URL)
    except Exception as e:
        raise RuntimeError(f"Failed to get access token: {e}")
    
    oauth_token = response.get("oauth_token")
    oauth_token_secret = response.get("oauth_token_secret")
    
    # Save tokens
    save_token(oauth_token, oauth_token_secret)
    
    print("\n✅ Authentication successful!")
    print("="*60 + "\n")
    
    return oauth_token, oauth_token_secret


def renew_access_token(oauth_token, oauth_token_secret):
    """
    Renew access token (extends validity for current session).
    E*TRADE tokens can be renewed to prevent mid-day expiration.
    
    Returns:
        tuple: (oauth_token, oauth_token_secret) - same values, but session extended
    """
    oauth = OAuth1Session(
        ETRADE_CONSUMER_KEY,
        client_secret=ETRADE_CONSUMER_SECRET,
        resource_owner_key=oauth_token,
        resource_owner_secret=oauth_token_secret
    )
    
    try:
        response = oauth.get(RENEW_TOKEN_URL)
        if response.status_code == 200:
            print("🔄 E*TRADE token renewed successfully")
            # Update timestamp in cache
            save_token(oauth_token, oauth_token_secret)
            return oauth_token, oauth_token_secret
        else:
            print(f"⚠️  Token renewal failed: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"⚠️  Token renewal error: {e}")
        return None, None


# ============================================================
# SESSION FACTORY
# ============================================================

def get_etrade_session():
    """
    Get an authenticated OAuth1Session for E*TRADE API calls.
    
    Automatically handles:
    - Loading cached tokens
    - Re-authenticating if expired
    - Creating properly configured session
    
    Returns:
        OAuth1Session: Ready-to-use session for API calls
        
    Example:
        session = get_etrade_session()
        response = session.get(f"{BASE_URL}/v1/accounts/list")
        data = response.json()
    """
    # Try cached token first
    cached = load_cached_token()
    
    if cached:
        oauth_token = cached["oauth_token"]
        oauth_token_secret = cached["oauth_token_secret"]
    else:
        # Need fresh authentication
        oauth_token, oauth_token_secret = authenticate_etrade()
    
    # Create session
    session = OAuth1Session(
        ETRADE_CONSUMER_KEY,
        client_secret=ETRADE_CONSUMER_SECRET,
        resource_owner_key=oauth_token,
        resource_owner_secret=oauth_token_secret
    )
    
    return session


def get_base_url():
    """Return the appropriate base URL (sandbox or production)."""
    return BASE_URL


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    """
    Run this script directly to test authentication:
        python etrade_auth.py
    """
    print("Testing E*TRADE authentication...")
    
    try:
        session = get_etrade_session()
        
        # Test API call
        response = session.get(f"{BASE_URL}/v1/accounts/list")
        
        if response.status_code == 200:
            print("✅ Authentication test PASSED")
            print(f"   Response: {response.text[:200]}...")
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Authentication test FAILED: {e}")
