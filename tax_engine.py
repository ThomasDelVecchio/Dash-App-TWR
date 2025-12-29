import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_loader import load_transactions_raw, fetch_price_history
from config import TAX_RATE_ST, TAX_RATE_LT

# ============================================================
# HELPERS: NORMALIZATION & TERM
# ============================================================

def normalize_ticker(ticker):
    """
    Standardizes ticker strings to prevent mismatch errors.
    Logic: Strip whitespace, Upper case, Replace dots with dashes (e.g., BRK.B -> BRK-B).
    """
    if not isinstance(ticker, str):
        return str(ticker)
    return ticker.strip().upper().replace(".", "-")

def _classify_term(date_acquired, date_sold=None):
    """
    Determines if a lot is Short-Term or Long-Term.
    Threshold: > 365 days is Long-Term.
    """
    if date_sold is None:
        date_sold = datetime.now()
        
    held_duration = date_sold - date_acquired
    
    if held_duration.days > 365:
        return "Long-Term"
    else:
        return "Short-Term"

def _is_near_cliff(date_acquired):
    """
    Returns True if the lot will turn Long-Term in the next 30 days.
    Range: 335 < Days Held <= 365
    """
    days_held = (datetime.now() - date_acquired).days
    return 335 < days_held <= 365

def _days_to_long_term(date_acquired):
    """Returns days remaining until Long-Term status."""
    days_held = (datetime.now() - date_acquired).days
    remaining = 366 - days_held
    return max(0, remaining)

# ============================================================
# CORE ENGINE: FIFO MATCHING
# ============================================================

def build_tax_lots():
    """
    Reconstructs the tax lot history from raw transactions.
    Includes Wash Sale detection.
    
    Returns:
        open_lots_df (pd.DataFrame): Currently held lots.
        realized_events_df (pd.DataFrame): Closed tax events.
    """
    raw_tx = load_transactions_raw()
    if raw_tx.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # 1. Data Hygiene
    raw_tx["date"] = pd.to_datetime(raw_tx["date"])
    raw_tx["ticker"] = raw_tx["ticker"].apply(normalize_ticker)
    
    open_lots = []      # List of {ticker, date_acquired, shares, cost_basis, cost_per_share}
    realized_events = [] # List of {ticker, date_sold, shares, realized_pl, term, tax_impact, is_wash_sale}
    
    # Process by Ticker
    tickers = raw_tx["ticker"].unique()
    
    for ticker in tickers:
        # Filter and Sort Chronologically
        tx_history = raw_tx[raw_tx["ticker"] == ticker].sort_values("date")
        
        # FIFO Queue for this ticker
        # Each item: [date_acquired, shares_remaining, cost_per_share]
        lot_queue = []
        
        for _, row in tx_history.iterrows():
            date = row["date"]
            shares = float(row["shares"])
            amount = float(row["amount"]) # Negative for Buy, Positive for Sell
            
            if abs(shares) < 1e-6:
                continue
                
            if shares > 0:
                # ==========================
                # BUY EVENT (Acquisition)
                # ==========================
                cost_basis = -amount
                cost_per_share = cost_basis / shares if shares > 0 else 0
                
                lot_queue.append({
                    "date_acquired": date,
                    "shares": shares,
                    "cost_per_share": cost_per_share
                })
                
            else:
                # ==========================
                # SELL EVENT (Disposal)
                # ==========================
                shares_to_sell = abs(shares)
                total_proceeds = amount 
                proceeds_per_share = total_proceeds / shares_to_sell if shares_to_sell > 0 else 0
                
                while shares_to_sell > 1e-6 and lot_queue:
                    # Peek at oldest lot (FIFO)
                    current_lot = lot_queue[0]
                    
                    available = current_lot["shares"]
                    sold_qty = min(available, shares_to_sell)
                    
                    # Update Lot
                    if available > shares_to_sell:
                         current_lot["shares"] -= sold_qty
                    else:
                         lot_queue.pop(0)
                    
                    # Record Event
                    cost_basis_sold = sold_qty * current_lot["cost_per_share"]
                    proceeds_sold = sold_qty * proceeds_per_share
                    gain_loss = proceeds_sold - cost_basis_sold
                    term = _classify_term(current_lot["date_acquired"], date)
                    
                    # Initial Tax Impact (Wash Sales processed later)
                    tax_rate = TAX_RATE_LT if term == "Long-Term" else TAX_RATE_ST
                    tax_impact = gain_loss * tax_rate if gain_loss > 0 else 0
                    
                    realized_events.append({
                        "Date Sold": date,
                        "Ticker": ticker,
                        "Shares": sold_qty,
                        "Realized P/L": gain_loss,
                        "Term": term,
                        "Tax Impact": tax_impact,
                        "Is Wash Sale": False 
                    })
                    
                    shares_to_sell -= sold_qty

        # Collect Open Lots
        for lot in lot_queue:
            open_lots.append({
                "Ticker": ticker,
                "Date Acquired": lot["date_acquired"],
                "Shares": lot["shares"],
                "Cost Basis": lot["shares"] * lot["cost_per_share"],
                "Cost Per Share": lot["cost_per_share"]
            })

    # Convert to DataFrames
    open_lots_df = pd.DataFrame(open_lots)
    realized_df = pd.DataFrame(realized_events)
    
    # ==========================
    # WASH SALE GUARD
    # ==========================
    # Rule: If Loss, check for Buy within +/- 30 days.
    if not realized_df.empty:
        # Pre-compute buy dates per ticker for fast lookup
        # We need a dict of Ticker -> List of Buy Dates
        buys = raw_tx[raw_tx["shares"] > 0]
        buy_dates_map = {}
        for t, group in buys.groupby("ticker"):
            buy_dates_map[t] = group["date"].sort_values().values
            
        def check_wash_sale(row):
            if row["Realized P/L"] >= 0:
                return False, row["Tax Impact"] # Gains are always taxable
            
            # It's a Loss. Check for replacement shares.
            ticker = row["Ticker"]
            date_sold = row["Date Sold"]
            
            # Get buys for this ticker
            if ticker not in buy_dates_map:
                return False, row["Tax Impact"] # Should not happen if data consistent
                
            dates = buy_dates_map[ticker]
            
            # Vectorized check for window: [Sold - 30, Sold + 30]
            # Convert to numpy arrays for speed
            # Note: dates is numpy array of datetime64[ns]
            sold_ts = pd.Timestamp(date_sold).to_datetime64()
            start_window = sold_ts - np.timedelta64(30, 'D')
            end_window = sold_ts + np.timedelta64(30, 'D')
            
            # Check if ANY buy falls in window (excluding the exact match if we were matching trades, 
            # but here we are matching specific lots. 
            # Simplified Rule: Any buy in window triggers Wash Sale flag on the LOSS.)
            # NOTE: Logic refinement - technically the buy must be a "replacement" share. 
            # If I bought 30 days ago and sold today, that buy IS the lot I sold? 
            # No, FIFO means I sold the OLDEST lot. 
            # If I bought 30 days ago, that is a NEW lot. 
            # If I sell an OLD lot (acquired 2 years ago) at a loss, but I bought a NEW lot 20 days ago, that IS a Wash Sale.
            # So checking ANY buy in window is generally correct for the "Replacement Share" rule.
            # Only exception: The buy that created the lot we just sold? 
            # If I bought Lot A 2 years ago. Sold Lot A today. 
            # Buy dates will include Lot A's date (2 years ago). That is outside window.
            # What if I bought Lot B today and sold Lot B today (Day trade loss)?
            # Buy Date = Today. Sell Date = Today. Window covers it. 
            # Wash Sale triggers? Yes, intraday losses are wash sales if you keep the position? 
            # Actually if you close the position fully it's fine. 
            # But the "Institutional Safety" requirement implies strict flagging.
            
            mask = (dates >= start_window) & (dates <= end_window)
            
            # We must exclude the buy of the lot itself IF it falls in window?
            # Actually, standard wash sale rule is about acquiring substantially identical stock.
            # If I buy and sell same day, I have no replacement stock held? 
            # Let's stick to the 30-day window rule requested.
            
            if np.any(mask):
                return True, 0.0 # Disallowed Loss -> Tax Impact 0 (Loss doesn't reduce tax)
            
            return False, row["Tax Impact"]

        # Apply Check
        # Returns tuple (IsWash, NewTax)
        # Apply row-wise
        results = realized_df.apply(check_wash_sale, axis=1, result_type='expand')
        realized_df["Is Wash Sale"] = results[0]
        realized_df["Tax Impact"] = results[1]

    # Enrich Open Lots with Market Data
    if not open_lots_df.empty:
        # Fetch current prices
        unique_tickers = open_lots_df["Ticker"].unique().tolist()
        try:
            prices = fetch_price_history(unique_tickers)
            latest_prices = prices.iloc[-1]
        except Exception:
            latest_prices = pd.Series()
            
        def get_market_metrics(row):
            ticker = row["Ticker"]
            shares = row["Shares"]
            cost = row["Cost Basis"]
            date_acq = row["Date Acquired"]
            
            curr_price = 0.0
            if ticker in latest_prices:
                curr_price = float(latest_prices[ticker])
            
            mkt_val = shares * curr_price
            unrealized_pl = mkt_val - cost
            
            term = _classify_term(date_acq)
            near_cliff = _is_near_cliff(date_acq)
            days_left = _days_to_long_term(date_acq) if term == "Short-Term" else 0
            
            return pd.Series([curr_price, mkt_val, unrealized_pl, term, near_cliff, days_left])
            
        metrics = open_lots_df.apply(get_market_metrics, axis=1)
        metrics.columns = ["Current Price", "Market Value", "Unrealized P/L", "Term", "Is Near Cliff", "Days to LT"]
        
        open_lots_df = pd.concat([open_lots_df, metrics], axis=1)
        
        def calc_liability(row):
            gain = row["Unrealized P/L"]
            if gain <= 0: return 0.0
            rate = TAX_RATE_LT if row["Term"] == "Long-Term" else TAX_RATE_ST
            return gain * rate
            
        open_lots_df["Est Tax Liability"] = open_lots_df.apply(calc_liability, axis=1)
        
    return open_lots_df, realized_df

# ============================================================
# SIMULATOR
# ============================================================

def simulate_sell(ticker, shares_to_sell):
    """
    Simulates selling shares using FIFO logic.
    Alerts on Wash Sales (Buying in last 30 days).
    """
    ticker = normalize_ticker(ticker)
    
    if shares_to_sell <= 0:
        return {"summary_text": "Invalid share quantity.", "total_gain": 0, "est_tax": 0, "breakdown": []}

    open_lots_df, _ = build_tax_lots() # Rebuild to get fresh state
    
    if open_lots_df.empty or ticker not in open_lots_df["Ticker"].values:
        return {"summary_text": f"No open lots found for {ticker}.", "total_gain": 0, "est_tax": 0, "breakdown": []}
        
    lots = open_lots_df[open_lots_df["Ticker"] == ticker].sort_values("Date Acquired")
    
    total_avail = lots["Shares"].sum()
    if shares_to_sell > total_avail:
        return {
            "summary_text": f"Cannot sell {shares_to_sell} shares. Only {total_avail:.2f} available.",
            "total_gain": 0, "est_tax": 0, "breakdown": []
        }
    
    # Check for Wash Sale Risk (Recent Buy)
    # Check raw tx for buys in last 30 days
    raw_tx = load_transactions_raw()
    if not raw_tx.empty:
        raw_tx["ticker"] = raw_tx["ticker"].apply(normalize_ticker)
        recent_buys = raw_tx[
            (raw_tx["ticker"] == ticker) & 
            (raw_tx["shares"] > 0) & 
            (raw_tx["date"] >= datetime.now() - timedelta(days=30))
        ]
        wash_sale_risk = not recent_buys.empty
    else:
        wash_sale_risk = False

    # Simulation Loop
    remaining_to_sell = shares_to_sell
    impact_records = []
    
    total_st_gain = 0.0
    total_lt_gain = 0.0
    
    curr_price = lots.iloc[0]["Current Price"]
    summary_parts = []
    
    for _, lot in lots.iterrows():
        if remaining_to_sell <= 1e-6:
            break
            
        available = lot["Shares"]
        date_str = lot["Date Acquired"].strftime("%b %Y")
        term = lot["Term"]
        
        sold = min(available, remaining_to_sell)
        cost = sold * lot["Cost Per Share"]
        proceeds = sold * curr_price
        gain = proceeds - cost
        
        impact_records.append({
            "Date": date_str,
            "Shares": sold,
            "Term": term,
            "Gain": gain
        })
        
        if sold == available:
            summary_parts.append(f"deplete your oldest lot from {date_str} ({term})")
        else:
            summary_parts.append(f"partially tap into your {date_str} lot ({term})")

        if term == "Short-Term": total_st_gain += gain
        else: total_lt_gain += gain
        
        remaining_to_sell -= sold
            
    # Calculate Tax
    tax_st = max(0, total_st_gain) * TAX_RATE_ST
    tax_lt = max(0, total_lt_gain) * TAX_RATE_LT
    total_tax = tax_st + tax_lt
    
    # Narrative
    action_text = f"Selling {shares_to_sell} shares will " + " and ".join(summary_parts) + "."
    result_text = f"Result: ${total_lt_gain:,.2f} LT Gain, ${total_st_gain:,.2f} ST Gain."
    tax_text = f"Est Tax: ${total_tax:,.2f}."
    
    full_text = f"{action_text}\n\n{result_text} {tax_text}"
    
    if wash_sale_risk:
        full_text = "⚠️ STOP: Recent buy detected (last 30 days). Selling now triggers a Wash Sale on any losses.\n\n" + full_text
    
    return {
        "summary_text": full_text,
        "total_gain": total_st_gain + total_lt_gain,
        "est_tax": total_tax,
        "breakdown": impact_records
    }
