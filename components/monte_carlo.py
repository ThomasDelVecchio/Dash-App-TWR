import numpy as np
import pandas as pd

# Proxy Map for Historical Simulation
# Maps Asset Classes to liquid ETFs with long history
ASSET_CLASS_BENCHMARKS = {
    "US Large Cap": "SPY",
    "US Growth": "QQQ", 
    "US Small Cap": "IWM",
    "International Equity": "VXUS",
    "Fixed Income": "BND",
    "US Bonds": "BND",
    "Gold / Precious Metals": "GLD",
    "Digital Assets": "BTC-USD", # Use BTC if available, otherwise fallback logic handles it
    "CASH": "CASH"
}

def run_monte_carlo_simulation(
    current_value, 
    weights, 
    horizon_years=10, 
    n_simulations=1000, 
    monthly_contribution=0,
    correlation_matrix=None,
    risk_return=None,
    prices_df=None,
    random_seed=None,
    ticker_weights=None,
    holdings_map=None
):
    """
    Runs a Monte Carlo simulation.
    
    Modes:
    1. Historical Bootstrapping (Preferred): If prices_df is provided.
       - Constructs a synthetic historical return series.
       - If ticker_weights provided: Uses actual ticker history (precise).
       - If ticker_weights missing: Uses Asset Class Proxies (approximate).
       - Samples from this history (Rolling Monthly Returns) to simulate future paths.
       
    2. Parametric GBM (Fallback): If prices_df is None or invalid.
       - Uses Mean/Variance/Correlation Matrix to assume a Normal Distribution.
    
    Args:
        current_value (float): Starting portfolio value.
        weights (dict): Dictionary of {asset_class: weight_fraction} (Used for GBM & Proxy Fallback).
        horizon_years (int): Simulation duration in years.
        n_simulations (int): Number of paths to simulate.
        monthly_contribution (float): Monthly cash added to portfolio.
        correlation_matrix (dict): Optional {Asset: {Asset: Corr}} matrix (for GBM).
        risk_return (dict): Optional {Asset: {return: %, vol: %}} (for GBM).
        prices_df (pd.DataFrame): Optional 10-year daily price history.
        random_seed (int): Optional seed for reproducible results (CRN).
        ticker_weights (dict): Optional {Ticker: Weight}. Enables precise ticker-level history.
        holdings_map (dict): Optional {Ticker: AssetClass}. Used for proxying short-history tickers.
        
    Returns:
        dict: Simulation results (years, percentiles, final_distribution, metrics).
    """
    # 1. Setup
    n_steps = horizon_years * 12
    dt = 1/12
    
    # Set Seed if provided (CRN - Common Random Numbers)
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Initialize paths (N x M matrix)
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = current_value
    
    # 2. Try Historical Bootstrapping
    historical_pool = None
    
    if prices_df is not None and not prices_df.empty:
        try:
            # Construct Synthetic Portfolio History
            daily_rets = prices_df.pct_change()
            
            # Ensure we have a Fallback series (SPY) for gap filling
            if "SPY" in daily_rets.columns:
                fallback_ret = daily_rets["SPY"]
            elif not daily_rets.empty:
                 fallback_ret = daily_rets.iloc[:, 0] 
            else:
                 fallback_ret = None
                 
            if fallback_ret is not None:
                port_daily_ret = pd.Series(0.0, index=daily_rets.index)
                valid_history = False
                
                # --- STRATEGY A: PRECISE TICKER HISTORY ---
                if ticker_weights and holdings_map:
                    for ticker, w in ticker_weights.items():
                        if w <= 0 or ticker == "CASH": continue
                        
                        # Get Ticker Return
                        if ticker in daily_rets.columns:
                            t_ret = daily_rets[ticker]
                        else:
                            t_ret = pd.Series(np.nan, index=daily_rets.index)
                            
                        # GAP FILLING (The "Short History" Fix)
                        # If ticker has NaNs (e.g. FBTC), fill with Proxy based on Asset Class
                        if t_ret.isna().any():
                            ac = holdings_map.get(ticker, "US Large Cap")
                            proxy_ticker = ASSET_CLASS_BENCHMARKS.get(ac, "SPY")
                            
                            if proxy_ticker in daily_rets.columns:
                                proxy_ret = daily_rets[proxy_ticker]
                            else:
                                proxy_ret = fallback_ret
                                
                            t_ret = t_ret.fillna(proxy_ret)
                            
                        port_daily_ret += (t_ret * w)
                        valid_history = True
                        
                # --- STRATEGY B: ASSET CLASS PROXIES (Fallback) ---
                else:
                    for ac, weight in weights.items():
                        if weight <= 0: continue
                        
                        proxy = ASSET_CLASS_BENCHMARKS.get(ac, "SPY")
                        if proxy in daily_rets.columns:
                            ac_ret = daily_rets[proxy]
                        else:
                            ac_ret = fallback_ret
                        
                        ac_ret = ac_ret.fillna(fallback_ret)
                        port_daily_ret += (ac_ret * weight)
                        valid_history = True
                    
                if valid_history:
                    # Create Rolling Monthly Returns (21 days)
                    # Optimization: Use Log Returns for vectorization
                    # log(1+r) -> sum -> exp -> -1
                    log_rets = np.log1p(port_daily_ret)
                    rolling_log = log_rets.rolling(window=21).sum()
                    rolling_monthly = np.expm1(rolling_log)
                    
                    historical_pool = rolling_monthly.dropna().values
                    
                    if len(historical_pool) < 20: 
                        historical_pool = None
                        
        except Exception as e:
            print(f"Monte Carlo Bootstrapping Failed: {e}. Reverting to GBM.")
            historical_pool = None

    # 3. Calculate Portfolio Metrics (for GBM and Reporting)
    port_mu = 0.0
    total_w = sum(weights.values())
    if total_w <= 0: return None
    
    if risk_return:
        for ac, w in weights.items():
            w_norm = w / total_w
            if ac in risk_return:
                port_mu += w_norm * (risk_return[ac]['return'] / 100.0)
            
    port_sigma = calculate_portfolio_sigma(weights, correlation_matrix, risk_return)

    # 4. Run Simulation Loop
    if historical_pool is not None:
        # --- BOOTSTRAPPING MODE ---
        for t in range(1, n_steps + 1):
            sampled_returns = np.random.choice(historical_pool, size=n_simulations, replace=True)
            paths[:, t] = paths[:, t-1] * (1 + sampled_returns) + monthly_contribution
            
    else:
        # --- GBM MODE (Fallback) ---
        drift = (port_mu - 0.5 * port_sigma**2) * dt
        shock_std = port_sigma * np.sqrt(dt)
        
        for t in range(1, n_steps + 1):
            z = np.random.normal(0, 1, n_simulations)
            growth_factor = np.exp(drift + shock_std * z)
            paths[:, t] = paths[:, t-1] * growth_factor + monthly_contribution
        
    # 5. Compile Results
    percentiles_over_time = {
        '10': np.percentile(paths, 10, axis=0),
        '50': np.percentile(paths, 50, axis=0),
        '90': np.percentile(paths, 90, axis=0)
    }
    
    # Calculate Tail Risk Metrics (VaR & CVaR)
    final_values = paths[:, -1]
    
    # VaR 95% (5th percentile floor)
    var_95 = np.percentile(final_values, 5)
    
    # CVaR 95% (Mean of the worst 5%)
    tail_values = final_values[final_values <= var_95]
    cvar_95 = tail_values.mean() if len(tail_values) > 0 else var_95
    
    return {
        'years': np.linspace(0, horizon_years, n_steps + 1).tolist(),
        'percentiles': {k: v.tolist() for k, v in percentiles_over_time.items()},
        'final_distribution': paths[:, -1].tolist(),
        'metrics': {
            'mu': port_mu,
            'sigma': port_sigma,
            'mode': 'Historical Bootstrapping' if historical_pool is not None else 'Parametric GBM',
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    }

def execute_trade_batch(current_weights, current_value, trades, holdings_map):
    """
    Executes a sequence of trades against the portfolio.
    
    Args:
        current_weights (dict): {AssetClass: Weight} (0-1).
        current_value (float): Total Portfolio Value.
        trades (list): List of dicts [{'ticker': 'SPY', 'side': 'Buy', 'amount': 1000}, ...].
        holdings_map (dict): {Ticker: AssetClass}.
        
    Returns:
        dict: New Weights {AssetClass: Weight}.
    """
    # 1. Expand weights to Value Map
    ac_values = {k: v * current_value for k, v in current_weights.items()}
    if "CASH" not in ac_values:
        ac_values["CASH"] = 0.0
        
    # 2. Process Batch
    for trade in trades:
        ticker = trade.get('ticker')
        side = trade.get('side')
        amount = float(trade.get('amount', 0))
        target_ticker = trade.get('target_ticker', None) # For Swap
        
        # Identify Asset Classes
        # Source Asset (for Sell/Swap) OR Target Asset (for Buy)
        primary_ac = holdings_map.get(ticker, "US Large Cap") # Default if unknown
        
        if side == "Buy":
            # Cash -> Asset
            # Check Cash availability? We allow dropping to 0 but logic says "prevent dropping BELOW zero"
            actual_amount = min(ac_values["CASH"], amount) if amount > ac_values["CASH"] else amount
            
            ac_values["CASH"] = max(0.0, ac_values["CASH"] - actual_amount)
            ac_values[primary_ac] = ac_values.get(primary_ac, 0.0) + actual_amount
            
        elif side == "Sell":
            # Asset -> Cash
            current_ac_val = ac_values.get(primary_ac, 0.0)
            actual_amount = min(current_ac_val, amount) if amount > current_ac_val else amount
            
            ac_values[primary_ac] = max(0.0, current_ac_val - actual_amount)
            ac_values["CASH"] += actual_amount
            
        elif side == "Swap":
            # Asset A -> Asset B
            sell_ac = primary_ac
            buy_ac = holdings_map.get(target_ticker, "US Large Cap")
            
            current_sell_val = ac_values.get(sell_ac, 0.0)
            actual_amount = min(current_sell_val, amount) if amount > current_sell_val else amount
            
            ac_values[sell_ac] = max(0.0, current_sell_val - actual_amount)
            ac_values[buy_ac] = ac_values.get(buy_ac, 0.0) + actual_amount
            
    # 3. Recompute Weights
    total_new_val = sum(ac_values.values())
    if total_new_val <= 0: total_new_val = 1.0 # Prevent div/0
    
    new_weights = {k: v / total_new_val for k, v in ac_values.items()}
    
    return new_weights

def calculate_portfolio_sigma(weights, correlation_matrix=None, risk_return=None):
    """
    Calculates portfolio volatility using Correlation Matrix if available,
    otherwise falls back to weighted average.
    """
    total_w = sum(weights.values())
    if total_w <= 0: return 0.0
    
    # If no risk data provided, assume zero risk
    if not risk_return:
        return 0.0

    if correlation_matrix:
        # Variance = w.T * Cov * w
        assets = list(weights.keys())
        valid_assets = [a for a in assets if a in risk_return]
        
        if not valid_assets:
            return 0.0
            
        w_vec = []
        vol_vec = []
        
        for ac in valid_assets:
            w_norm = weights[ac] / total_w
            w_vec.append(w_norm)
            vol_vec.append(risk_return[ac]['vol'] / 100.0)
            
        w_vec = np.array(w_vec)
        vol_vec = np.array(vol_vec)
        
        # Build Correlation Matrix (N x N)
        n = len(valid_assets)
        corr_mat = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                asset_i = valid_assets[i]
                asset_j = valid_assets[j]
                
                rho = 0.0
                if i == j:
                    rho = 1.0
                else:
                    # Try to find in provided matrix (nested dict)
                    # Supports both {A: {B: 0.5}} and {B: {A: 0.5}}
                    if asset_i in correlation_matrix and asset_j in correlation_matrix[asset_i]:
                        rho = correlation_matrix[asset_i][asset_j]
                    elif asset_j in correlation_matrix and asset_i in correlation_matrix[asset_j]:
                            rho = correlation_matrix[asset_j][asset_i]
                    else:
                        rho = 0.0 # Uncorrelated fallback
                
                corr_mat[i, j] = rho
        
        D = np.diag(vol_vec)
        cov_mat = D @ corr_mat @ D
        port_var = w_vec.T @ cov_mat @ w_vec
        return np.sqrt(port_var)
        
    else:
        # Fallback: Weighted Average
        sigma = 0.0
        for ac, w in weights.items():
            w_norm = w / total_w
            if ac in risk_return:
                sigma += w_norm * (risk_return[ac]['vol'] / 100.0)
        return sigma

def calculate_trade_impact(current_weights, current_value, trade_ticker, trade_amount, trade_side, holdings_map, prices_df, swap_target_ticker=None, risk_return=None, correlation_matrix=None):
    """
    BACKWARD COMPATIBILITY WRAPPER
    Translates legacy single-trade arguments into a batch for the new engine.
    """
    # Construct Batch
    trade_struct = {
        "ticker": trade_ticker,
        "amount": trade_amount,
        "side": trade_side,
        "target_ticker": swap_target_ticker
    }
    
    batch = [trade_struct]
    
    return execute_trade_batch(current_weights, current_value, batch, holdings_map)
