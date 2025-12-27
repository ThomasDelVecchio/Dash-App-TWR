
# Centralized Knowledge Base for Help Index & Chatbot

HELP_TOPICS = {
    "config_guide": {
        "title": "Configuration Guide (config.py)",
        "content": r"""
The `config.py` file controls global settings and targets for the application.

#### **Key Variables**

* **`TARGET_PORTFOLIO_VALUE`**:
    * *Description*: The goal amount for the portfolio (e.g., $50,000).
    * *Usage*: Used in visualizations to show progress against a target.

* **`TARGET_MONTHLY_CONTRIBUTION`**:
    * *Description*: The expected monthly deposit amount (e.g., $400).
    * *Usage*: Used in **Projections** and **Monte Carlo** simulations to model future growth.

* **`RISK_FREE_RATE`**:
    * *Description*: The annual risk-free rate (e.g., 0.04 for 4%).
    * *Usage*: Used as the hurdle rate for **Sharpe** and **Sortino** ratio calculations.

* **`FMP_API_KEY`**:
    * *Description*: Your API key for Financial Modeling Prep.
    * *Usage*: Used to fetch dynamic **Sector Allocations** for ETFs (e.g., looking up that SPY is 30% Tech).
        """
    },

    "twr": {
        "title": "Time-Weighted Return (TWR)",
        "content": r"""
**Time-Weighted Return (TWR)** is the standard method for measuring investment manager performance. It eliminates the distorting effects of external cash flows (deposits and withdrawals), reflecting only the investment growth.

#### **Calculation Methodology**

**1. Daily Return Calculation**

$$r_t = \frac{PV_t - (PV_{t-1} + F_t)}{PV_{t-1} + F_t}$$

**Where:**
*   $PV_t$: Portfolio Value at the end of day $t$
*   $PV_{t-1}$: Portfolio Value at the end of the previous day $t-1$
*   $F_t$: Net External Flows (Deposits/Withdrawals) on day $t$
*   **Note**: Flows are assumed to occur at the **Start of Day** (SOD). The denominator represents the total capital available for investment during the day.

**2. Geometric Linking (Chain-Linking)**

$$r_{\text{TWR}} = \left[\prod_{t=1}^{n}(1 + r_t)\right] - 1$$

All daily returns are geometrically linked to produce the total period return.
        """
    },

    "dietz_ticker": {
        "title": "Modified Dietz (Security Level)",
        "content": r"""
**Modified Dietz** is a money-weighted return calculation used for individual securities. It accounts for the timing and size of flows, making it appropriate for assessing position-level performance where the manager controls the timing of trades.

#### **Formula**

$$r_{\text{ModDietz}} = \frac{V_1 - V_0 - C + I}{V_0 + \sum_{i=1}^{n}(C_i \times w_i)}$$

**Where:**
* $V_1$: Market Value at the end of the period
* $V_0$: Market Value at the start of the period
* $C$: Net Capital Flows (Cost of Buys - Proceeds from Sells)
* $I$: Income (Dividends received)
* $w_i$: Time-weighting factor for each flow, representing the fraction of the period the capital was invested

**Time-Weighting Factor:**

$$w_i = \frac{D_{\text{total}} - D_i}{D_{\text{total}}}$$

**Where:**
*   $D_{\text{total}}$: Total days in the period
*   $D_i$: Days elapsed since the start of the period until flow $i$
        """
    },

    "dietz_asset": {
        "title": "Modified Dietz (Asset Class Level)",
        "content": r"""
Calculates the return for an entire Asset Class (e.g., "US Large Cap") by aggregating the values and flows of all constituent tickers.

#### **Methodology**

The formula is identical to the Security Level Modified Dietz, but the inputs are aggregated across all securities in the class:

$$r_{\text{AC}} = \frac{\sum V_{1,i} - \sum V_{0,i} - \sum CF_i}{\sum V_{0,i} + \sum(w_i \times CF_i)}$$

**Where:**
*   **Numerator**: Total Gain/Loss in dollar terms
*   **Denominator**: Average Weighted Capital Employed
        """
    },

    "attribution": {
        "title": "Daily Attribution",
        "content": r"""
Attribution analysis decomposes the daily change in Portfolio Value into its core components.

#### **Decomposition**

$$\Delta PV = PV_t - PV_{t-1}$$

This total change is split into:

**1. External Flows**

Money entering or leaving the portfolio (Deposits/Withdrawals).

**2. Market Effect**

The pure investment result:

$$M = \Delta PV - F_{\text{ext}}$$

$$M = (PV_t - PV_{t-1}) - F_{\text{ext}}$$

#### **Asset Class Breakdown**

The Market Effect is further broken down by Asset Class:

$$M_{\text{AC}} = (V_{t,\text{AC}} - V_{t-1,\text{AC}}) - F_{\text{int,AC}} + D_{\text{AC}}$$

**Where:**
*   $V$: Market Value of the Asset Class
*   $F_{\text{int}}$: Net Internal Flows (Buys/Sells) for that Asset Class
*   $D$: Dividends received by that Asset Class
        """
    },
    "ctr_methodologies": {
        "title": "Contribution to Return (CTR) Methodologies",
        "content": r"""
The report uses two different methodologies to calculate Contribution to Return (CTR), each suited for different analytical purposes.

#### **1. Daily Attribution (Simplified Method)**

*   **Used In**: The "Daily Breakdown" chart on the Attribution page when the main chart is in Daily mode.
*   **Methodology**: This is a simple arithmetic calculation for a single day. The formula for the **Contribution (%)** of a given asset class is:

$$C_i = \frac{M_i}{PV_{t-1} + F_{ext,t}}$$

*   **Where**:
    *   $C_i$: Contribution of asset class $i$
    *   $M_i$: Market Effect of asset class $i$
    *   $PV_{t-1}$: Portfolio Value at the end of the previous day
    *   $F_{ext,t}$: External flows on the current day

The **Market Effect** for the asset class ($M_i$) is calculated as:

$$M_i = (V_{t,i} - V_{t-1,i}) - F_{int,i} + D_i$$

*   **Where**:
    *   $V_{t,i}$: Market value of asset class $i$ at the end of the day
    *   $V_{t-1,i}$: Market value of asset class $i$ at the end of the previous day
    *   $F_{int,i}$: Net internal flows (buys/sells) for asset class $i$
    *   $D_i$: Dividends for asset class $i$

*   **Purpose**: This method is ideal for understanding the drivers of performance on a specific day. However, because it is not geometrically linked, the sum of daily contributions will not equal the total portfolio return over longer periods.

#### **2. Multi-Period Attribution (Frongello Linking)**

*   **Used In**: The "Since Inception" summary, as well as the Weekly and Monthly breakdown views.
*   **Methodology**: This method uses the Frongello linking algorithm, a sophisticated technique that geometrically links daily asset class returns. The linked contribution for an asset class ($C_{i,T}$) over a period $T$ is calculated by summing the daily linked contributions:

$$C_{i,T} = \sum_{t=1}^{T} c_{i,t} \times \left( \prod_{j=1}^{t-1} (1 + R_j) \right)$$

*   **Where**:
    *   $c_{i,t}$: The simple daily contribution of asset class $i$ on day $t$, as calculated in the simplified method.
    *   $R_j$: The total portfolio return on day $j$.

*   **Purpose**: This is the industry-standard method for performance attribution over time, as it accurately accounts for the compounding of returns and ensures that the sum of the contributions from each asset class equals the total portfolio return over the entire period.
        """
    },
    "monte_carlo": {
        "title": "Monte Carlo Simulation",
        "content": r"""
The Monte Carlo engine projects future portfolio value by simulating thousands of possible market paths. It uses **two distinct modes**:

#### **1. Historical Bootstrapping (Preferred)**

Uses actual historical data to preserve real-world non-normality (fat tails).

*   **Method**: Randomly samples monthly returns from the portfolio's reconstructed history
*   **Path Formula**:

$$P_t = P_{t-1} \times (1 + r_{\text{sampled}}) + C_{\text{monthly}}$$

#### **2. Parametric GBM (Fallback)**

Used if historical prices are unavailable. Assumes returns follow a Geometric Brownian Motion (log-normal distribution).

$$P_t = P_{t-1} \cdot e^{\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\,Z}$$

**Where:**
*   $\mu$: Expected Return (Drift) from `config.RISK_RETURN`
*   $\sigma$: Portfolio Volatility from `config.CORRELATION_MATRIX`
*   $Z$: Random variable from Standard Normal Distribution $\mathcal{N}(0,1)$
*   $\Delta t$: Time step (e.g., $1/12$ for monthly)

#### **Risk Metrics**

*   **VaR 95%** (Value at Risk): The 5th percentile of simulated outcomes
*   **CVaR 95%** (Conditional Value at Risk): The average value of the worst 5% of outcomes (Expected Tail Loss)
        """
    },

    "projections": {
        "title": "Long-Term Projections",
        "content": r"""
Calculates the deterministic Future Value ($FV$) of the portfolio under fixed return assumptions.

#### **1. Lump Sum Growth**

Growth of the current portfolio value ($PV_0$) without additional contributions:

$$FV_{\text{lump}} = PV_0 \times (1 + r)^n$$

#### **2. Periodic Contributions**

Growth of monthly contributions ($C$) added at the end of each month:

$$FV_{\text{contrib}} = C \times \frac{\left(1 + \frac{r}{12}\right)^{12n} - 1}{\frac{r}{12}}$$

#### **Total Future Value**

$$FV_{\text{total}} = FV_{\text{lump}} + FV_{\text{contrib}}$$
        """
    },

    "growth_capital": {
        "title": "Growth of Invested Capital",
        "content": r"""
Visualizes the relationship between the money you put in versus what it is worth now.

#### **Net Invested Capital (Dashed Line)**

The cumulative sum of all External Flows (Deposits - Withdrawals):

$$I_t = \sum_{i=0}^{t}(\text{Deposits}_i - \text{Withdrawals}_i)$$

#### **Portfolio Value (Stacked Area)**

The actual Mark-to-Market value of the portfolio:

$$PV_t = \sum(\text{Shares}_i \times \text{Price}_i) + \text{Cash}$$

#### **Investment Profit**

$$\text{Profit}_t = PV_t - I_t$$
        """
    },

    "cash_recon": {
        "title": "Cash / Recon P/L",
        "content": r"""
A balancing item ensuring the sum of all parts equals the total.

$$P/L_{\text{Recon}} = P/L_{\text{Total}} - \sum P/L_{\text{Tickers}}$$

This residual captures:
*   Interest on cash
*   Dividends in transit / unallocated
*   Fees not attributed to a specific ticker
*   Small rounding differences between Transaction-based P/L and Flow-based P/L
        """
    },

    "horizon_gating": {
        "title": "Horizon Gating & Time Period Logic",
        "content": r"""
To ensure the accuracy and integrity of performance metrics, the application applies specific "horizon gating" rules per **GIPS® standards**. This means that calculations are only performed and displayed when sufficient and meaningful data is available for a given time period. These rules vary by metric and are applied consistently across all levels (Portfolio, Asset Class, and Ticker).

The **End Date** for all calculations is the most recent date for which market data is available, referred to as the "as-of" date. The **Start Date** logic is more complex and depends on the horizon and the level of analysis.

#### **Portfolio Level (TWR & P/L)**

The start date for portfolio-level calculations is determined as follows:

*   **Since Inception (`SI`):** The start date is the `inception_date` of the portfolio, which is the date of the first economic activity (either a deposit or a trade). At inception, **Start Value = $0** and the initial funding is treated as a flow.
*   **Month-to-Date (`MTD`):** The start date is the last trading day of the previous calendar month. The calculation is only performed if the portfolio's inception was on or before this start date.
*   **Year-to-Date (`YTD`):** The start date is January 1st of the current year. This calculation is only performed if the portfolio was active on or before January 1st.
*   **Fixed Periods (`1D`, `1W`, `1M`, `3M`, `6M`, `1Y`, `3Y`, `5Y`):** The start date is calculated by looking back from the "as-of" date by the corresponding period. A key gating rule is that the portfolio must have existed for the entire duration of the period. For example, a "1Y" return is only calculated if the portfolio has been active for at least 365 days.

#### **Ticker & Asset Class Level (Modified Dietz & P/L)**

The start date logic at the Ticker and Asset Class level generally follows the portfolio-level rules, but with **GIPS-compliant gating** to ensure that only securities with full horizon history are included.

*   **Since Inception (`SI`):**
    *   **Ticker:** The start date is the **portfolio inception date** (not the ticker's first trade). At inception, the ticker has **Start Value = $0** and **Shares = 0**. The first trade is captured as a flow into the position.
    *   **Asset Class:** The start date is the portfolio inception date. Returns are calculated across all tickers in the class that had any activity during the SI period.
*   **Other Horizons (MTD, 1M, 1Y, etc.):**
    *   The initial start date is determined using the same logic as the portfolio level.
    *   **GIPS Gating:** A crucial check is performed. A ticker or asset class return is only calculated if the security's **first trade was STRICTLY BEFORE** the horizon start date. This prevents inflated returns from positions opened mid-period (where V₀=0 would make any gain appear as infinite return).
    *   **Example:** A "1M" return for US Large Cap will only include VOO if VOO's first trade was before the 1M horizon start. If SPMO was bought 2 weeks ago, it will NOT be included in the 1M Asset Class return.

#### **Why STRICTLY BEFORE (Not On-Or-Before)?**

Per GIPS® standards, a security must have a **known starting value** to calculate a meaningful return. If a security is purchased ON the horizon start date:
- V₀ would be the cost of the purchase (not a market value from the prior period)
- The denominator would conflate capital invested with capital at risk
- This violates the time-weighting principle

By requiring first trade **STRICTLY BEFORE** the horizon start, we ensure V₀ reflects an actual market value at the start of the measurement period.

#### **Annualization**

In line with GIPS standards, returns are only annualized for periods of one year or longer (e.g., `1Y`, `3Y`, `5Y`, and `SI` if the period is > 1 year). Short-term returns (`1D`, `1W`, `MTD`, `1M`, `3M`, `6M`, `YTD`) are never annualized to avoid misleading projections.

By enforcing these detailed gating rules, the application ensures that all displayed returns are meaningful, comparable, and prevent common misinterpretations of performance data.
        """
    },
    
    "risk_metrics": {
        "title": "Risk Metrics (Sharpe & Sortino)",
        "content": r"""
Risk-adjusted metrics measure how much return you are generating for each unit of risk taken.

#### **1. Sharpe Ratio**

$$\text{Sharpe} = \frac{R_p - R_f}{\sigma_p}$$

*   **Numerator**: Excess Return (Portfolio Return $R_p$ minus Risk-Free Rate $R_f$).
*   **Denominator**: Total Volatility (Standard Deviation of Daily Returns $\sigma_p$).
*   **Interpretation**: The classic measure of efficiency. A ratio > 1.0 is considered good; > 2.0 is excellent. It penalizes **all** volatility, both upside and downside.

#### **2. Sortino Ratio**

$$\text{Sortino} = \frac{R_p - R_f}{\sigma_{down}}$$

*   **Numerator**: Excess Return (same as Sharpe).
*   **Denominator**: Downside Deviation ($\sigma_{down}$). This only considers returns that fall below 0% (or the target).
*   **Interpretation**: A better measure for strategies where upside volatility is desirable. It only penalizes "bad" volatility (losses).

*Note: All calculations assume a Risk-Free Rate ($R_f$) as defined in `config.py`.*
        """
    },

    "drawdown": {
        "title": "Drawdown Analysis",
        "content": r"""
Drawdown measures the decline from a historical peak in portfolio value. It provides a realistic view of loss exposure.

#### **Definitions**

*   **High-Water Mark (HWM)**: The highest peak value the portfolio has ever reached.
*   **Drawdown (%)**: The percentage drop from the HWM to the current value.
$$\text{DD}_t = \frac{PV_t - \text{HWM}_t}{\text{HWM}_t}$$
*   **Maximum Drawdown**: The deepest trough (largest percentage loss) experienced over the selected period.
*   **Recovery Period**: The number of days it took for the portfolio to reclaim its previous HWM after a drawdown.

*Note: Drawdown is calculated on the **Time-Weighted Return (TWR) Curve** (Growth of $1), ensuring that withdrawals are not mistaken for investment losses.*
        """
    },

    "active_risk": {
        "title": "Active Risk (Beta & Tracking Error)",
        "content": r"""
Metrics used to evaluate the portfolio's behavior relative to a benchmark (e.g., S&P 500).

#### **1. Beta ($\beta$)**

$$\beta = \frac{\text{Cov}(R_p, R_b)}{\text{Var}(R_b)}$$

*   **Measures**: Sensitivity to market movements.
*   **Interpretation**:
    *   $\beta = 1.0$: Portfolio moves in sync with the market.
    *   $\beta > 1.0$: Portfolio is **more volatile** (aggressive) than the market.
    *   $\beta < 1.0$: Portfolio is **less volatile** (defensive) than the market.

#### **2. Tracking Error (Active Risk)**

$$\text{TE} = \sigma(R_p - R_b)$$

*   **Measures**: The standard deviation of the **excess returns** (Portfolio - Benchmark).
*   **Interpretation**: Indicates how much the portfolio deviates from the benchmark.
    *   **Low TE**: The portfolio closely mimics the index (Passive/Closet Indexing).
    *   **High TE**: The manager is taking active bets different from the index (Active Management).
        """
    }
}
