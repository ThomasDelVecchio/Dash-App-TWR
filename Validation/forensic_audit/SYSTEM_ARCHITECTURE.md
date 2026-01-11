# DELVEX Portfolio Analytics — System Architecture

> **A GIPS-Compliant Multi-Page Dash Application for Institutional-Grade Performance, Risk, and Attribution Analysis**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Architectural Philosophy](#core-architectural-philosophy)
3. [Layer-by-Layer Breakdown](#layer-by-layer-breakdown)
4. [UI & Feature Breakdown](#ui--feature-breakdown)
5. [Key Technical Innovations](#key-technical-innovations)
6. [Visual Flowcharts](#visual-flowcharts)
7. [Deployment & Usage](#deployment--usage)
8. [Code Quality & Standards](#code-quality--standards)

---

## Executive Summary

**DELVEX Portfolio Analytics** is a production-grade Python application designed for portfolio managers, financial advisors, and individual investors who demand institutional-quality performance measurement.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **GIPS-Compliant TWR** | Time-Weighted Return calculations that meet Global Investment Performance Standards |
| **Modified Dietz Attribution** | Security and asset-class level money-weighted returns for precise P/L attribution |
| **Time Machine Analysis** | Point-in-time portfolio reconstruction for historical "what-if" scenarios |
| **Monte Carlo Simulation** | Historical bootstrapping engine for probabilistic outcome modeling |
| **Tax Authority** | Multi-strategy tax lot tracking (FIFO/LIFO/HIFO) with wash sale detection |
| **Dynamic Sector Analysis** | Look-through ETF decomposition using multi-source metadata (FMP/YF) |

<div style="page-break-before: always;"></div>

### Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend: Dash + Plotly + Dash Bootstrap Components        │
│  Styling:  Cyborg Theme (Dark Mode) + Custom CSS            │
├─────────────────────────────────────────────────────────────┤
│  Backend:  Python 3.10+ | Pandas | NumPy                    │
│  Data:     yfinance (Prices) | FMP API (ETF Sectors)        │
├─────────────────────────────────────────────────────────────┤
│  Reports:  python-docx (DOCX) | Plotly Static Export        │
└─────────────────────────────────────────────────────────────┘
```

---


## Core Architectural Philosophy

This application follows a strict **5-Layer Separation of Concerns** pattern, ensuring that each module has a single responsibility and can be tested, maintained, and extended independently.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          UI LAYER                                   │
│         app.py  |  pages/*.py  |  components/*.py                   │
│         (Dash Layouts, Callbacks, User Interaction)                 │
├─────────────────────────────────────────────────────────────────────┤
│                        WRAPPER LAYER                                │
│                      dash_wrappers.py                               │
│         (Figure Generation, Caching, UI-Engine Bridge)              │
├─────────────────────────────────────────────────────────────────────┤
│                        ENGINE LAYER                                 │
│                     portfolio_engine.py                             │
│         (Orchestration, Time Machine, Pipeline Control)             │
├─────────────────────────────────────────────────────────────────────┤
│                         MATH LAYER                                  │
│                      financial_math.py                              │
│         (Pure Math: TWR, Modified Dietz, Annualization)             │
├─────────────────────────────────────────────────────────────────────┤
│                         DATA LAYER                                  │
│                       data_loader.py                                │
│         (CSV Ingestion, yfinance, Metadata Cache)                   │
└─────────────────────────────────────────────────────────────────────┘
```

<div style="page-break-before: always;"></div>

### Design Principles

1. **Single Source of Truth**: All financial math lives in `financial_math.py`. No calculations are duplicated in UI files.
2. **Immutable Data Flow**: Raw data flows upward through transformations; no layer modifies data belonging to a lower layer.
3. **Testability**: Each layer can be unit-tested in isolation (see `Validation/forensic_audit/`).
4. **Cacheability**: Expensive operations are cached at the appropriate layer (`_DATA_CACHE`, `_PRICE_CACHE`, `_METADATA_CACHE`).

---

## Layer-by-Layer Breakdown

### 1. Data Layer — `data_loader.py`

**Responsibility**: Raw data ingestion, external API calls, and metadata management.

#### Core Functions

| Function | Purpose |
|----------|---------|
| `load_holdings()` | Parses `sample holdings.csv` for current positions |
| `load_cashflows_external()` | Parses `cashflows.csv` for deposits/withdrawals |
| `load_transactions_raw()` | Extracts trade-level data (buys/sells) from cashflows |
| `fetch_price_history(tickers)` | Retrieves 10-year daily OHLC via yfinance |
| `fetch_etf_sectors(ticker)` | Multi-source sector weight resolution |

#### Multi-Source Metadata Resolution

The sector attribution system implements a **waterfall resolution pattern**:

```python
def fetch_etf_sectors(ticker: str) -> dict:
    """
    Priority:
    1. Local Cache (metadata_cache.json)
    2. FMP API (Financial Modeling Prep)
    3. yfinance funds_data
    4. Equity Fallback (Single sector = 100%)
    """
```

This ensures resilience against API failures while maintaining data consistency.

<div style="page-break-before: always;"></div>

#### Caching Strategy

```python
# In-memory price cache (session lifetime)
_PRICE_CACHE = {}

# Persistent metadata cache (JSON file)
_METADATA_CACHE = {}  # Loaded from metadata_cache.json
```

---

### 2. Math Layer — `financial_math.py`

**Responsibility**: The **Source of Truth** for all financial calculations. This module contains pure functions with no side effects.

#### Core Algorithms

##### Time-Weighted Return (TWR)

The TWR implementation follows GIPS standards for portfolio-level performance:

```python
def compute_period_twr(pv, cf, start_date, end_date) -> float:
    """
    True TWR using daily chain-linking.
    
    Formula:
        TWR = Π(1 + r_i) - 1
        
    Where r_i is the daily return adjusted for external flows:
        r_i = (PV_end - PV_start - Flow) / (PV_start + Flow)
    
    Key Compliance Points:
    - Flows applied at START of day (GIPS-correct)
    - Daily factor chain captures timing of capital movements
    """
```
<div style="page-break-before: always;"></div>

##### Modified Dietz (Security-Level Attribution)

For money-weighted returns at the security level:

```python
def modified_dietz_for_ticker_window(ticker, transactions, prices, ...) -> float:
    """
    Modified Dietz Formula:
    
        MD = (EMV - BMV - CF) / (BMV + Σ(CF_i × W_i))
        
    Where:
        EMV = Ending Market Value
        BMV = Beginning Market Value  
        CF  = Net Cash Flows
        W_i = Time-weighting factor = (T - t_i) / T
    """
```

##### Universal Annualization Logic Gate

The system enforces a single, consistent rule for annualization:

```python
def annualize_return(r_cum, start_date, end_date) -> float:
    """
    Logic Gate:
    - Duration > 365 days → Annualize using CAGR
    - Duration ≤ 365 days → Return Cumulative
    
    This prevents misleading annualized figures for short periods.
    """
    years = days / 365.25
    
    if years > 1.0:
        return (1.0 + r_cum) ** (1.0 / years) - 1.0
    return r_cum
```

#### Supported Horizons

```python
HORIZONS = ["1D", "1W", "MTD", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"]
```

Each horizon has specific gating logic to prevent misleading calculations when insufficient data exists.

---

<div style="page-break-before: always;"></div>

### 3. Engine Layer — `portfolio_engine.py`

**Responsibility**: Orchestrates the data-to-output pipeline and implements the **Time Machine** feature.

#### Primary Entry Point

```python
def run_engine(end_date=None):
    """
    Main calculation pipeline.
    
    Returns:
        twr_df          : Horizon-level TWR table
        sec_table       : Security-level Modified Dietz returns + metadata
        class_df        : Asset-class level Modified Dietz returns
        pv              : Daily Portfolio Value series
        twr_si          : Since-Inception TWR
        twr_si_ann      : Since-Inception TWR (annualized if > 1 year)
        pl_si           : Since-Inception P/L ($)
    """
```

#### Time Machine Logic

The "Time Machine" enables **point-in-time portfolio reconstruction**, allowing users to analyze the portfolio as it existed on any historical date.

```python
if end_date is not None:
    # 1. Clip Transactions & Flows
    transactions_raw = transactions_raw[transactions_raw["date"] <= end_date]
    cashflows_ext = cashflows_ext[cashflows_ext["date"] <= end_date]
    
    # 2. Reconstruct Holdings at end_date
    computed_shares = transactions_raw.groupby("ticker")["shares"].sum()
    
    # 3. Rebuild Cash Balance
    total_cash = ext_cash + trading_cash + div_cash
    
    # 4. Write temporary clipped cashflows for PV reconstruction
    # ...ensures build_portfolio_value_series_from_flows is consistent
```

This enables:
- Historical performance snapshots
- "What if I had sold on date X?" analysis
- Forensic auditing of past decisions

---

<div style="page-break-before: always;"></div>

### 4. Wrapper Layer — `dash_wrappers.py`

**Responsibility**: Bridge between the Engine and UI layers. Manages caching and generates Plotly visualizations.

#### Server-Side Data Cache

```python
_DATA_CACHE = None

def get_data():
    """Retrieve cached data, initializing if necessary."""
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = run_analytics_engine()
    return _DATA_CACHE

def refresh_data(end_date=None):
    """Force refresh of the data cache (e.g., on date picker change)."""
    global _DATA_CACHE
    _DATA_CACHE = run_analytics_engine(end_date=end_date)
    return _DATA_CACHE
```

#### Key Visualization Functions

| Function | Output |
|----------|--------|
| `get_cumulative_return_chart()` | Portfolio vs. Benchmark cumulative returns |
| `get_excess_return_chart()` | Alpha visualization over benchmarks |
| `get_risk_return_scatter()` | Volatility vs. Return bubble chart |
| `get_drawdown_chart()` | Underwater equity curve |
| `get_smart_attribution_chart()` | Periodic P/L attribution (daily/weekly/monthly) |
| `get_monte_carlo_overlay()` | Probabilistic outcome cones |
| `get_correlation_heatmap()` | 90-day rolling correlation matrix |

---

<div style="page-break-before: always;"></div>

### 5. UI Layer — `app.py` & `pages/*.py`

**Responsibility**: User interface, routing, and callback orchestration.

#### Application Structure

```
app.py                  # Main entry point, layout, global callbacks
pages/
├── overview.py         # KPI dashboard, AI brief
├── performance.py      # Return charts, horizon tables
├── allocations.py      # Pie charts, weight analysis
├── attribution.py      # P/L breakdown by asset class
├── holdings.py         # Security-level detail table
├── risk.py             # Risk metrics, projections, simulator
├── trade_lab.py        # What-If Monte Carlo simulator
├── rebalancing.py      # Tax-aware rebalancing & deployment
├── taxes.py            # Tax lot analysis, harvesting radar
├── flows.py            # Cash flow timeline
├── settings.py         # Configuration management
└── help_index.py       # Documentation/help pages
```

#### Theming

The application uses **Dash Bootstrap Components** with the **Cyborg** dark theme:

```python
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="Portfolio Analytics"
)
```

Custom styles are defined in `assets/styles.css` for additional refinements.

---

<div style="page-break-before: always;"></div>

## UI & Feature Breakdown

### Overview Page

**Purpose**: Executive-level health check with real-time KPIs.

| Component | Description |
|-----------|-------------|
| **KPI Cards** | Bloomberg-style cards displaying Portfolio Value, TWR, P/L, and MTD return |
| **AI Brief** | LLM-generated morning summary analyzing portfolio status and key metrics |
| **Quick Stats** | Inception date, days held, benchmark comparison |

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   VALUE     │     TWR     │     P/L     │    MTD      │
│  $47,232    │   +12.4%    │  +$5,232    │   +2.1%     │
│  ▲ +$1.2k   │  Ann. CAGR  │  Since Inc  │  vs +1.8%   │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

---

### Performance Page

**Purpose**: Deep-dive into portfolio returns with benchmark comparison.

| Chart | Description |
|-------|-------------|
| **Cumulative Return** | Line chart showing portfolio growth vs. selected benchmarks |
| **Excess Return** | Alpha visualization (portfolio - benchmark) over time |
| **Horizon Returns Table** | Modified Dietz returns by horizon (1D through SI) |
| **Horizon P/L Table** | Dollar P/L by horizon for economic impact analysis |
| **Growth of Capital** | Portfolio value vs. cumulative cash invested by asset class |

---

<div style="page-break-before: always;"></div>

### Risk Intelligence Page

**Purpose**: Risk assessment, correlation analysis, and forward projections.

| Component | Description |
|-----------|-------------|
| **Risk vs. Return Scatter** | Plots annualized volatility (10Y) vs. TTM return by asset class |
| **Correlation Heatmap** | 90-day rolling correlation matrix for diversification analysis |
| **Drawdown Chart** | "Underwater" visualization showing peak-to-trough declines |
| **20-Year Projections** | Interactive CAGR projector with contribution assumptions |
| **Allocation Simulator** | Adjust target weights to see impact on portfolio risk profile |

```
Risk vs. Return Plot:
                    ▲ Expected Return (TTM)
                    │
           ● Gold   │      ● US Growth
                    │   ● US Large Cap
           ● Bonds  │
                    │
                    └──────────────────────▶ Volatility (10Y Ann.)
```

---

### Attribution Page

**Purpose**: Understand exactly where alpha was generated or lost.

| Component | Description |
|-----------|-------------|
| **Active Strategy Table** | Portfolio return vs. benchmarks with excess calculation |
| **Attribution Waterfall** | Periodic P/L broken down by asset class (click to drill down) |
| **SI Summary** | Lifetime attribution showing contribution by asset class |

The attribution engine supports three granularities:
- **Daily**: Precise day-by-day P/L attribution
- **Weekly**: Aggregated weekly view for trend analysis  
- **Monthly**: High-level strategic performance review

---

<div style="page-break-before: always;"></div>

### Trade Lab Page

**Purpose**: "What-If" simulator using Monte Carlo engines.

| Feature | Description |
|---------|-------------|
| **Trade Ticket** | Input hypothetical Buy/Sell/Swap transactions |
| **Monte Carlo Engine** | Historical bootstrapping simulation (not parametric GBM) |
| **Outcome Cones** | 10th-90th percentile probability bands |
| **Success Probability** | Likelihood of achieving target values over 10-year horizon |

```python
def run_monte_carlo_simulation(...):
    """
    Modes:
    1. Historical Bootstrapping (Preferred):
       - Constructs synthetic portfolio return history
       - Samples rolling 21-day (monthly) returns
       - Captures real-world fat tails and correlations
       
    2. Parametric GBM (Fallback):
       - Assumes log-normal distribution
       - Uses mean/variance/correlation inputs
    """
```

---

### Tax Authority Page

**Purpose**: Tax-aware portfolio management and harvesting optimization.

| Component | Description |
|-----------|-------------|
| **Liability Sunburst** | Visual breakdown of ST/LT gains and losses |
| **Tactical Radar** | Decision matrix for harvest vs. hold recommendations |
| **Cliff Watch** | Lots turning long-term within 30 days (HOLD alerts) |
| **Harvesting Radar** | Unrealized losses available for tax-loss harvesting |
| **Tax Simulator** | Model the tax impact of hypothetical sales (FIFO/LIFO/HIFO) |

<div style="page-break-before: always;"></div>

---

### Rebalancing Tool Page

**Purpose**: Target-based portfolio rebalancing with tax-optimized execution.

| Component | Description |
|-----------|-------------|
| **Deployment Parameters** | Input for cash injections and "Allow Sales" toggle for full rebalancing |
| **Rebalancing Schedule** | AG Grid showing recommended Buys/Sells, trade amounts, and pro-forma weights |
| **Weight Drift Analysis** | Bar chart comparing Current vs. Target vs. Pro-Forma allocations |
| **Tax Impact Summary** | Real-time estimation of realized P/L and tax liability from proposed trades |
| **Cliff Watch** | Tracking tool for when new purchases will transition to long-term tax status |

#### Rebalancing Engine Logic

The rebalancing engine implements a **dual-phase optimization** strategy:

1.  **Tax-Aware Liquidation**: If sales are enabled, the engine identifies overweight assets and calls `calculate_tax_optimized_sales()` to generate proceeds while prioritizing loss harvesting and avoiding short-term gains.
2.  **Waterfall Allocation**: Available cash (new injection + sale proceeds) is distributed to underweight assets using a multi-pass waterfall. Capital is first directed to assets with the highest drift relative to targets until they are satisfied or cash is exhausted.

---

### Allocations & Holdings Pages

**Purpose**: Current portfolio composition and weight analysis.

| Feature | Description |
|---------|-------------|
| **Asset Class Pie** | Current vs. Target allocation visualization |
| **Sector Sunburst** | Look-through sector analysis (ETFs decomposed to constituents) |
| **Holdings Table** | AG Grid with sorting, filtering, and export capabilities |
| **Exited Tickers Toggle** | Option to show/hide closed positions |

---

<div style="page-break-before: always;"></div>

### Custom Report Page

**Purpose**: Institutional-quality report generator with customizable sections and print-ready PDF output.

| Feature | Description |
|---------|-------------|
| **Section Composer** | Modular interface to select and reorder report components (Performance, Risk, Holdings, etc.) |
| **Print Preview** | "Light Mode" toggle to visualize how charts will appear on white paper for PDFs |
| **PDF Optimization** | Specialized CSS classes for clean page breaks and A4 formatting |
| **Ghost Footer** | Compliance disclaimers that appear only in the printed document |

**Available Report Modules**:
- **Morning Brief**: AI-generated market summary and portfolio commentary
- **Horizon Analysis**: Standardized trailing return periods
- **Tax Lot Explorer**: Detailed visualization of open tax lots
- **Flows Summary**: Inflow/Outflow tracking for selected periods
- **Performance Deep Dive**: Detailed risk/return analytics

---

<div style="page-break-before: always;"></div>

## Key Technical Innovations

### 1. Flow-Based PV Reconstruction

The portfolio value series is built from first principles using raw cashflows:

```python
def build_portfolio_value_series_from_flows(holdings, prices, cashflows_path):
    """
    Algorithm:
    1. Start from zero positions and zero cash
    2. Apply all rows in cashflows.csv chronologically
    3. For each trading day:
       a. Apply flows dated on that day (start-of-day)
       b. Snapshot PV = Cash + Σ(Shares × Price)
    
    Guarantees:
    - Final Cash matches input exactly
    - Final Shares match input exactly
    - Full audit trail from first dollar to current state
    """
```

This approach ensures **perfect reconciliation** with the holdings file and enables forensic auditing.

### 2. Dynamic Sector Attribution

For ETFs, the system performs **look-through analysis**:

```python
# Example: VTI (Vanguard Total Stock Market)
fetch_etf_sectors("VTI") →
{
    "Technology": 28.5,
    "Healthcare": 13.2,
    "Financial Services": 12.8,
    "Consumer Cyclical": 10.5,
    ...
}
```

These weights are then aggregated at the portfolio level to show true sector exposure.

<div style="page-break-before: always;"></div>

### 3. Forensic Audit Suite

The `Validation/forensic_audit/` directory contains automated tests that verify mathematical accuracy:

| Audit Module | Coverage |
|--------------|----------|
| `audit_01_math_core.py` | TWR chain-linking, Modified Dietz formula |
| `audit_02_horizon_gating.py` | Horizon boundary conditions (MTD, YTD, SI) |
| `audit_03_pl_attribution.py` | P/L = MV_end - MV_start - Flows |
| `audit_04_risk_intelligence.py` | Volatility, correlation, Sharpe calculations |
| `audit_05_consistency_matrix.py` | Cross-layer data consistency |
| `audit_06_gips_scorecard.py` | GIPS compliance checklist |
| `audit_07_stress_test.py` | Edge cases (empty data, negative values) |

Run the full audit suite:
```bash
python Validation/forensic_audit/run_audit.py
```

### 4. Time Machine Architecture

The Time Machine is implemented via a **clipped data pipeline**:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Raw Cashflows  │ ──▶ │   Date Filter   │ ──▶ │ Temp Clipped    │
│  (Full History) │     │  (end_date ≤ X) │     │ Cashflows File  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│ Reconstructed   │ ◀── │  PV Builder     │ ◀────────────┘
│    Holdings     │     │  (from flows)   │
└─────────────────┘     └─────────────────┘
```

This ensures that `build_portfolio_value_series_from_flows()` always sees consistent data.

---

<div style="page-break-before: always;"></div>

## Visual Flowcharts

### Data Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│                       DATA SOURCES                          │
│  [CSV Files]       [yfinance API]        [FMP API]          │
└──────────┬──────────────────┬────────────────────┬──────────┘
           │                  │                    │
           ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                           │
│  [load_holdings]   [fetch_prices]    [fetch_sectors]        │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐      ┌────────────────────────┐
│        ENGINE LAYER         │      │       MATH LAYER       │
│ [Time Machine] -> [Horizon] │◀────▶│ [TWR] [Modified Dietz] │
└──────────┬──────────────────┘      │    [Annualization]     │
           │                         └────────────────────────┘
           ▼
┌─────────────────────────────────────────────────────────────┐
│                      WRAPPER LAYER                          │
│             [Data Cache] ───▶ [Figure Generators]           │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                         UI LAYER                            │
│           [App.py] ───▶ [Pages] ───▶ [Components]           │
└─────────────────────────────────────────────────────────────┘
```

<div style="page-break-before: always;"></div>

### Time Machine Reconstruction Process

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                           │
│              [Date Picker: 2024-06-15]                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA CLIPPING                          │
│     [Clip Transactions]  [Clip Cashflows]  [Clip Divs]      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   STATE RECONSTRUCTION                      │
│        [Sum Shares by Ticker]    [Calc Cash Balance]        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     PV SERIES BUILD                         │
│   [Write Temp Files] ──▶ [build_portfolio_value_series]     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        OUTPUT                               │
│      [TWR DF]        [Sec Table]         [PV Series]        │
└─────────────────────────────────────────────────────────────┘
```

---

<div style="page-break-before: always;"></div>

## Deployment & Usage

### Running the Dashboard

```bash
# Start the Dash server
python app.py

# Access at: http://127.0.0.1:8050/
```

### Generating Reports

```bash
# Generate DOCX/PDF report (outputs to Output/ directory)
python main.py

# CLI console summary (prints tables to terminal)
python main.py console
```

### Running Validation Suite

```bash
# Full forensic audit
python Validation/forensic_audit/run_audit.py

# Individual audit modules
python Validation/forensic_audit/audit_01_math_core.py
```

### Configuration

Key parameters are centralized in `config.py`:

```python
# Portfolio Targets
TARGET_PORTFOLIO_VALUE = 50000.0
TARGET_MONTHLY_CONTRIBUTION = 400

# Tax Rates
TAX_RATE_ST = 0.35  # Short-Term Capital Gains
TAX_RATE_LT = 0.15  # Long-Term Capital Gains

# Risk Parameters
RISK_FREE_RATE = 0.04  # For Sharpe/Sortino ratios
```
<div style="page-break-before: always;"></div>

### Environment Variables

```bash
# Required for FMP API access (ETF sector data)
export FMP_API_KEY="your_api_key_here"

# Or use .env file (loaded via python-dotenv)
echo "FMP_API_KEY=your_api_key_here" > .env
```

---

## Code Quality & Standards

### Clean Code Principles Applied

| Principle | Implementation |
|-----------|----------------|
| **Single Responsibility** | Each layer has one job (Data/Math/Engine/Wrapper/UI) |
| **Don't Repeat Yourself** | Shared functions in `financial_math.py` and `report_formatting.py` |
| **Dependency Inversion** | UI depends on abstractions (wrappers), not concrete engine details |
| **Explicit over Implicit** | All horizon logic gated with clear conditions |
| **Fail Fast** | Reconciliation errors raise immediately with actionable messages |

### Testing Philosophy

The Forensic Audit Suite follows a **trust-but-verify** approach:

1. **Synthetic Data Tests**: Controlled inputs with known outputs
2. **Invariant Checks**: Mathematical identities that must always hold
3. **GIPS Compliance**: Industry-standard performance measurement rules
4. **Edge Cases**: Empty data, negative values, single-day horizons

### Documentation Standards

- All public functions include docstrings with parameter/return descriptions
- Complex algorithms include inline comments explaining the math
- The `help_index.py` page provides user-facing documentation
- This architecture document serves as the developer reference

---

<div style="page-break-before: always;"></div>

## Summary

DELVEX Portfolio Analytics demonstrates enterprise-grade software engineering applied to quantitative finance:

- **Rigorous Architecture**: 5-layer separation ensures maintainability
- **Mathematical Precision**: GIPS-compliant calculations with audit trails  
- **User Experience**: Dark-mode Bloomberg-style interface with interactive charts
- **Extensibility**: Clean abstractions allow new features without refactoring core logic
- **Quality Assurance**: Automated forensic audits verify mathematical correctness

This codebase represents the intersection of **software craftsmanship** and **financial engineering**, suitable for production deployment in portfolio management workflows.

---

*Document Version: 1.0 | Last Updated: January 2026*
