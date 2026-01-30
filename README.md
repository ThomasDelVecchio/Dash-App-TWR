# Portfolio Analytics Dashboard

This is a comprehensive portfolio analytics dashboard built with Plotly Dash. It provides in-depth analysis of your investment portfolio, including performance, risk, and attribution analysis. The application features an interactive web interface.

## Features

*   **Interactive Dashboard:** A multi-page web application for exploring your portfolio data.
*   **E\*TRADE Integration:** Automatic sync of transactions and holdings from your E\*TRADE brokerage account via OAuth 1.0a API.
*   **Trade Execution (E\*TRADE):** Preview and place equity orders from the Trade page, with sandbox/live environment safeguards.
*   **Performance Analysis:** Track your portfolio's performance with metrics like Time-Weighted Return (TWR) and Modified Dietz (GIPS-compliant).
*   **Risk Intelligence:** Analyze risk with volatility scatters, correlation heatmaps, and Monte Carlo simulations (historical bootstrapping).
*   **Strategy Backtesting:** Compare quarterly rebalanced strategy benchmarks with growth, drawdowns, and risk/return scorecards.
*   **Attribution Analysis:** Understand the sources of your portfolio's returns with Brinson-Fachler and Frongello linking models.
*   **Tax Optimization:** Visualize tax lots, identify harvesting opportunities, and simulate trade tax impacts (FIFO/LIFO/HIFO).
*   **Rebalancing:** Drill-down rebalancing tool with tax-aware trade generation and drift analysis.
*   **AI Assistant:** Integrated chatbot for natural language portfolio queries and ad-hoc analysis.
*   **Custom Reporting:** Drag-and-drop report builder for creating print-ready PDFs with specific modules.
*   **Data-Driven Insights:** Get AI-powered "Morning Brief" summaries and insights into your portfolio's performance.
*   **Google Drive Integration:** Seamlessly export generated reports and data directly to a specific Google Drive folder.

## Installation

1.  **Unzip the project files** to a directory of your choice.

2.  **Install the requirements:**
    Open your terminal or command prompt in this folder and run:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you prefer virtual environments, feel free to set one up, but it is not required.)*

## Configuration
#### System Settings
You can configure system behavior (tax rates, risk targets, API keys) using one of two methods:

**Method 1: Environment Variables (Recommended)**
Create a file named `.env` in the root directory (use `.env.example` as a template). Variables set here will override `config.py`.
```ini
FMP_API_KEY=your_actual_api_key
RISK_FREE_RATE=0.045
TARGET_PORTFOLIO_VALUE=100000
```

**Method 2: config.py (Direct Edit)**
You can directly edit the `config.py` file in the root directory.
```python
# config.py
FMP_API_KEY = "demo"  # Replace with your key
RISK_FREE_RATE = 0.04
```

*Note: The application prioritizes `.env` values if they exist.*

### E\*TRADE Integration (Optional)

You can automatically sync your transactions and holdings directly from your E\*TRADE brokerage account. This eliminates manual CSV maintenance and ensures your portfolio data is always up-to-date.

**Prerequisites:**
1.  **Create an E\*TRADE Developer Account:** Visit the [E\*TRADE Developer Portal](https://developer.etrade.com/).
2.  **Register an Application:** Create a new app and note your **Consumer Key** and **Consumer Secret**.
3.  **Enable Production Access:** Request production API access (sandbox is for testing only).

**Configuration:**
Add your E\*TRADE credentials to the `.env` file:
```ini
ETRADE_CONSUMER_KEY=your_consumer_key_here
ETRADE_CONSUMER_SECRET=your_consumer_secret_here
ETRADE_ACCOUNT_ID=your_account_number
ETRADE_SANDBOX=false
ETRADE_AUTO_SYNC=true
```

**First-Time Authentication:**
Run the authentication script to authorize the application:
```bash
python etrade_auth.py
```
This will open your browser to E\*TRADE's login page. After authorization, a verification code is displayed. Enter this code in the terminal to complete setup. Tokens are cached in `etrade_token.json` and automatically renewed.

**Automatic Sync:**
When `ETRADE_AUTO_SYNC=true`, the app automatically syncs on startup:
- New transactions are appended to `cashflows.csv`
- Holdings are updated in `sample holdings.csv`
- Sync status is displayed in the sidebar

**Trade Execution (Optional):**
The Trade page supports order preview and placement via the E\*TRADE API. It uses the same OAuth credentials and respects your environment:
- `ETRADE_SANDBOX=true` to test orders in sandbox
- `ETRADE_SANDBOX=false` for live trading (real money)

Recommended flow:
1. Preview an order to validate costs and warnings
2. Confirm and place the order from the Trade page

Order confirmations are stored locally in `order_history.json`.

**External Holdings (Stock Plans, Other Brokers):**
For positions not accessible via the E\*TRADE API (e.g., employee stock plans), create a `holdings_external.csv` file:
```csv
ticker,shares,asset_class,target_pct
VTI, 5, Large Cap, 50
```
These positions are automatically merged during sync.

**Security Notes:**
- Consumer key/secret are stored only in `.env` (never committed to git)
- OAuth tokens expire at midnight ET and are auto-renewed
- All API calls use verified HTTPS connections

### Loading Your Own Data

The app comes pre-loaded with sample files in the `sample_data/` folder: **`sample holdings.csv`** and **`cashflows.csv`**. *IMPORTANT* Move them into the root folder with all other py files before running 

To import your own portfolio:
1. Replace the **`sample holdings.csv`** and **`cashflows.csv`** files in the root directory with your own data (keeping the same filenames).
2. Ensure your CSV files follow the column structure of the samples.

### Google Drive Integration (Optional)

You can export generated reports and data directly to Google Drive. This requires setting up a Google Cloud Project and authenticating your account.

**Prerequisites:**
1.  **Create a Google Cloud Project:** Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  **Enable Google Drive API:** Search for "Google Drive API" and click "Enable".
3.  **Configure OAuth Consent Screen:** Set the user type to "External" and add yourself as a test user. Add the `.../auth/drive.file` scope.
4.  **Create Credentials:**
    *   Go to **Credentials** -> **Create Credentials** -> **OAuth client ID**.
    *   Select **Desktop app** for the application type.
    *   Click **Download JSON** for the created client ID.
5.  **Setup Files:**
    *   Rename the downloaded file to `client_secret.json`.
    *   Place `client_secret.json` in the root directory of this project.

**Authentication:**
Run the following command in your terminal:
```bash
python authorize.py
```
This will open your web browser. Log in with your Google account and grant the requested permissions. A `token.json` file will be created in your root directory, which the application uses for subsequent exports.

**Target Folder (Optional):**
By default, files are uploaded to a specific folder defined in `dash_wrappers.py`. To change the destination folder:
1.  Open `dash_wrappers.py`.
2.  Find the variable `PARENT_FOLDER_ID`.
3.  Replace the value with the ID of your desired Google Drive folder (found in the folder's URL).

### Price Data Sources

The application supports a **hybrid price data system** that can combine multiple data sources:

| Mode | Configuration | Description |
|------|---------------|-------------|
| **yfinance-only** (Default) | `FMP_PRICE_ENABLED=false` | Free 20-year history from Yahoo Finance. Suitable for personal use. |
| **Hybrid FMP+yfinance** | `FMP_PRICE_ENABLED=true` | FMP for recent 5 years + yfinance for extended history (up to 20 years). More reliable recent data. |

**To enable Hybrid Mode:**
```ini
# In .env file
FMP_API_KEY=your_fmp_api_key
FMP_PRICE_ENABLED=true
FMP_PRICE_LOOKBACK_YEARS=5
```

**Benefits of Hybrid Mode:**
- More reliable recent price data (fewer gaps, better corporate action handling)
- Automatic fallback to yfinance if FMP fails for a ticker
- UI badge on Overview/Performance pages shows which source was used

For comprehensive documentation, see `Validation/FMP_USAGE_GUIDE.md`.


## Usage

### Interactive Dashboard

To run the interactive Dash application, run the following command:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:8050/`.

## Pages

The dashboard consists of the following pages:

*   **/ (Overview):** A high-level overview of your portfolio, including KPIs, a portfolio value chart, and an AI-generated "Morning Brief".
*   **/performance:** Detailed performance analysis, including returns, benchmarks, and historical performance.
*   **/allocations:** A breakdown of your portfolio's asset allocation by asset class, sector, and individual holdings.
*   **/attribution:** Attribution analysis to understand the drivers of your portfolio's returns.
*   **/flows:** A summary of your portfolio's cash flows.
*   **/holdings:** A detailed view of your current holdings.
*   **/rebalancing:** Interactive rebalancing tool with tax-aware trade generation and drift analysis.
*   **/trade:** E\*TRADE order preview and execution hub with sandbox/live badges and confirmation flow.
*   **/risk:** Risk analysis, including volatility, drawdowns, and correlation matrices.
*   **/strategy-backtesting:** Strategy Arena for quarterly rebalanced backtests, drawdowns, and risk/return scorecards.
*   **/trade-lab:** A laboratory for simulating trades and analyzing their potential impact via Monte Carlo simulations.
*   **/taxes:** Tax analysis, including tax-lot accounting, cliffs, and simulated tax-loss harvesting.
*   **/custom-report:** A customizable report builder to select, reorder, and print specific portfolio modules (PDF-ready).
*   **/settings:** Application settings, including theme and other preferences.
*   **/help:** A help index with information about the application and its features.

### AI Assistant
The dashboard features a persistent, draggable AI chatbot powered by Google Gemini. It has context awareness of your portfolio data and can answer questions about performance, risk, and specific holdings. It can also run Python code snippets to perform ad-hoc analysis.

## Forensic Validation & Auditing

This application relies on a strict **5-Layer Separation of Concerns** (UI, Wrapper, Engine, Math, Data) to ensure data integrity. To guarantee mathematical accuracy and GIPS compliance, the project includes a rigorous **Forensic Audit Suite** located in `Validation/forensic_audit/`.

### Running the Full Audit
To run the complete validation suite (all tests), execute:

```bash
python Validation/forensic_audit/run_audit.py
```

This will run all audit modules and print a pass/fail summary to the console.

### Key Audit Modules
You can also run individual audit scripts to verify specific components:

*   **Math Core (`audit_01_math_core.py`):** Verifies the TWR chain-linking and Modified Dietz formulas against known manual calculations.
*   **Horizon Gating (`audit_02_horizon_gating.py`):** Ensures returns are only calculated when sufficient data and strict horizon rules are met (e.g., no annualized returns for < 1 year).
*   **P/L Attribution (`audit_03_pl_attribution.py`):** Validates that Portfolio P/L matches exactly with `MV_end - MV_start - NetFlows`.
*   **Risk Intelligence (`audit_04_risk_intelligence.py`):** verifies Sharpe ratios, volatility calculations, and drawdown logic.
*   **GIPS Scorecard (`audit_06_gips_scorecard.py`):** Checks for key GIPS compliance requirements (start-of-day flows, fair value, etc.).

For more details on the testing methodology, refer to `Validation/SYSTEM_ARCHITECTURE.md`.

## Dependencies

The following Python libraries are required to run the application:

*   dash
*   dash-bootstrap-components
*   dash-ag-grid
*   pandas
*   numpy
*   yfinance (for price data - default mode)
*   requests (for FMP API - optional hybrid mode)
*   plotly
*   python-dotenv
*   python-docx
*   matplotlib

## ⚠️ Disclaimer & Data Usage
**This software is for educational and research purposes only.**
* **Not Financial Advice:** The author assumes no responsibility for any financial losses incurred from using this tool.
* **Data Usage:** The default `yfinance` connector is for personal, non-commercial use. For business use (e.g., client reporting), enable **Hybrid Mode** with a commercial FMP API key (`FMP_PRICE_ENABLED=true`).
