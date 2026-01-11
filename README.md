# Portfolio Analytics Dashboard

This is a comprehensive portfolio analytics dashboard built with Plotly Dash. It provides in-depth analysis of your investment portfolio, including performance, risk, and attribution analysis. The application features an interactive web interface.

## Features

*   **Interactive Dashboard:** A multi-page web application for exploring your portfolio data.
*   **Performance Analysis:** Track your portfolio's performance with metrics like Time-Weighted Return (TWR) and Modified Dietz (GIPS-compliant).
*   **Risk Intelligence:** Analyze risk with volatility scatters, correlation heatmaps, and Monte Carlo simulations (historical bootstrapping).
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

### Loading Your Own Data

The app comes pre-loaded with sample files in the `sample_data/` folder: **`sample holdings.csv`** and **`cashflows.csv`**. *IMPORTANT* Move them into the root folder with all other py files before running 

To import your own portfolio:
1. Replace the **`sample holdings.csv`** and **`cashflows.csv`** files in the root directory with your own data (keeping the same filenames).
2. Ensure your CSV files follow the column structure of the samples.

### API Keys (Optional)

The application can use the Financial Modeling Prep (FMP) API to fetch detailed ETF sector weightings. While not required to run the app, this data enhances the accuracy of your asset allocation analysis.

To enable this feature:
1.  Open the `config.py` file.
2.  Find the line `FMP_API_KEY = "demo"`.
3.  Replace `"demo"` with your actual FMP API key, like this:
    ```python
    FMP_API_KEY = "YOUR_SECRET_API_KEY"
    ```

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
*   **/risk:** Risk analysis, including volatility, drawdowns, and correlation matrices.
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
*   yfinance
*   requests
*   plotly
*   python-dotenv
*   python-docx
*   matplotlib

## ⚠️ Disclaimer & Data Usage
**This software is for educational and research purposes only.**
* **Not Financial Advice:** The author assumes no responsibility for any financial losses incurred from using this tool.
* **Data Usage:** The default `yfinance` connector is for personal, non-commercial use. If you use this software for business (e.g., client reporting), you must use a commercial data provider like FMP.
