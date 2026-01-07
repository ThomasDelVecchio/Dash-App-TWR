# Portfolio Analytics Dashboard

This is a comprehensive portfolio analytics dashboard built with Plotly Dash. It provides in-depth analysis of your investment portfolio, including performance, risk, and attribution analysis. The application features an interactive web interface, and also provides command-line tools for generating reports.

## Features

*   **Interactive Dashboard:** A multi-page web application for exploring your portfolio data.
*   **Performance Analysis:** Track your portfolio's performance with metrics like Time-Weighted Return (TWR) and Modified Dietz.
*   **Risk Management:** Analyze your portfolio's risk profile with various risk metrics and visualizations.
*   **Attribution Analysis:** Understand the sources of your portfolio's returns.
*   **Data-Driven Insights:** Get AI-powered summaries and insights into your portfolio's performance.
*   **Report Generation:** Generate detailed portfolio reports in DOCX and PDF formats.
*   **Google Drive Integration:** Seamlessly export generated Word reports and data directly to a specific Google Drive folder.

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
The app comes pre-loaded with `sample_data/` so you can explore the dashboard immediately.

To import your own portfolio:
1. Delete the files inside the `sample_data/` folder.
2. Place your own **`transactions.csv`** and **`cashflows.csv`** in the root directory.

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

### Command-Line Reporting

You can also generate reports from the command line:

*   **Generate DOCX/PDF report:**
    ```bash
    python main.py
    ```
    This will generate a `TWR_MD_Report.docx` and `TWR_MD_Report.pdf` in the `Output/` directory.

*   **Run console report:**
    ```bash
    python main.py console
    ```
    This will print a summary of your portfolio's performance to the console.

## Pages

The dashboard consists of the following pages:

*   **/ (Overview):** A high-level overview of your portfolio, including KPIs, a portfolio value chart, and an AI-generated "Morning Brief".
*   **/performance:** Detailed performance analysis, including returns, benchmarks, and historical performance.
*   **/allocations:** A breakdown of your portfolio's asset allocation by asset class, sector, and individual holdings.
*   **/attribution:** Attribution analysis to understand the drivers of your portfolio's returns.
*   **/flows:** A summary of your portfolio's cash flows.
*   **/holdings:** A detailed view of your current holdings.
*   **/risk:** Risk analysis, including volatility, drawdowns, and other risk metrics.
*   **/trade-lab:** A laboratory for simulating trades and analyzing their potential impact on your portfolio.
*   **/taxes:** Tax analysis, including tax-lot accounting and simulated tax-loss harvesting.
*   **/settings:** Application settings, including theme and other preferences.
*   **/help:** A help index with information about the application and its features.

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
*   docx2pdf
*   matplotlib

## ⚠️ Disclaimer & Data Usage
**This software is for educational and research purposes only.**
* **Not Financial Advice:** The author assumes no responsibility for any financial losses incurred from using this tool.
* **Data Usage:** The default `yfinance` connector is for personal, non-commercial use. If you use this software for business (e.g., client reporting), you must use a commercial data provider like FMP.
