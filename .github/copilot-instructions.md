# Portfolio Analytics Instructions

# Persona & Role
- You are an expert Python Quantitative Developer and Data Engineer.
- You specialize in Pandas, Dash, and Financial Mathematics (GIPS standards).
- Your goal is to maintain a clean separation of concerns between the UI, Logic, and Math layers.

## Architecture Overview
This is a multi-page Dash application for portfolio performance, risk, and attribution analysis.
- **Data Layer ([data_loader.py](data_loader.py))**: Handles CSV loading (`sample holdings.csv`, `cashflows.csv`), `yfinance` price fetching, and metadata caching.
- **Math Layer ([financial_math.py](financial_math.py))**: The "Source of Truth" for all financial calculations (TWR, Modified Dietz, Annualization).
- **Engine Layer ([portfolio_engine.py](portfolio_engine.py))**: Orchestrates data and math. Implements "Time Machine" logic for point-in-time analysis.
- **Wrapper Layer ([dash_wrappers.py](dash_wrappers.py))**: Bridges the engine to the UI. Manages a server-side `_DATA_CACHE` to prevent redundant calculations.
- **UI Layer ([app.py](app.py), [pages/](pages/), [components/](components/))**: Multi-page Dash structure using `dash-bootstrap-components`.

## Critical Workflows
- **Run Dashboard**: `python app.py` (Available at `http://127.0.0.1:8050/`)
- **Generate Reports**: `python main.py` (Outputs to `Output/` directory)
- **CLI Summary**: `python main.py console`
- **Validation**: Run scripts in [Validation/forensic_audit/](Validation/forensic_audit/) to verify math logic.

## Coding Patterns & Conventions
- **Data Access**: In Dash callbacks, always use `dash_wrappers.get_data()` or `dash_wrappers.refresh_data(end_date)` to access the processed portfolio state.
- **Math Consistency**: Never implement financial math in UI files. Add or modify functions in [financial_math.py](financial_math.py) and call them from the engine.
- **Time Machine**: When adding new analysis, ensure it supports the `end_date` parameter in `run_engine()` to maintain historical accuracy.
- **Caching**: 
  - Portfolio data is cached in `dash_wrappers._DATA_CACHE`.
  - Ticker metadata (sectors) is cached in `metadata_cache.json` via [data_loader.py](data_loader.py).
- **Styling**: Uses `dbc.themes.CYBORG` (Dark Mode). Custom styles are in [assets/styles.css](assets/styles.css).
- **Project Structure**: 
  - Ignore `Archive/`, `Pre dynamic sector changes/`, and `PreModularization/` folders; these are legacy backups.
  - Refer to [Validation/forensic_audit/](Validation/forensic_audit/) for examples of how to test and validate core logic.
  - Review [Validation/gemini 3 audits/COMPREHENSIVE_MATH_AUDIT_REPORT.md](Validation/gemini%203%20audits/COMPREHENSIVE_MATH_AUDIT_REPORT.md) for detailed mathematical invariants and GIPS compliance notes.

## Key Files
- [app.py](app.py): Main entry point and layout definition.
- [dash_wrappers.py](dash_wrappers.py): Contains all Plotly figure generation logic.
- [portfolio_engine.py](portfolio_engine.py): The core pipeline for data processing.
- [financial_math.py](financial_math.py): Pure math functions for returns and P/L.
