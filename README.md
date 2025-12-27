
## Interactive Dashboard

This project includes a production-grade Dash analytics dashboard that replicates the report's metrics and charts in an interactive web application.

### Dashboard Features
-   **Multi-Page Interface**: Overview, Performance, Allocations, Attribution, Flows, Holdings, Risk & Projections.
-   **Interactive Charts**: Zoom, pan, and hover functionality powered by Plotly.
-   **Global Controls**: Dark mode toggle, Date range slider, Benchmark selection.
-   **Data Management**: Drag-and-drop upload for Holdings and Cashflows CSVs.

### Running the Dashboard

1.  Ensure you have the required libraries (including `dash` and `dash-bootstrap-components`):
    ```bash
    pip install dash dash-bootstrap-components pandas numpy yfinance plotly
    ```

2.  Run the application:
    ```bash
    python app.py
    ```

3.  Open your browser to `http://127.0.0.1:8050`.
