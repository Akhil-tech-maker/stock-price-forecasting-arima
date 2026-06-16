# AeroCast | Stock Price Forecasting Dashboard

AeroCast is a production-ready, interactive financial analytics and time series forecasting dashboard built using Python, Streamlit, and statsmodels. It supports historical stock price retrieval from Yahoo Finance, dataset uploads via CSV, descriptive exploratory data analysis (EDA), moving average technical indicator overlays, Augmented Dickey-Fuller (ADF) stationarity testing, and ARIMA/Auto-ARIMA modeling.

## Project Architecture

The project is structured modularly:
```text
stock_price_forecasting/
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── src/
    ├── __init__.py             # Package initializer
    ├── data_handler.py         # Fetch yfinance prices and process CSV files
    ├── preprocessing.py        # Outliers management, missing values treatment, resampling
    ├── eda.py                  # Statistical metrics, rolling volatility, daily returns
    ├── moving_average.py       # SMA/EMA indicators and Golden/Death cross detection
    ├── stationarity.py         # ADF unit root test and auto-differencing logic
    ├── arima_model.py          # Train-test split, manual and Auto-ARIMA fitting
    ├── forecasting.py          # Out-of-sample holdout and out-of-sample future predictions
    ├── evaluation.py           # Regression metrics (RMSE, MAE, MAPE, R2) & residuals stats
    ├── visualization.py        # Plotly charts and Matplotlib residual diagnostics subplots
    ├── report_generator.py     # PDF compilations, Excel multi-sheets, and CSV exporters
    └── utils.py                # Currency formatting and ticker string standardizer
```

## Features

1. **Flexible Data Sources:** Download live stock statistics using Yahoo Finance tickers (e.g. `AAPL`, `MSFT`, `INFY`) or upload custom CSV datasets.
2. **Interactive Exploratory Analysis (EDA):** Instantly evaluate data dimensions, null counts, descriptive statistics, and plot interactive charts (closing prices, trading volumes, daily percentage returns, distributions, and correlation matrices).
3. **Technical Trend Moving Averages:** Calculate SMA 20/50/100/200, EMA 20/50, and automatically detect Golden Cross / Death Cross and EMA crossovers.
4. **Statistical Unit Root Testing:** Run the Augmented Dickey-Fuller (ADF) test, examine critical values, and automatically difference non-stationary series.
5. **Robust ARIMA & Auto-ARIMA Fitting:** Customize $(p, d, q)$ parameters using sliders or run stepwise searches based on AIC/BIC.
6. **Chronological Hold-out Validation:** Configure train/test splits (70/30, 80/20, 90/10) to validate model performance on unseen test data before forecasting futures.
7. **Comprehensive Model Evaluation:** Compute RMSE, MAE, MAPE, and $R^2$ scores. Perform diagnostic checks on residuals (time series plots, KDE normal distribution fits, and Autocorrelation Function (ACF) plots).
8. **Out-of-sample Future Forecasting:** Forecast closing prices up to 30, 60, or custom days forward with shaded 95% confidence intervals.
9. **Export Capabilities:** Download forecast predictions as CSV, multi-sheet Excel reports, or compilation PDF reports complete with statistics, charts, and automated business insights.

## Library Stack
* Python 3.11+
* Streamlit
* Pandas
* NumPy
* Plotly
* Matplotlib
* Seaborn
* Statsmodels
* pmdarima (Auto-ARIMA search)
* Scikit-Learn
* Scipy
* Openpyxl (Excel exports)
* ReportLab (PDF compilation)
* yfinance (Live pricing API)

## Running the Application

1. **Install Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit Dashboard:**
   Run the following command in your terminal from the project's root folder:
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser:**
   The terminal will print local and network URLs (usually `http://localhost:8501`). Open this address in your web browser to access the dashboard.
