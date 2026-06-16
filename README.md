# 📈 Stock Price Forecasting using ARIMA

An end-to-end time series forecasting project that analyzes historical stock prices, visualizes trends and moving averages, builds ARIMA models, forecasts future stock prices, and evaluates predictions through an interactive Streamlit dashboard.

---

## 📌 Overview

Stock price forecasting is an important application of time series analysis that helps investors and analysts understand market trends and estimate future price movements.

This project uses historical stock data to:

- Analyze stock trends and patterns
- Calculate moving averages
- Test for stationarity
- Train and tune ARIMA models
- Forecast future stock prices
- Compare predictions with actual values
- Generate interactive visualizations and reports

---

## ✨ Features

### Dataset Management
- Upload stock datasets in CSV format
- Automatic date parsing
- Detect and handle missing values
- Sort data chronologically
- Generate descriptive statistics

### Exploratory Data Analysis (EDA)
- Closing price trend analysis
- Daily return analysis
- Rolling statistics
- Price distribution analysis
- Correlation analysis
- Volatility analysis

### Technical Analysis
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Rolling Mean
- Rolling Standard Deviation

### Time Series Forecasting
- Stationarity testing using ADF Test
- Automatic differencing
- Manual ARIMA configuration
- Auto ARIMA model selection
- Multi-day future forecasting

### Model Evaluation
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

### Interactive Dashboard
- Historical stock charts
- Forecast comparison charts
- Confidence intervals
- Residual analysis
- Downloadable reports

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- Statsmodels
- pmdarima
- Scikit-learn
- OpenPyXL
- ReportLab
- yfinance

---

## 🔄 Project Workflow

Historical Stock Data  
↓  
Data Cleaning and Preprocessing  
↓  
Date Indexing and Time Series Preparation  
↓  
Exploratory Data Analysis  
↓  
Moving Average Analysis  
↓  
Stationarity Testing  
↓  
ARIMA Model Training  
↓  
Forecast Generation  
↓  
Model Evaluation  
↓  
Interactive Visualizations  
↓  
Report Generation

---

## 📂 Dataset

Expected dataset format:

| Date | Close |
|------|--------|
| 2024-01-01 | 185.64 |
| 2024-01-02 | 187.21 |
| 2024-01-03 | 184.98 |

Optional columns:

- Open
- High
- Low
- Adj Close
- Volume

Minimum required columns:

- Date
- Close

---

## 📊 Exploratory Data Analysis

The dashboard provides:

- Dataset summary
- Missing value analysis
- Closing price trend
- Daily returns
- Price distribution
- Volatility analysis
- Rolling statistics
- Correlation heatmap

---

## 📈 Moving Average Analysis

The project calculates:

### Simple Moving Averages
- SMA 20
- SMA 50
- SMA 100
- SMA 200

### Exponential Moving Averages
- EMA 20
- EMA 50

These indicators help identify:

- Market trends
- Support and resistance levels
- Potential buy and sell signals

---

## 📉 Stationarity Testing

The application performs:

### Augmented Dickey-Fuller (ADF) Test

Displays:

- Test Statistic
- p-value
- Critical Values
- Stationarity conclusion

If the series is non-stationary, differencing is applied automatically.

---

## 🤖 ARIMA Model

The forecasting model uses:

### ARIMA(p, d, q)

Where:

- p = Autoregressive order
- d = Differencing order
- q = Moving Average order

Supports:

- Manual parameter selection
- Auto ARIMA optimization using AIC and BIC scores

---

## 🔮 Forecasting Features

Generate forecasts for:

- Next 7 Days
- Next 15 Days
- Next 30 Days
- Custom forecast periods

Display:

- Forecasted prices
- Confidence intervals
- Forecast plots
- Historical comparisons

---

## 📏 Model Evaluation Metrics

The forecasting models are evaluated using:

- MAE
- MSE
- RMSE
- MAPE
- R² Score

Additional analysis:

- Residual plots
- Forecast error distribution
- Actual vs Predicted comparison

---

## 📊 Visualizations

Interactive charts include:

- Historical Closing Price Trend
- Moving Average Charts
- Forecast vs Actual Comparison
- Future Forecast Visualization
- Confidence Interval Charts
- Residual Analysis
- Error Distribution Plots

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/stock-price-forecasting-arima.git
cd stock-price-forecasting-arima
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 📁 Project Structure

```text
stock-price-forecasting-arima/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
├── reports/
├── assets/
├── screenshots/
├── notebooks/
├── saved_models/
├── src/
│   ├── data_handler.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── stationarity.py
│   ├── moving_average.py
│   ├── arima_model.py
│   ├── forecasting.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── report_generator.py
│   └── utils.py
└── LICENSE
```

---

## 📈 Sample Forecast

Example:

| Date | Forecast Price |
|------|----------------|
| 2026-07-01 | ₹1,520 |
| 2026-07-02 | ₹1,526 |
| 2026-07-03 | ₹1,531 |

---

## 📄 Results

- Successfully analyzed historical stock trends.
- Generated moving average indicators.
- Built and optimized ARIMA forecasting models.
- Predicted future stock prices with confidence intervals.
- Evaluated model performance using multiple error metrics.
- Delivered interactive dashboards and downloadable reports.

---

## 🔮 Future Improvements

- Add LSTM and Prophet forecasting models
- Integrate real-time stock market APIs
- Portfolio optimization module
- Stock recommendation engine
- Sentiment analysis using financial news
- Multi-stock comparison dashboard
- Cloud deployment with Docker and CI/CD

---

## 📸 Screenshots

Add screenshots of:

- Dashboard Home Page
- Stock Trend Analysis
- Moving Average Charts
- ADF Test Results
- Forecasting Dashboard
- Actual vs Predicted Comparison
- Future Forecast Visualization

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Akhil Parmar**  
Computer Engineering Student | Data Science and Machine Learning Enthusiast

Feel free to contribute, raise issues, and share feedback.
