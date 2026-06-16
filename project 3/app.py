import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# Import project modules
from src.data_handler import fetch_yfinance_data, load_csv_data
from src.preprocessing import clean_data, handle_outliers, prepare_time_frequency
from src.eda import get_dataset_info, get_summary_stats, calculate_returns_and_volatility, get_key_metrics
from src.moving_average import calculate_moving_averages, detect_crossover_signals
from src.stationarity import perform_adf_test, make_stationary
from src.arima_model import train_test_split_data, fit_manual_arima, fit_auto_arima
from src.forecasting import generate_test_forecast, generate_future_forecast
from src.evaluation import calculate_evaluation_metrics, get_residuals_metrics
from src.utils import clean_ticker, format_currency, get_percentage_change_desc
import src.visualization as vis
import src.report_generator as rep

# Set page layout and config
st.set_page_config(
    page_title="AeroCast | Stock Price Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B !important;
    }
    
    /* Force markdown text to be dark when background is light */
    [data-testid="stMarkdownContainer"] {
        color: #1E293B !important;
    }
    
    /* Ensure all headers inside main content are dark */
    [data-testid="stMarkdownContainer"] h1, 
    [data-testid="stMarkdownContainer"] h2, 
    [data-testid="stMarkdownContainer"] h3, 
    [data-testid="stMarkdownContainer"] h4, 
    [data-testid="stMarkdownContainer"] h5, 
    [data-testid="stMarkdownContainer"] h6 {
        color: #0F172A !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }
    section[data-testid="stSidebar"] label {
        color: #94A3B8 !important;
    }
    
    /* Premium Header */
    .app-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .app-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        color: white !important;
    }
    .app-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
        color: white !important;
    }
    
    /* Styled Cards */
    .kpi-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.5rem;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #059669;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    .kpi-sub-down {
        font-size: 0.8rem;
        color: #DC2626;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    /* Container sections */
    .section-container {
        background: white;
        padding: 1.75rem;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        color: #1E293B !important;
    }
    .section-container h3, 
    .section-container h4, 
    .section-container p, 
    .section-container li,
    .section-container ol,
    .section-container ul {
        color: #1E293B !important;
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
        transition: opacity 0.2s !important;
    }
    div.stButton > button:hover {
        opacity: 0.9 !important;
    }
    
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
if 'ticker' not in st.session_state:
    st.session_state.ticker = "AAPL"
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime.now() - timedelta(days=730) # 2 years default
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.now()
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'ma_df' not in st.session_state:
    st.session_state.ma_df = None
if 'crossover_df' not in st.session_state:
    st.session_state.crossover_df = None
if 'stationarity_info' not in st.session_state:
    st.session_state.stationarity_info = None
if 'd_order' not in st.session_state:
    st.session_state.d_order = 0
if 'adf_history' not in st.session_state:
    st.session_state.adf_history = None
if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None
if 'model_fit' not in st.session_state:
    st.session_state.model_fit = None
if 'arima_order' not in st.session_state:
    st.session_state.arima_order = (1, 1, 1)
if 'test_forecast_df' not in st.session_state:
    st.session_state.test_forecast_df = None
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = None
if 'future_forecast_df' not in st.session_state:
    st.session_state.future_forecast_df = None
if 'forecast_days' not in st.session_state:
    st.session_state.forecast_days = 30
if 'train_ratio' not in st.session_state:
    st.session_state.train_ratio = 0.8
if 'outlier_method' not in st.session_state:
    st.session_state.outlier_method = 'none'
if 'fill_method' not in st.session_state:
    st.session_state.fill_method = 'interpolate'
if 'auto_arima_results' not in st.session_state:
    st.session_state.auto_arima_results = None

# Helper to render KPI cards
def render_kpi(label, value, sub_text=None, is_down=False):
    sub_class = "kpi-sub-down" if is_down else "kpi-sub"
    sub_html = f'<div class="{sub_class}">{sub_text}</div>' if sub_text else ''
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 1.5rem;">
    <h2 style="color: #60A5FA; margin: 0; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.05em;">AeroCast</h2>
    <p style="color: #94A3B8; margin: 0; font-size: 0.85rem;">ARIMA Stock Forecasting</p>
</div>
<hr style="margin: 0 0 1rem 0;"/>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Home",
        "2. Dataset Upload",
        "3. Data Exploration",
        "4. Moving Average Analysis",
        "5. Stationarity Testing",
        "6. ARIMA Configuration",
        "7. Model Training",
        "8. Forecasting",
        "9. Model Evaluation",
        "10. Future Predictions",
        "11. Export Reports"
    ]
)

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

# Sidebar metadata panel if data is loaded
if st.session_state.df is not None:
    st.sidebar.subheader("Active Dataset")
    st.sidebar.text(f"Ticker/Source: {st.session_state.ticker}")
    st.sidebar.text(f"Observations: {len(st.session_state.df)} days")
    st.sidebar.text(f"Range: {st.session_state.df.index.min().strftime('%Y-%m-%d')} to {st.session_state.df.index.max().strftime('%Y-%m-%d')}")
    if st.session_state.model_fit is not None:
        st.sidebar.success(f"Model: ARIMA{st.session_state.arima_order}")
    else:
        st.sidebar.info("Model: Not Trained")
else:
    st.sidebar.warning("No dataset loaded.")


# ----------------- PAGE ROUTING -----------------

# PAGE 1: HOME
if page == "1. Home":
    st.markdown("""
<div class="app-header">
    <h1>AeroCast Stock Forecasting</h1>
    <p>Interactive time series analysis and forecasting platform utilizing Box-Jenkins (ARIMA) methodology.</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("""
<div class="section-container">
<h3>Welcome to AeroCast</h3>
<p>This web application is a production-ready forecasting tool designed for finance students, researchers, and retail analysts. 
It integrates advanced time series cleaning, statistical stationarity testing, moving average crossover systems, and ARIMA/Auto-ARIMA models to predict future price trends.</p>

<h4>Key Application Pipeline</h4>
<div style="background-color: #F1F5F9; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0;">
<code style="color: #1E293B; font-size: 0.95rem; line-height: 1.6;">
Historical Stock Data &rarr; Preprocessing &rarr; Stationarity Check &rarr; Auto/Manual ARIMA Selection &rarr; Train-Test Split &rarr; Out-of-sample Forecast Validation &rarr; Evaluation Metrics (RMSE, MAPE) &rarr; Out-of-sample Future Forecast (Confidence Bands) &rarr; Business Insights & Report Generation
</code>
</div>

<h4>How to Get Started:</h4>
<ol>
<li>Navigate to <b>2. Dataset Upload</b> in the sidebar.</li>
<li>Fetch historical closing price data using a Yahoo Finance ticker (e.g. <i>AAPL, TSLA, INFY, MSFT</i>) or upload your custom CSV file.</li>
<li>Clean the data, set up the moving averages, and test for stationarity.</li>
<li>Configure your model, split your data, train ARIMA, and inspect predictions and residuals.</li>
<li>Export the results to PDF or Excel sheets with auto-generated investment recommendations.</li>
</ol>
</div>
""", unsafe_allow_html=True)


# PAGE 2: DATASET UPLOAD
elif page == "2. Dataset Upload":
    st.markdown("""
    <div class="app-header">
        <h1>Dataset Management</h1>
        <p>Upload your own CSV records or pull live historical stock prices directly from Yahoo Finance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("Data Source Selection")
        source = st.radio("Choose Input Method:", ["Yahoo Finance Ticker Search", "Local CSV Upload"])
        
        if source == "Yahoo Finance Ticker Search":
            ticker_input = st.text_input("Enter Stock Ticker Symbol:", value="AAPL", max_chars=10, help="E.g., AAPL for Apple, MSFT for Microsoft, INFY for Infosys")
            
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                start_d = st.date_input("Start Date:", value=datetime.now() - timedelta(days=730))
            with d_col2:
                end_d = st.date_input("End Date:", value=datetime.now())
                
        else:
            uploaded_file = st.file_uploader("Choose a CSV File:", type=["csv"], help="Must contain a Date column and a Close column (or a price column)")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("Preprocessing Controls")
        
        fill_method = st.selectbox(
            "Handle Missing Values:",
            options=["interpolate", "ffill", "drop"],
            index=0,
            help="Interpolate calculates linear values between dates. Ffill forward fills values. Drop removes nulls."
        )
        
        outlier_method = st.selectbox(
            "Handle Outliers (IQR Method):",
            options=["none", "clip", "remove"],
            index=0,
            help="Clip caps prices to 1.5 * IQR. Remove deletes outlier rows. None leaves them unchanged."
        )
        
        resample_freq = st.selectbox(
            "Time Series Resampling Frequency:",
            options=["B (Business Days)", "D (Calendar Days)", "None"],
            index=0,
            help="ARIMA models require a regular frequency. Stock markets trade on Business Days (B)."
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Trigger load
    load_btn = st.button("Load & Preprocess Dataset")
    
    if load_btn:
        with st.spinner("Processing stock data..."):
            try:
                # 1. Fetch/Upload data
                if source == "Yahoo Finance Ticker Search":
                    ticker_clean = clean_ticker(ticker_input)
                    if not ticker_clean:
                        st.error("Please enter a valid ticker.")
                        st.stop()
                    df_raw = fetch_yfinance_data(ticker_clean, start_d.strftime('%Y-%m-%d'), end_d.strftime('%Y-%m-%d'))
                    st.session_state.ticker = ticker_clean
                else:
                    if uploaded_file is None:
                        st.error("Please upload a CSV file.")
                        st.stop()
                    df_raw = load_csv_data(uploaded_file)
                    st.session_state.ticker = uploaded_file.name.split('.')[0].upper()
                
                # Save raw in session
                st.session_state.df = df_raw
                
                # 2. Clean missing data
                cleaned = clean_data(df_raw, fill_method=fill_method)
                
                # 3. Handle outliers
                cleaned = handle_outliers(cleaned, column='Close', method=outlier_method)
                
                # 4. Set frequency
                if resample_freq == "B (Business Days)":
                    cleaned = prepare_time_frequency(cleaned, freq='B', fill_method=fill_method)
                elif resample_freq == "D (Calendar Days)":
                    cleaned = prepare_time_frequency(cleaned, freq='D', fill_method=fill_method)
                    
                st.session_state.df_cleaned = cleaned
                
                # 5. Compute moving averages
                ma_df = calculate_moving_averages(cleaned, column='Close')
                st.session_state.ma_df = ma_df
                st.session_state.crossover_df = detect_crossover_signals(ma_df)
                
                # 6. Compute Stationarity
                close_series = cleaned['Close']
                diff_series, d_order, adf_history = make_stationary(close_series, max_diff=2)
                st.session_state.stationarity_info = adf_history[0]['results']
                st.session_state.d_order = d_order
                st.session_state.adf_history = adf_history
                
                # Reset downstream states to prevent mismatch
                st.session_state.model_fit = None
                st.session_state.test_forecast_df = None
                st.session_state.future_forecast_df = None
                st.session_state.evaluation_metrics = None
                st.session_state.auto_arima_results = None
                
                st.success(f"Successfully loaded {len(cleaned)} trading sessions for {st.session_state.ticker}!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
    # Data preview
    if st.session_state.df_cleaned is not None:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader(f"Data Preview ({st.session_state.ticker})")
        
        col_preview1, col_preview2 = st.columns([1, 1.5])
        with col_preview1:
            st.markdown("##### First 5 Records:")
            st.dataframe(st.session_state.df_cleaned.head(5))
            st.markdown("##### Last 5 Records:")
            st.dataframe(st.session_state.df_cleaned.tail(5))
        with col_preview2:
            st.markdown("##### Dataset Description Summary:")
            st.dataframe(get_summary_stats(st.session_state.df_cleaned))
        st.markdown("</div>", unsafe_allow_html=True)


# PAGE 3: DATA EXPLORATION (EDA)
elif page == "3. Data Exploration":
    st.markdown("""
    <div class="app-header">
        <h1>Exploratory Data Analysis (EDA)</h1>
        <p>Analyze trends, distributions, statistical measures, correlation matrix, and returns volatility.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is None:
        st.warning("Please upload or download a dataset in the 'Dataset Upload' page first.")
        st.stop()
        
    df = st.session_state.df_cleaned
    df_analyzed = calculate_returns_and_volatility(df)
    metrics = get_key_metrics(df)
    
    # KPI Grid
    st.subheader(f"Descriptive Indicators for {st.session_state.ticker}")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi("Trading Days", f"{metrics['total_trading_days']} Days")
    with c2:
        render_kpi("Latest Closing Price", format_currency(metrics['latest_close']))
    with c3:
        render_kpi("Highest Price", format_currency(metrics['highest_price']), f"On {metrics['highest_date']}")
    with c4:
        render_kpi("Lowest Price", format_currency(metrics['lowest_price']), f"On {metrics['lowest_date']}")
    with c5:
        render_kpi("Average Closing Price", format_currency(metrics['average_close']))
        
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Time Series Plots")
    
    chart_type = st.selectbox("Select Interactive Visualization Plot:", [
        "Closing Price Trend",
        "Volume Trend",
        "Daily Returns (%)",
        "Price Distribution",
        "Rolling Statistics",
        "Correlation Heatmap"
    ])
    
    if chart_type == "Closing Price Trend":
        fig = vis.plot_close_price(df)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Volume Trend":
        fig = vis.plot_volume(df)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Daily Returns (%)":
        fig = vis.plot_returns(df_analyzed)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Price Distribution":
        fig = vis.plot_price_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Rolling Statistics":
        window = st.slider("Volatility Window (days):", min_value=5, max_value=100, value=20)
        fig = vis.plot_rolling_stats(df, window=window)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Correlation Heatmap":
        fig = vis.plot_correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Dataset information summary
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Data Structures Summary")
    info = get_dataset_info(df)
    
    meta_df = pd.DataFrame([
        {"Characteristic": "Dataset Dimensions (Shape)", "Value": str(info['shape'])},
        {"Characteristic": "Timeline Start Date", "Value": info['index_start']},
        {"Characteristic": "Timeline End Date", "Value": info['index_end']},
        {"Characteristic": "Date Index Frequency", "Value": str(df.index.freqstr)}
    ])
    
    col_meta1, col_meta2 = st.columns([1, 1])
    with col_meta1:
        st.markdown("##### Metadata Table")
        st.dataframe(meta_df, use_container_width=True)
    with col_meta2:
        st.markdown("##### Missing Values Count")
        missing_df = pd.DataFrame(list(info['missing_values'].items()), columns=["Column", "Missing Count"])
        st.dataframe(missing_df, use_container_width=True)
        
    st.markdown("</div>", unsafe_allow_html=True)


# PAGE 4: MOVING AVERAGE ANALYSIS
elif page == "4. Moving Average Analysis":
    st.markdown("""
    <div class="app-header">
        <h1>Moving Average Analysis</h1>
        <p>Simple and Exponential Moving Averages overlayed with automated trend crossovers and buy/sell signals.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is None:
        st.warning("Please upload or download a dataset in the 'Dataset Upload' page first.")
        st.stop()
        
    ma_df = st.session_state.ma_df
    crossover_df = st.session_state.crossover_df
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Moving Average Trend Chart")
    fig = vis.plot_moving_averages(ma_df, crossover_df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Detected Crossover Events (Buy/Sell Signals)")
    st.markdown("""
    * **Golden Cross:** Short-term SMA 50 crosses above long-term SMA 200 (classic Bullish indicator).
    * **Death Cross:** Short-term SMA 50 crosses below long-term SMA 200 (classic Bearish indicator).
    * **Short-term Crossover:** EMA 20 crosses EMA 50 (captures short-to-intermediate price momentum shifts).
    """)
    if not crossover_df.empty:
        st.dataframe(crossover_df, use_container_width=True)
    else:
        st.info("No crossover events detected in this stock index timeline (requires sufficient historical timeline to calculate SMA 200).")
    st.markdown("</div>", unsafe_allow_html=True)


# PAGE 5: STATIONARITY TESTING
elif page == "5. Stationarity Testing":
    st.markdown("""
    <div class="app-header">
        <h1>Stationarity Testing</h1>
        <p>Evaluate time series stationarity using the Augmented Dickey-Fuller (ADF) test, applying auto-differencing if needed.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is None:
        st.warning("Please upload or download a dataset in the 'Dataset Upload' page first.")
        st.stop()
        
    df = st.session_state.df_cleaned
    adf_history = st.session_state.adf_history
    d_order = st.session_state.d_order
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Augmented Dickey-Fuller (ADF) Test Results")
    
    # Render table of adf steps
    for i, step in enumerate(adf_history):
        res = step['results']
        d = step['d']
        
        st.markdown(f"#### Differencing Level $d = {d}$" + (" (Original Series)" if d == 0 else f" ({d}-order Differenced)"))
        
        if np.isnan(res['test_statistic']):
            st.error(res['conclusion'])
            continue
            
        c_p1, c_p2 = st.columns([1, 1.5])
        with c_p1:
            st.markdown(f"**Test Statistic:** `{res['test_statistic']:.4f}`")
            st.markdown(f"**p-value:** `{res['p_value']:.4f}`")
            st.markdown(f"**Stationary:** " + ("`Yes` ✅" if res['is_stationary'] else "`No` ❌"))
        with c_p2:
            st.markdown("**Critical Values Table:**")
            crit_df = pd.DataFrame(list(res['critical_values'].items()), columns=["Confidence Level", "Critical Value"])
            st.dataframe(crit_df)
            
        st.info(res['conclusion'])
        st.markdown("<hr style='border-color: #E2E8F0;' />", unsafe_allow_html=True)
        
    st.markdown(f"**Auto-Differencing Decision Summary:** The dataset requires a differencing order of **d = {d_order}** to achieve stationarity for ARIMA modeling.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualizing Original vs Differenced
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Stationarity Transformations Visualization")
    
    if d_order == 0:
        st.info("The original closing price is already stationary. No differencing plots needed.")
        fig = vis.plot_close_price(df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Create differencing comparison plot
        fig_diff = make_subplots(rows=2, cols=1, subplot_titles=("Original Series (Non-Stationary)", f"Differenced Series (Order d={d_order} - Stationary)"))
        fig_diff.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Original Close', line=dict(color='#3b82f6')), row=1, col=1)
        
        diff_y = df['Close']
        for _ in range(d_order):
            diff_y = diff_y.diff().dropna()
            
        fig_diff.add_trace(go.Scatter(x=diff_y.index, y=diff_y, name=f'{d_order}d Differenced', line=dict(color='#10b981')), row=2, col=1)
        
        fig_diff.update_layout(height=600, template='plotly_white', showlegend=False)
        st.plotly_chart(fig_diff, use_container_width=True)
        
    st.markdown("</div>", unsafe_allow_html=True)


# PAGE 6: ARIMA CONFIGURATION
elif page == "6. ARIMA Configuration":
    st.markdown("""
    <div class="app-header">
        <h1>ARIMA Model Tuning & Configuration</h1>
        <p>Configure manual (p, d, q) orders or execute the automated search algorithms (Auto ARIMA) based on AIC/BIC.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is None:
        st.warning("Please upload or download a dataset in the 'Dataset Upload' page first.")
        st.stop()
        
    df = st.session_state.df_cleaned
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("Manual Parameter Settings")
        st.markdown("Select ARIMA hyperparameters:")
        
        p_val = st.slider("Auto-Regressive order (p):", min_value=0, max_value=10, value=st.session_state.arima_order[0], help="Number of lag observations in the model.")
        d_val = st.slider("Differencing order (d):", min_value=0, max_value=3, value=st.session_state.d_order, help="Number of times the raw observations are differenced.")
        q_val = st.slider("Moving Average order (q):", min_value=0, max_value=10, value=st.session_state.arima_order[2], help="Size of the moving average window.")
        
        if st.button("Save Manual Order"):
            st.session_state.arima_order = (p_val, d_val, q_val)
            st.success(f"Configured order manually: ARIMA{st.session_state.arima_order}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("Auto ARIMA Tuning")
        st.markdown("Auto-ARIMA automatically tests combinations of (p, d, q) to optimize the Information Criterion.")
        
        info_criteria = st.selectbox("Optimizing Criterion:", ["aic", "bic"])
        max_p = st.number_input("Max AR order (p):", min_value=1, max_value=15, value=5)
        max_q = st.number_input("Max MA order (q):", min_value=1, max_value=15, value=5)
        
        run_auto = st.button("Execute Auto-ARIMA Search")
        
        if run_auto:
            with st.spinner("Finding best ARIMA parameters (this may take a moment)..."):
                try:
                    res = fit_auto_arima(df['Close'], max_p=int(max_p), max_d=3, max_q=int(max_q), information_criterion=info_criteria)
                    st.session_state.auto_arima_results = res
                    st.session_state.arima_order = res['best_order']
                    st.session_state.d_order = res['best_order'][1]
                    
                    st.success(f"Optimal configuration identified: **ARIMA{st.session_state.arima_order}**")
                except Exception as e:
                    st.error(f"Auto-ARIMA failed: {str(e)}")
                    
        # Show auto results if they exist
        if st.session_state.auto_arima_results is not None:
            res = st.session_state.auto_arima_results
            st.markdown(f"""
            **Auto-ARIMA Best Fit Details:**
            * Best Order: `ARIMA{res['best_order']}`
            * AIC: `{res['aic']:.4f}`
            * BIC: `{res['bic']:.4f}`
            """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Standard Pipeline check
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Selected Model Configuration")
    st.markdown(f"The active model order for training in the next steps is set to: **ARIMA{st.session_state.arima_order}**")
    st.markdown("</div>", unsafe_allow_html=True)


# PAGE 7: MODEL TRAINING
elif page == "7. Model Training":
    st.markdown("""
    <div class="app-header">
        <h1>Model Training & Validation</h1>
        <p>Split the dataset chronologically and fit the ARIMA model on the training series.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is None:
        st.warning("Please upload or download a dataset in the 'Dataset Upload' page first.")
        st.stop()
        
    df = st.session_state.df_cleaned
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Data Split Configuration")
    
    train_ratio = st.select_slider(
        "Select Training Data Ratio:",
        options=[0.7, 0.8, 0.9],
        value=st.session_state.train_ratio,
        format_func=lambda x: f"{int(x*100)}% Train / {int((1-x)*100)}% Test"
    )
    st.session_state.train_ratio = train_ratio
    
    train_df, test_df = train_test_split_data(df, train_ratio=train_ratio)
    
    col_sp1, col_sp2 = st.columns(2)
    with col_sp1:
        st.markdown(f"**Training Set (History):** {len(train_df)} observations")
        st.markdown(f"Timeline: `{train_df.index.min().strftime('%Y-%m-%d')}` to `{train_df.index.max().strftime('%Y-%m-%d')}`")
    with col_sp2:
        st.markdown(f"**Testing Set (Hold-out):** {len(test_df)} observations")
        st.markdown(f"Timeline: `{test_df.index.min().strftime('%Y-%m-%d')}` to `{test_df.index.max().strftime('%Y-%m-%d')}`")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Train Button
    train_btn = st.button("Train ARIMA Model")
    
    if train_btn:
        with st.spinner(f"Fitting ARIMA{st.session_state.arima_order} on training data..."):
            model_fit, error_msg = fit_manual_arima(
                train_df['Close'],
                st.session_state.arima_order[0],
                st.session_state.arima_order[1],
                st.session_state.arima_order[2]
            )
            
            if error_msg:
                st.error(f"Model Fit Failed: {error_msg}")
                st.session_state.model_fit = None
            else:
                st.session_state.model_fit = model_fit
                st.session_state.train_df = train_df
                st.session_state.test_df = test_df
                
                # Automatically pre-calculate test forecasts
                test_forecast = generate_test_forecast(model_fit, test_df['Close'])
                st.session_state.test_forecast_df = test_forecast
                
                metrics = calculate_evaluation_metrics(test_forecast['Actual'].values, test_forecast['Predicted'].values)
                st.session_state.evaluation_metrics = metrics
                
                # Pre-calculate future forecast
                future_forecast = generate_future_forecast(model_fit, steps=st.session_state.forecast_days, last_date=df.index[-1], freq=df.index.freqstr or 'B')
                st.session_state.future_forecast_df = future_forecast
                
                st.success("ARIMA Model Fitted Successfully!")
                
    if st.session_state.model_fit is not None:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("Model Diagnostic Summary")
        st.text(st.session_state.model_fit.summary())
        st.markdown("</div>", unsafe_allow_html=True)


# PAGE 8: FORECASTING (TEST SET)
elif page == "8. Forecasting":
    st.markdown("""
    <div class="app-header">
        <h1>Validation Forecasting</h1>
        <p>Compare predicted prices vs actual historical test values during the hold-out validation period.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model_fit is None or st.session_state.test_forecast_df is None:
        st.warning("Please train the model on the 'Model Training' page first.")
        st.stop()
        
    train_df = st.session_state.train_df
    test_df = st.session_state.test_df
    test_forecast_df = st.session_state.test_forecast_df
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Actual Test Data vs Forecast Comparison")
    fig = vis.plot_forecast_comparison(train_df, test_df, test_forecast_df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Detailed Validation Forecast Table")
    # Clean output columns
    tbl = test_forecast_df.copy()
    tbl['Date'] = tbl.index.strftime('%Y-%m-%d')
    tbl = tbl[['Date', 'Actual', 'Predicted', 'Lower_CI', 'Upper_CI', 'Error', 'Abs_Percent_Error']]
    st.dataframe(tbl.reset_index(drop=True), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# PAGE 9: MODEL EVALUATION
elif page == "9. Model Evaluation":
    st.markdown("""
    <div class="app-header">
        <h1>Model Evaluation</h1>
        <p>Analyze model error regression metrics and residual distributions for white noise verification.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model_fit is None or st.session_state.evaluation_metrics is None:
        st.warning("Please train the model on the 'Model Training' page first.")
        st.stop()
        
    metrics = st.session_state.evaluation_metrics
    
    # KPI Grid
    st.subheader("Model Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.4f}", "Penalizes large errors")
    with c2:
        render_kpi("Mean Absolute Error (MAE)", f"{metrics['MAE']:.4f}", "Average error magnitude")
    with c3:
        render_kpi("Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE']:.2f}%", f"Rating: {metrics['Rating']}")
    with c4:
        render_kpi("R-squared (R² Score)", f"{metrics['R2']:.4f}", "Proportion of variance explained")
        
    st.info(f"**Best Forecast Performance Indicator Summary:** {metrics['Interpretation']}")
    
    # Residual Plots
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Residuals Diagnostics Plot")
    fig_resid = vis.plot_residuals_diagnostics(st.session_state.model_fit)
    st.pyplot(fig_resid)
    st.markdown("""
    * **Residuals Time Series:** Residuals should fluctuate randomly around 0 (no pattern or heteroskedasticity).
    * **Residuals Density Distribution:** Should follow a normal distribution centered at 0.
    * **Autocorrelation (ACF) Plot:** Lags should remain within the blue critical boundaries. Significantly high lags suggest unresolved autocorrelations that could be modeled further.
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# PAGE 10: FUTURE PREDICTIONS
elif page == "10. Future Predictions":
    st.markdown("""
    <div class="app-header">
        <h1>Future Price Forecasting</h1>
        <p>Generate out-of-sample closing price forecasts with 95% confidence bands for the selected trading period.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model_fit is None:
        st.warning("Please train the model on the 'Model Training' page first.")
        st.stop()
        
    df = st.session_state.df_cleaned
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Forecast Period Settings")
    
    f_periods = st.selectbox(
        "Choose Forecast Period:",
        ["7 Days", "15 Days", "30 Days", "60 Days", "Custom"]
    )
    
    if f_periods == "7 Days":
        steps = 7
    elif f_periods == "15 Days":
        steps = 15
    elif f_periods == "30 Days":
        steps = 30
    elif f_periods == "60 Days":
        steps = 60
    else:
        steps = st.number_input("Enter Custom Days:", min_value=1, max_value=365, value=st.session_state.forecast_days)
        
    st.session_state.forecast_days = steps
    
    # Refit / recalculate future forecast
    if st.button("Generate Out-of-Sample Predictions"):
        with st.spinner(f"Projecting next {steps} trading sessions..."):
            future_forecast = generate_future_forecast(
                st.session_state.model_fit,
                steps=steps,
                last_date=df.index[-1],
                freq=df.index.freqstr or 'B'
            )
            st.session_state.future_forecast_df = future_forecast
            st.success("Future forecasts generated!")
            
    if st.session_state.future_forecast_df is not None:
        future_forecast_df = st.session_state.future_forecast_df
        
        st.subheader("Future Forecast Chart")
        fig = vis.plot_future_forecast(df, future_forecast_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f"Future Closing Price Predictions Table (Next {steps} Days)")
        tbl_future = future_forecast_df.copy()
        tbl_future['Date'] = tbl_future.index.strftime('%Y-%m-%d')
        tbl_future = tbl_future[['Date', 'Forecast', 'Lower_CI', 'Upper_CI']]
        st.dataframe(tbl_future.reset_index(drop=True), use_container_width=True)
        
    st.markdown("</div>", unsafe_allow_html=True)


# PAGE 11: EXPORT REPORTS
elif page == "11. Export Reports":
    st.markdown("""
    <div class="app-header">
        <h1>Export Forecasts & Reports</h1>
        <p>Export evaluation summaries, price datasets, and download PDF financial evaluation reports.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model_fit is None or st.session_state.future_forecast_df is None:
        st.warning("Please train the model and generate future predictions first.")
        st.stop()
        
    df = st.session_state.df_cleaned
    test_forecast_df = st.session_state.test_forecast_df
    future_forecast_df = st.session_state.future_forecast_df
    metrics = st.session_state.evaluation_metrics
    order = st.session_state.arima_order
    stationarity = st.session_state.stationarity_info
    crossover_df = st.session_state.crossover_df
    ticker = st.session_state.ticker
    
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("Download Center")
    st.markdown("Select your desired export file format below:")
    
    # 1. CSV
    csv_data = rep.export_csv(future_forecast_df)
    st.download_button(
        label="Download Forecast Predictions (CSV)",
        data=csv_data,
        file_name=f"{ticker}_arima_forecast.csv",
        mime="text/csv"
    )
    
    # 2. Excel
    excel_data = rep.export_excel(df, test_forecast_df, future_forecast_df, metrics, order, ticker)
    st.download_button(
        label="Download Full Excel Workbook (XLSX)",
        data=excel_data,
        file_name=f"{ticker}_forecasting_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # 3. PDF Report
    pdf_data = rep.export_pdf_report(df, test_forecast_df, future_forecast_df, metrics, order, stationarity, crossover_df, ticker)
    st.download_button(
        label="Download Comprehensive PDF Evaluation Report (PDF)",
        data=pdf_data,
        file_name=f"{ticker}_forecast_report.pdf",
        mime="application/pdf"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Render PDF content preview as markdown
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.subheader("AI-Generated Recommendations & Findings Preview")
    insights = rep.generate_business_insights(ticker, metrics, future_forecast_df, crossover_df)
    for ins in insights:
        st.markdown(f"* {ins}")
    st.markdown("</div>", unsafe_allow_html=True)
