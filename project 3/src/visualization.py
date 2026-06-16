import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

# Define standard aesthetic colors
COLORS = {
    'primary': '#636EFA',   # Blue
    'secondary': '#EF553B', # Orange
    'success': '#00CC96',   # Green (Buy)
    'danger': '#AB63FA',    # Purple (Sell)
    'dark': '#19D3F3',      # Cyan
    'light': '#E5ECF6',
    'neutral': '#7F7F7F',
    'fit': '#FFA15A',       # Orange-Yellow
}

def plot_close_price(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color=COLORS['primary'], width=2)))
    
    fig.update_layout(
        title='Historical Closing Price',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_volume(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=COLORS['dark']))
    
    fig.update_layout(
        title='Trading Volume',
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_returns(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Daily_Return_Pct'], mode='lines', name='Daily Return (%)', line=dict(color=COLORS['secondary'], width=1)))
    
    fig.update_layout(
        title='Daily Returns (%)',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        template='plotly_white',
        hovermode='x'
    )
    return fig

def plot_price_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['Close'],
        nbinsx=50,
        name='Close Price',
        marker_color=COLORS['primary'],
        opacity=0.75,
        histnorm='probability density'
    ))
    
    # Add KDE-like line
    # (Simplified: we'll just plot a normal distribution curve based on statistics)
    mean = df['Close'].mean()
    std = df['Close'].std()
    x_range = np.linspace(df['Close'].min(), df['Close'].max(), 200)
    y_normal = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x_range - mean) ** 2) / (2 * std ** 2))
    
    fig.add_trace(go.Scatter(x=x_range, y=y_normal, mode='lines', name='Normal Approximation', line=dict(color='red', width=2)))
    
    fig.update_layout(
        title='Price Distribution',
        xaxis_title='Price',
        yaxis_title='Density',
        template='plotly_white',
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='Viridis',
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        template='plotly_white',
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_rolling_stats(df: pd.DataFrame, window: int = 20) -> go.Figure:
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color=COLORS['primary'], width=1.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=rolling_mean, name=f'Rolling Mean ({window}d)', line=dict(color=COLORS['fit'], width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=rolling_std, name=f'Rolling Std ({window}d)', line=dict(color=COLORS['secondary'], width=1.5, dash='dash')), secondary_y=True)
    
    fig.update_layout(
        title=f'Rolling Statistics (Window = {window} Days)',
        xaxis_title='Date',
        template='plotly_white',
        hovermode='x unified'
    )
    fig.update_yaxes(title_text="Close & Mean Price", secondary_y=False)
    fig.update_yaxes(title_text="Standard Deviation (Volatility)", secondary_y=True)
    
    return fig

def plot_moving_averages(ma_df: pd.DataFrame, crossover_df: pd.DataFrame = None) -> go.Figure:
    fig = go.Figure()
    
    # Base price
    fig.add_trace(go.Scatter(x=ma_df.index, y=ma_df['Close'], name='Close Price', line=dict(color='#000000', width=1.5)))
    
    # SMAs
    for col, color, width in [('SMA_20', '#1f77b4', 1.0), ('SMA_50', '#ff7f0e', 1.2), ('SMA_100', '#2ca02c', 1.2), ('SMA_200', '#d62728', 1.5)]:
        if col in ma_df.columns:
            fig.add_trace(go.Scatter(x=ma_df.index, y=ma_df[col], name=col, line=dict(color=color, width=width)))
            
    # EMAs
    for col, color, width in [('EMA_20', '#9467bd', 1.0), ('EMA_50', '#8c564b', 1.2)]:
        if col in ma_df.columns:
            fig.add_trace(go.Scatter(x=ma_df.index, y=ma_df[col], name=col, line=dict(color=color, width=width, dash='dash')))
            
    # Overlay Buy/Sell crossover signals
    if crossover_df is not None and not crossover_df.empty:
        # Buy signals
        buys = crossover_df[crossover_df['Signal'] == 'Buy']
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys['Date'], y=buys['Price'],
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=1, color='black')),
                hovertext=buys['Type']
            ))
            
        # Sell signals
        sells = crossover_df[crossover_df['Signal'] == 'Sell']
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells['Date'], y=sells['Price'],
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1, color='black')),
                hovertext=sells['Type']
            ))
            
    fig.update_layout(
        title='Closing Price and Moving Averages with Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_forecast_comparison(train_df: pd.DataFrame, test_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    # Train
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Close'], name='Training Data', line=dict(color=COLORS['neutral'], width=1.5)))
    
    # Test Actual
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Close'], name='Actual Test Data', line=dict(color=COLORS['primary'], width=2)))
    
    # Test Predicted
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted'], name='Predicted Test Data (Forecast)', line=dict(color=COLORS['secondary'], width=2)))
    
    # Confidence Interval Bounds
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['Upper_CI'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='Upper CI'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['Lower_CI'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(239, 85, 59, 0.15)', # Semi-transparent secondary color
        name='95% Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title='Model Forecast vs Actual Test Data',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_future_forecast(df: pd.DataFrame, future_df: pd.DataFrame, show_history_months: int = 6) -> go.Figure:
    fig = go.Figure()
    
    # Slice historical data to show last N months (for visual clarity)
    last_date = df.index[-1]
    history_start = last_date - pd.DateOffset(months=show_history_months)
    sliced_history = df.loc[history_start:]
    
    # Historical Close
    fig.add_trace(go.Scatter(x=sliced_history.index, y=sliced_history['Close'], name='Historical Close', line=dict(color=COLORS['primary'], width=2)))
    
    # Future Forecast
    fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Forecast'], name='Future Forecast', line=dict(color=COLORS['success'], width=2)))
    
    # Upper/Lower CI Bounds
    fig.add_trace(go.Scatter(
        x=future_df.index, y=future_df['Upper_CI'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='Upper Bound'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_df.index, y=future_df['Lower_CI'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0, 204, 150, 0.15)', # Semi-transparent green
        name='95% Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f'Future Price Forecasting (Next {len(future_df)} periods)',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_residuals_diagnostics(model_fit) -> plt.Figure:
    """
    Generate residual diagnostic plots using Matplotlib and Seaborn.
    Creates 3 subplots:
    1. Residuals over time
    2. Residuals distribution (Histogram + KDE)
    3. Residuals Autocorrelation (ACF)
    """
    residuals = model_fit.resid
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Set standard seaborn style
    sns.set_theme(style='whitegrid')
    
    # 1. Residuals over time
    axes[0].plot(residuals.index, residuals.values, color='#3182bd', lw=1)
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title('Residuals Time Series')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Residual')
    
    # 2. Residuals distribution (Histogram + KDE)
    sns.histplot(residuals.values, kde=True, ax=axes[1], color='#e6550d', stat='density')
    
    # Fit normal distribution and plot
    from scipy.stats import norm
    mu, std = norm.fit(residuals)
    xmin, xmax = axes[1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axes[1].plot(x, p, 'k', linewidth=2, label=f'Normal ($\mu$={mu:.2f}, $\sigma$={std:.2f})')
    axes[1].legend()
    axes[1].set_title('Residuals Density Distribution')
    axes[1].set_xlabel('Residual Value')
    
    # 3. Autocorrelation plot (ACF)
    plot_acf(residuals, ax=axes[2], lags=min(40, len(residuals)-2), color='#31a354')
    axes[2].set_title('Residuals Autocorrelation (ACF)')
    axes[2].set_xlabel('Lag')
    axes[2].set_ylabel('Autocorrelation')
    
    plt.tight_layout()
    return fig
