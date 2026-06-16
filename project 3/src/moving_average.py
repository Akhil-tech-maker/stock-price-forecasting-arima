import pandas as pd
import numpy as np

def calculate_moving_averages(df: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
    """
    Calculate Simple Moving Averages (SMA 20, 50, 100, 200) and
    Exponential Moving Averages (EMA 20, 50).
    """
    ma_df = df.copy()
    
    # Simple Moving Averages
    ma_df['SMA_20'] = ma_df[column].rolling(window=20).mean()
    ma_df['SMA_50'] = ma_df[column].rolling(window=50).mean()
    ma_df['SMA_100'] = ma_df[column].rolling(window=100).mean()
    ma_df['SMA_200'] = ma_df[column].rolling(window=200).mean()
    
    # Exponential Moving Averages
    ma_df['EMA_20'] = ma_df[column].ewm(span=20, adjust=False).mean()
    ma_df['EMA_50'] = ma_df[column].ewm(span=50, adjust=False).mean()
    
    return ma_df

def detect_crossover_signals(ma_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect buy/sell crossovers:
    1. Golden Cross / Death Cross: SMA_50 vs SMA_200
    2. Short-term Trend Cross: EMA_20 vs EMA_50
    3. Price Crossover: Close vs SMA_20
    
    Returns a DataFrame listing the signal events.
    """
    signals = []
    
    # 1. SMA 50 vs SMA 200 (Golden/Death Cross)
    if 'SMA_50' in ma_df.columns and 'SMA_200' in ma_df.columns:
        # Drop rows where we don't have SMA 200 data
        sub_df = ma_df.dropna(subset=['SMA_50', 'SMA_200'])
        if len(sub_df) > 1:
            diff = sub_df['SMA_50'] - sub_df['SMA_200']
            # Find where sign changes
            crossings = np.diff(np.sign(diff))
            idx = sub_df.index[1:]
            
            for date, cross, prev_diff, curr_diff in zip(idx, crossings, diff[:-1], diff[1:]):
                close_val = sub_df.loc[date, 'Close']
                if cross > 0 or (prev_diff <= 0 and curr_diff > 0): # Cross above
                    signals.append({
                        'Date': date,
                        'Type': 'Golden Cross (SMA 50/200)',
                        'Signal': 'Buy',
                        'Price': float(close_val),
                        'Description': 'SMA 50 crossed above SMA 200 (Long-term Bullish Signal)'
                    })
                elif cross < 0 or (prev_diff >= 0 and curr_diff < 0): # Cross below
                    signals.append({
                        'Date': date,
                        'Type': 'Death Cross (SMA 50/200)',
                        'Signal': 'Sell',
                        'Price': float(close_val),
                        'Description': 'SMA 50 crossed below SMA 200 (Long-term Bearish Signal)'
                    })
                    
    # 2. EMA 20 vs EMA 50 (Short-term Crossover)
    if 'EMA_20' in ma_df.columns and 'EMA_50' in ma_df.columns:
        sub_df = ma_df.dropna(subset=['EMA_20', 'EMA_50'])
        if len(sub_df) > 1:
            diff = sub_df['EMA_20'] - sub_df['EMA_50']
            crossings = np.diff(np.sign(diff))
            idx = sub_df.index[1:]
            
            for date, cross, prev_diff, curr_diff in zip(idx, crossings, diff[:-1], diff[1:]):
                close_val = sub_df.loc[date, 'Close']
                if cross > 0 or (prev_diff <= 0 and curr_diff > 0):
                    signals.append({
                        'Date': date,
                        'Type': 'Short-term Cross (EMA 20/50)',
                        'Signal': 'Buy',
                        'Price': float(close_val),
                        'Description': 'EMA 20 crossed above EMA 50 (Short-term Bullish)'
                    })
                elif cross < 0 or (prev_diff >= 0 and curr_diff < 0):
                    signals.append({
                        'Date': date,
                        'Type': 'Short-term Cross (EMA 20/50)',
                        'Signal': 'Sell',
                        'Price': float(close_val),
                        'Description': 'EMA 20 crossed below EMA 50 (Short-term Bearish)'
                    })
                    
    # Create DataFrame and sort by Date descending (most recent first)
    if signals:
        signals_df = pd.DataFrame(signals).sort_values(by='Date', ascending=False)
        signals_df = signals_df.reset_index(drop=True)
    else:
        signals_df = pd.DataFrame(columns=['Date', 'Type', 'Signal', 'Price', 'Description'])
        
    return signals_df
