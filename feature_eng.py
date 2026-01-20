import pandas as pd
import numpy as np

def calculate_rsi(series, window=14):
    """
    Calculates RSI manually using Pandas.
    Formula: 100 - (100 / (1 + RS))
    """
    delta = series.diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Calculate Exponential Moving Average (EMA) of gains and losses
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

def feature_engineering(df_prices):
    """
    Takes raw prices (Adj Close) and creates a rich feature dataset.
    """
    print("\n[Feature Engineering] Generating Technical Indicators...")
    features = pd.DataFrame(index=df_prices.index)
    
    # 1. Log Returns (The Target)
    # We want to predict this!
    features['log_return'] = np.log(df_prices / df_prices.shift(1))
    
    # 2. Rolling Volatility (20-day standard deviation of returns)
    # This tells the AI the "Risk Regime"
    features['volatility_20'] = features['log_return'].rolling(window=20).std()
    
    # 3. Simple Moving Average (SMA) Gap
    # Price relative to 50-day average (Normalized trend)
    # If > 0, price is above trend.
    sma_50 = df_prices.rolling(window=50).mean()
    features['trend_sma_50'] = (df_prices - sma_50) / sma_50
    
    # 4. RSI (Relative Strength Index)
    features['rsi'] = calculate_rsi(df_prices, window=14)
    
    # 5. Momentum (Return over last 10 days)
    features['momentum_10'] = df_prices.pct_change(10)
    
    # CLEANUP: Drop NaN values created by rolling windows (The first 50 days will be empty)
    features = features.dropna()
    
    print(f"✔ Features created. Shape: {features.shape}")
    return features

# --- TEST HARNESS ---
# (Run this to verify the math works on dummy data)
if __name__ == "__main__":
    # Create dummy data
    dates = pd.date_range(start="2023-01-01", periods=100)
    prices = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates)
    
    print("Testing Feature Engineering on dummy data...")
    df_features = feature_engineering(prices)
    
    print("\nHead of Features:")
    print(df_features.head())
    
    print("\nTail of Features:")
    print(df_features.tail())