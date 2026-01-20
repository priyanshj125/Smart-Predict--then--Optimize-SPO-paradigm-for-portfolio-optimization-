import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
CONFIG = {
    "tickers": ['AAPL', 'MSFT', 'JPM', 'XOM', 'GLD', 'TLT'],
    "start_date": "2015-01-01",
    "end_date": "2024-01-01",
    "data_path": "data/processed_returns.csv" # Where to save the output
}

def create_directory_structure():
    """Creates the necessary folders if they don't exist."""
    if not os.path.exists('data'):
        os.makedirs('data')
        print("✔ Created 'data' directory.")

def fetch_data(tickers, start_date, end_date):
    """
    Ingest stage: Downloads Adjusted Close prices.
    """
    print(f"\n[1/3] Fetching data for {len(tickers)} assets...")
    try:
        # yfinance auto_adjust=True returns the Dividend/Split adjusted price
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        # Handling MultiIndex if yfinance returns it
        if isinstance(data.columns, pd.MultiIndex):
            # If 'Close' is in the top level, select it
            if 'Close' in data.columns.levels[0]:
                df = data['Close']
            else:
                # Fallback: sometimes yfinance returns just the tickers if auto_adjust is on
                df = data
        else:
            df = data['Close']
            
        # Basic cleaning
        df = df.ffill().dropna()
        print(f"✔ Download complete. Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"✘ Error downloading data: {e}")
        return pd.DataFrame()

def process_data(df_prices):
    """
    Transform stage: Calculates Log Returns.
    """
    print("\n[2/3] Processing data (Log Returns)...")
    
    # Log Returns = ln( Price_t / Price_{t-1} )
    df_log_ret = np.log(df_prices / df_prices.shift(1))
    
    # Drop the first row (NaN) created by the shift
    df_log_ret = df_log_ret.dropna()
    
    print(f"✔ Data processed. Missing values: {df_log_ret.isnull().sum().sum()}")
    return df_log_ret

def save_and_visualize(df_returns, output_path):
    """
    Load stage: Saves to CSV and generates a plot.
    """
    print("\n[3/3] Saving and Visualizing...")
    
    # 1. Save to CSV (The Feature Store)
    df_returns.to_csv(output_path)
    print(f"✔ Dataset saved to: {output_path}")
    
    # 2. Visualize Cumulative Returns
    plt.figure(figsize=(12, 6))
    
    # Cumulative sum of log returns = Total Return over time
    cumulative_returns = df_returns.cumsum()
    
    for col in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[col], label=col, linewidth=1.5)
        
    plt.title("Institutional Data Pipeline: Cumulative Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Growth (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    The Orchestrator Function.
    """
    
    create_directory_structure()
    
    # Step 1: Extract
    df_prices = fetch_data(CONFIG["tickers"], CONFIG["start_date"], CONFIG["end_date"])
    
    if not df_prices.empty:
        # Step 2: Transform
        df_returns = process_data(df_prices)
        
        # Step 3: Load & Visualize
        save_and_visualize(df_returns, CONFIG["data_path"])
        
        print("\n--- FINISHED SUCCESSFULLY ---")
    else:
        print("\n--- FAILED ---")

# This block allows the script to be run directly
if __name__ == "__main__":
    main()