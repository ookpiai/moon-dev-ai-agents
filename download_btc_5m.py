"""
Download BTC 5-minute data using yfinance
"""
import yfinance as yf
import pandas as pd

print("Downloading BTC-USD 5-minute data...")

# Download 60 days of 5-minute data (max for 5m interval)
btc = yf.Ticker('BTC-USD')
df = btc.history(period='60d', interval='5m')

# Format for backtesting
df.index.name = 'datetime'
df.columns = df.columns.str.lower()

# Save to CSV
output_path = 'src/data/rbi/BTC-USD-5m.csv'
df.to_csv(output_path)

print(f"[OK] Downloaded {len(df)} bars")
print(f"  From: {df.index[0]}")
print(f"  To:   {df.index[-1]}")
print(f"  Saved to: {output_path}")
