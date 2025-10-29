"""
Download BTC 15-minute data for 2025 using yfinance
"""
import yfinance as yf
import pandas as pd

print("Downloading BTC-USD 15-minute data...")

# Download max available 15m data (yfinance limit is 60 days for 15m)
btc = yf.Ticker('BTC-USD')
df = btc.history(period='60d', interval='15m')

# Format for backtesting
df.index.name = 'datetime'
df.columns = df.columns.str.lower()

# Save to CSV
output_path = 'src/data/rbi/BTC-USD-15m-2025.csv'
df.to_csv(output_path)

print(f"[OK] Downloaded {len(df)} bars")
print(f"  From: {df.index[0]}")
print(f"  To:   {df.index[-1]}")
print(f"  Saved to: {output_path}")
