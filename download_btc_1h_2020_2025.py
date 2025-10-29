"""
Download BTC 1-hour data from 2020-2025 using yfinance
"""
import yfinance as yf
import pandas as pd

print("Downloading BTC-USD 1-hour data (2020-2025)...")

# Download max available hourly data (730 days = 2 years for 1h interval)
btc = yf.Ticker('BTC-USD')
df = btc.history(period='max', interval='1h')

# Filter to 2020 onwards
df = df[df.index >= '2020-01-01']

# Format for backtesting
df.index.name = 'datetime'
df.columns = df.columns.str.lower()

# Save to CSV
output_path = 'src/data/rbi/BTC-USD-1h-2020-2025.csv'
df.to_csv(output_path)

print(f"[OK] Downloaded {len(df)} bars")
print(f"  From: {df.index[0]}")
print(f"  To:   {df.index[-1]}")
print(f"  Duration: {(df.index[-1] - df.index[0]).days} days")
print(f"  Saved to: {output_path}")
