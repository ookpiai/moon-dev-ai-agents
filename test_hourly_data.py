"""Quick test on hourly data"""
from backtesting import Backtest
import pandas as pd
import sys
sys.path.append('src/data/pinescript_conversions/backtests')
from ADX_Squeeze_R_Based_BT_v2 import ADX_Squeeze_R_Based

# Load hourly data
df = pd.read_csv("src/data/rbi/BTC-USD-1h-2020-2025.csv")
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.rename(columns={'datetime': 'Datetime', 'open': 'Open', 'high': 'High',
                        'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

print("="*80)
print("HOURLY DATA TEST (2023-2025)")
print("="*80)
print(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
print(f"Duration: {(df.index[-1] - df.index[0]).days} days")
print("="*80)

bt = Backtest(df, ADX_Squeeze_R_Based, cash=100000, commission=0.001, exclusive_orders=True)
stats = bt.run()

print("\nRESULTS:")
print(f"Return: {stats['Return [%]']:.2f}%")
print(f"Max DD: {stats['Max. Drawdown [%]']:.2f}%")
print(f"Trades: {stats['# Trades']}")
print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
print(f"Profit Factor: {stats['Profit Factor']:.2f}")
print(f"Sharpe: {stats['Sharpe Ratio']:.2f}")
print("="*80)
