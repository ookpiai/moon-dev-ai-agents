"""
Extract Daily NAV and Trades List from ADX Squeeze Backtest
"""
from backtesting import Backtest
import pandas as pd
import sys
sys.path.append('src/data/pinescript_conversions/backtests')
from ADX_Squeeze_R_Based_BT_v2 import ADX_Squeeze_R_Based

print("="*80)
print("EXTRACTING BACKTEST DATA - ADX SQUEEZE STRATEGY")
print("="*80)

# Load 15m data from 2023
df = pd.read_csv("src/data/rbi/BTC-USD-15m.csv")
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.rename(columns={'datetime': 'Datetime', 'open': 'Open', 'high': 'High',
                        'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"\nRunning backtest on {len(df)} bars ({df.index[0]} to {df.index[-1]})...")
print()

# Run backtest
bt = Backtest(df, ADX_Squeeze_R_Based, cash=100000, commission=0.001, exclusive_orders=True)
stats = bt.run()

# Extract equity curve (all timestamps)
equity_curve = stats['_equity_curve']
print(f"[OK] Extracted equity curve: {len(equity_curve)} data points")

# Resample to daily NAV (end-of-day values)
daily_nav = equity_curve['Equity'].resample('D').last().dropna()
daily_nav_df = pd.DataFrame({
    'Date': daily_nav.index,
    'NAV': daily_nav.values,
    'Daily_Return_Pct': daily_nav.pct_change() * 100
})
daily_nav_df['Date'] = daily_nav_df['Date'].dt.strftime('%Y-%m-%d')

print(f"[OK] Resampled to daily NAV: {len(daily_nav_df)} days")

# Extract trades
trades = stats['_trades']
if len(trades) > 0:
    trades_df = trades.copy()

    # Format datetime columns
    if 'EntryTime' in trades_df.columns:
        trades_df['EntryTime'] = pd.to_datetime(trades_df['EntryTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    if 'ExitTime' in trades_df.columns:
        trades_df['ExitTime'] = pd.to_datetime(trades_df['ExitTime']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Calculate additional metrics
    if 'ReturnPct' in trades_df.columns:
        trades_df['PnL_USD'] = trades_df['ReturnPct'] * 1000  # Approximate based on position size

    print(f"[OK] Extracted {len(trades_df)} trades")
    print(f"    Win Rate: {(trades_df['ReturnPct'] > 0).sum() / len(trades_df) * 100:.2f}%")
    print(f"    Avg Return: {trades_df['ReturnPct'].mean():.2f}%")
else:
    trades_df = pd.DataFrame()
    print("[WARN] No trades found")

# Save to CSV
nav_path = "backtest_daily_nav.csv"
trades_path = "backtest_trades.csv"

daily_nav_df.to_csv(nav_path, index=False)
print(f"\n[SAVED] Daily NAV -> {nav_path}")

if len(trades_df) > 0:
    trades_df.to_csv(trades_path, index=False)
    print(f"[SAVED] Trades list -> {trades_path}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
print(f"Starting NAV: ${daily_nav_df['NAV'].iloc[0]:,.2f}")
print(f"Ending NAV: ${daily_nav_df['NAV'].iloc[-1]:,.2f}")
print(f"Total Return: {((daily_nav_df['NAV'].iloc[-1] / daily_nav_df['NAV'].iloc[0]) - 1) * 100:.2f}%")
print(f"Total Trades: {len(trades_df)}")
print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
print("="*80)

# Display first few rows
print("\nDAILY NAV (first 10 days):")
print(daily_nav_df.head(10).to_string(index=False))

if len(trades_df) > 0:
    print("\nTRADES (first 10):")
    print(trades_df[['EntryTime', 'ExitTime', 'Size', 'EntryPrice', 'ExitPrice', 'ReturnPct']].head(10).to_string(index=False))

print("\n[OK] Data extraction complete!")
