import pandas as pd
import os

print("=" * 80)
print("POLYMARKET SYSTEM STATUS")
print("=" * 80)

# Check data files
data_dir = "src/data/polymarket/training_data"
if os.path.exists(f"{data_dir}/market_snapshots.csv"):
    df = pd.read_csv(f"{data_dir}/market_snapshots.csv")
    print(f"\n[DATA] Market Snapshots: {len(df):,} records")
    print(f"[DATA] File size: {os.path.getsize(f'{data_dir}/market_snapshots.csv') / 1024:.1f}KB")
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"[DATA] Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"[DATA] Unique markets: {df['market_id'].nunique()}")

if os.path.exists(f"{data_dir}/orderbook_snapshots.csv"):
    df_ob = pd.read_csv(f"{data_dir}/orderbook_snapshots.csv")
    print(f"\n[DATA] Order Book Snapshots: {len(df_ob):,} records")

print("\n" + "=" * 80)
print("READY TO:")
print("  1. Continue data collection (running in background)")
print("  2. Run meta-learner training once we have 24+ hours of data")
print("  3. Start orchestrator for live trading simulation")
print("=" * 80)
