"""
Validate Polymarket Data Quality
Quick script to check if collected data has quality issues
"""

import pandas as pd
from pathlib import Path

data_file = Path("src/data/polymarket/training_data/market_snapshots.csv")

if not data_file.exists():
    print("[ERROR] No data file found yet")
    exit(1)

df = pd.read_csv(data_file)

print(f"\n{'='*60}")
print(f"POLYMARKET DATA QUALITY VALIDATION")
print(f"{'='*60}\n")

print(f"Total Snapshots: {len(df)}")
print(f"Unique Markets: {df['market_id'].nunique()}")

print(f"\n[TIME] time_to_resolution_days:")
print(f"   Mean:   {df['time_to_resolution_days'].mean():.1f} days")
print(f"   Median: {df['time_to_resolution_days'].median():.1f} days")
print(f"   Min:    {df['time_to_resolution_days'].min():.1f} days")
print(f"   Max:    {df['time_to_resolution_days'].max():.1f} days")

# Check for broken data (999 = placeholder)
broken_time = (df['time_to_resolution_days'] == 999).sum()
if broken_time > 0:
    print(f"   [WARN] WARNING: {broken_time} markets with time=999 (broken)")
else:
    print(f"   [OK] All markets have valid time_to_resolution")

print(f"\n[SPREAD] spread:")
print(f"   Mean:   {df['spread'].mean():.4f} ({df['spread'].mean()*100:.2f}%)")
print(f"   Median: {df['spread'].median():.4f} ({df['spread'].median()*100:.2f}%)")
print(f"   Min:    {df['spread'].min():.4f}")
print(f"   Max:    {df['spread'].max():.4f} ({df['spread'].max()*100:.2f}%)")

# Check for broken spreads (98% = closed markets)
broken_spread = (df['spread'] > 0.5).sum()
if broken_spread > 0:
    print(f"   [WARN] WARNING: {broken_spread} markets with spread >50% (closed?)")
else:
    print(f"   [OK] All markets have reasonable spreads")

print(f"\n[LIQUIDITY] liquidity:")
print(f"   Mean:   ${df['liquidity'].mean():,.0f}")
print(f"   Median: ${df['liquidity'].median():,.0f}")
print(f"   Min:    ${df['liquidity'].min():,.0f}")

print(f"\n[VOLUME] volume_24h:")
print(f"   Mean:   ${df['volume_24h'].mean():,.0f}")
print(f"   Median: ${df['volume_24h'].median():,.0f}")
print(f"   Min:    ${df['volume_24h'].min():,.0f}")

print(f"\n[TYPES] Market Types:")
for mtype, count in df['market_type'].value_counts().items():
    print(f"   {mtype}: {count}")

print(f"\n[REGIME] Regime Distribution:")
for regime, count in df['regime'].value_counts().items():
    print(f"   {regime}: {count}")

print(f"\n{'='*60}")

# Overall verdict
if broken_time == 0 and broken_spread == 0:
    print("[OK] DATA QUALITY: EXCELLENT")
    print("Ready for backtesting after 24-48 hours of collection")
elif broken_time < len(df) * 0.1 and broken_spread < len(df) * 0.1:
    print("[WARN] DATA QUALITY: GOOD (minor issues)")
else:
    print("[ERROR] DATA QUALITY: NEEDS FIXES")

print(f"{'='*60}\n")
