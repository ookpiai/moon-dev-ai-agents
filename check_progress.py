#!/usr/bin/env python3
"""
Quick progress checker for Polymarket data collection
Usage: python check_progress.py
"""

import sys
import codecs
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# File path
data_file = Path('src/data/polymarket/training_data/market_snapshots.csv')

if not data_file.exists():
    print("❌ No data file found yet. Collector may still be initializing.")
    exit(1)

# Load data
df = pd.read_csv(data_file)
snapshots = len(df)

# Targets
MIN_REQUIRED = 1440  # 24 hours
OPTIMAL = 10080      # 1 week

# Calculate progress
pct_min = (snapshots / MIN_REQUIRED) * 100
pct_opt = (snapshots / OPTIMAL) * 100

# Estimate time remaining (20 snapshots/hour)
hours_to_min = max(0, (MIN_REQUIRED - snapshots) / 20)
hours_to_opt = max(0, (OPTIMAL - snapshots) / 20)

# Get time range
if len(df) > 0:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_time = df['timestamp'].min()
    last_time = df['timestamp'].max()
    duration = last_time - start_time
else:
    start_time = "N/A"
    last_time = "N/A"
    duration = timedelta(0)

# Print report
print("=" * 80)
print("📊 POLYMARKET DATA COLLECTION PROGRESS")
print("=" * 80)
print(f"\n📈 Current Status:")
print(f"   Snapshots collected: {snapshots:,}")
print(f"   Collection started: {start_time}")
print(f"   Last snapshot: {last_time}")
print(f"   Duration: {duration}")

print(f"\n📊 Progress to MINIMUM (24 hours - {MIN_REQUIRED:,} snapshots):")
print(f"   {pct_min:.1f}% complete")
print(f"   {max(0, MIN_REQUIRED - snapshots):,} snapshots remaining")
print(f"   ~{hours_to_min:.1f} hours to go")

print(f"\n🎯 Progress to OPTIMAL (1 week - {OPTIMAL:,} snapshots):")
print(f"   {pct_opt:.1f}% complete")
print(f"   {max(0, OPTIMAL - snapshots):,} snapshots remaining")
print(f"   ~{hours_to_opt / 24:.1f} days to go")

# Unique markets
unique_markets = df['market_id'].nunique()
print(f"\n🔍 Market Coverage:")
print(f"   Unique markets tracked: {unique_markets}")

# Market types
if 'market_type' in df.columns:
    market_types = df['market_type'].value_counts()
    print(f"\n📋 Market Types:")
    for mtype, count in market_types.items():
        print(f"   {mtype}: {count}")

# Regimes
if 'regime' in df.columns:
    regimes = df['regime'].value_counts()
    print(f"\n💡 Market Regimes:")
    for regime, count in regimes.items():
        print(f"   {regime}: {count}")

print("\n" + "=" * 80)

# Status message
if snapshots >= OPTIMAL:
    print("✅ OPTIMAL data collected! Ready to train meta-learner.")
elif snapshots >= MIN_REQUIRED:
    print("✅ MINIMUM data collected! You can train now, but more data = better results.")
else:
    print("⏳ Still collecting... Let it run for at least 24 hours.")

print("=" * 80 + "\n")
