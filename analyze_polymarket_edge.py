#!/usr/bin/env python3
"""
Analyze Polymarket backtest results to assess edge
"""
import json
import pandas as pd
import sys

print("=" * 80)
print("POLYMARKET EDGE ANALYSIS")
print("=" * 80)
print()

# Load backtest stats
backtest_file = "src/data/polymarket/backtests/backtest_stats_20251028_135615.json"
trades_file = "src/data/polymarket/backtests/backtest_trades_20251028_135615.csv"

with open(backtest_file, 'r') as f:
    stats = json.load(f)

trades_df = pd.read_csv(trades_file)

print("📊 BACKTEST SUMMARY")
print("-" * 80)
print(f"Period: {stats['backtest_period']}")
print(f"Total Snapshots: {stats['total_snapshots']:,}")
print(f"Total Signals Generated: {stats['total_signals']}")
print(f"Total Trades Executed: {stats['total_trades']}")
print()

print("💰 PERFORMANCE METRICS")
print("-" * 80)
print(f"Initial Capital: ${stats['initial_capital']:,.2f}")
print(f"Final Capital: ${stats['final_capital']:,.2f}")
print(f"Total PnL: ${stats['total_pnl']:,.2f}")
print(f"Total Return: {stats['total_return']*100:.2f}%")
print(f"Max Drawdown: ${stats['max_drawdown']:,.2f}")
print()

print("🎯 WIN/LOSS ANALYSIS")
print("-" * 80)
print(f"Winning Trades: {stats['winning_trades']}")
print(f"Losing Trades: {stats['losing_trades']}")
print(f"Win Rate: {stats['win_rate']*100:.1f}%")
print(f"Average Win: ${stats['avg_win']:.2f}")
print(f"Average Loss: ${stats['avg_loss']:.2f}")
print(f"Average PnL per Trade: ${stats['avg_pnl']:.2f}")
print(f"Profit Factor: {stats['profit_factor']:.2f}")
print()

print("🚪 EXIT ANALYSIS")
print("-" * 80)
for exit_type, count in stats['exit_rules'].items():
    pct = (count / stats['total_trades']) * 100
    print(f"{exit_type}: {count} trades ({pct:.1f}%)")
print()

print("🔍 TRADE BREAKDOWN")
print("-" * 80)
print(f"Entry EV Range: {trades_df['entry_ev'].min():.4f} to {trades_df['entry_ev'].max():.4f}")
print(f"Entry Z-Score Range: {trades_df['entry_z'].min():.2f} to {trades_df['entry_z'].max():.2f}")
print(f"Average Position Size: ${trades_df['size'].mean():.2f}")
print(f"Max Position Size: ${trades_df['size'].max():.2f}")
print()

print("📉 WORST TRADES")
print("-" * 80)
worst_trades = trades_df.nsmallest(5, 'pnl')[['question', 'entry_price', 'exit_price', 'pnl', 'exit_reason']]
for idx, trade in worst_trades.iterrows():
    print(f"Loss ${trade['pnl']:.2f}: {trade['question'][:50]}")
    print(f"  Entry: {trade['entry_price']:.3f} → Exit: {trade['exit_price']:.3f} ({trade['exit_reason']})")
print()

print("=" * 80)
print("🎯 EDGE ASSESSMENT")
print("=" * 80)
print()

if stats['win_rate'] == 0:
    print("❌ NO EDGE DETECTED")
    print()
    print("CRITICAL FINDINGS:")
    print("- 0% win rate (0 winning trades out of 310)")
    print("- Lost 87.5% of capital (-$8,748)")
    print("- 66% of exits via stop loss (206/310 trades)")
    print("- 34% of exits via z-reversion (104/310 trades)")
    print()
    print("LIKELY CAUSES:")
    print("1. Entry signals are triggering on TEMPORARY z-score spikes")
    print("2. Markets mean-revert BEFORE we can capture edge")
    print("3. Stop loss at -14% is getting hit consistently")
    print("4. High entry z-scores (1.0-2.7) may be too aggressive")
    print()
    print("RECOMMENDATIONS:")
    print("- WAIT for more training data (need 1-2 weeks minimum)")
    print("- Meta-learner weights are all 0.0 (insufficient variance)")
    print("- Current backtest is on SAME data used for training")
    print("- Need out-of-sample testing with fresh data")
    print()
elif stats['win_rate'] < 0.40:
    print("⚠️ WEAK EDGE - NOT TRADEABLE")
    print(f"Win rate ({stats['win_rate']*100:.1f}%) too low")
    print(f"Profit factor ({stats['profit_factor']:.2f}) below 1.5 threshold")
elif stats['total_return'] < 0:
    print("❌ NEGATIVE EDGE - LOSING MONEY")
    print(f"Total return: {stats['total_return']*100:.2f}%")
else:
    print("✅ POTENTIAL EDGE DETECTED")
    print(f"Win rate: {stats['win_rate']*100:.1f}%")
    print(f"Profit factor: {stats['profit_factor']:.2f}")
    print(f"Return: {stats['total_return']*100:.2f}%")

print()
print("=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("1. Let VM collect data for 7-14 days")
print("2. Retrain meta-learner with new data")
print("3. Run out-of-sample backtest on fresh data")
print("4. Analyze feature importance (which signals matter)")
print("5. Consider adjusting entry thresholds (z > 2.0 instead of 1.5)")
print("=" * 80)
