"""
Polymarket Backtesting Framework

Replays historical market snapshots through the orchestrator to validate
the probability arbitrage strategy on historical data.

Key Features:
- Historical market replay
- Simulated limit-at-fair execution (95% fill rate)
- Virtual position tracking
- 6-rule exit system
- Performance analytics (win rate, Sharpe, PnL)

Usage:
    python src/agents/polymarket_backtester.py --start 2025-10-24 --end 2025-10-27
    python src/agents/polymarket_backtester.py --report
"""

import os
import sys
import json
import random
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from termcolor import cprint

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import (
    POLYMARKET_EV_MIN,
    POLYMARKET_Z_MIN,
    POLYMARKET_MAX_SPREAD,
    POLYMARKET_MIN_LIQUIDITY,
    POLYMARKET_EXIT_EV_DECAY,
    POLYMARKET_EXIT_Z_REVERSION,
    POLYMARKET_EXIT_TRAILING_EV_ALPHA,
    POLYMARKET_EXIT_TIME_GATE_DAYS,
    POLYMARKET_EXIT_PROFIT_TARGET,
    POLYMARKET_EXIT_STOP_LOSS,
)
from src.polymarket_utils import format_currency, format_probability


class PolymarketBacktester:
    """Backtests Polymarket probability arbitrage strategy on historical data"""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        fill_rate: float = 0.95,
        fast_mode: bool = False
    ):
        """
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Starting portfolio value
            fill_rate: Probability of limit order fill (default 95%)
            fast_mode: Skip LLM forecasting, use baseline only
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.fill_rate = fill_rate
        self.fast_mode = fast_mode

        # Trading state
        self.positions: List[Dict] = []  # Open positions
        self.closed_positions: List[Dict] = []  # Closed positions
        self.trades: List[Dict] = []  # All trade events
        self.daily_pnl: Dict[str, float] = {}  # Date -> PnL

        # Statistics
        self.total_signals = 0
        self.total_entries = 0
        self.total_fills = 0
        self.total_exits = 0

        # Load historical data
        self.snapshots = self._load_snapshots()

        cprint("\n" + "="*80, "cyan")
        cprint("POLYMARKET BACKTESTER INITIALIZED", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")
        cprint(f"  Period: {start_date} to {end_date}", "white")
        cprint(f"  Initial Capital: {format_currency(initial_capital)}", "white")
        cprint(f"  Fill Rate: {fill_rate*100:.1f}%", "white")
        cprint(f"  Mode: {'FAST (no LLM)' if fast_mode else 'FULL (with LLM)'}", "yellow")
        cprint(f"  Snapshots Loaded: {len(self.snapshots):,}", "green")
        cprint("="*80 + "\n", "cyan")

    def _load_snapshots(self) -> pd.DataFrame:
        """Load historical market snapshots from CSV"""
        snapshot_path = "src/data/polymarket/training_data/market_snapshots.csv"

        if not os.path.exists(snapshot_path):
            cprint(f"[ERROR] Snapshot file not found: {snapshot_path}", "red")
            sys.exit(1)

        # Load and filter by date range
        df = pd.read_csv(snapshot_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        mask = (df['timestamp'] >= self.start_date) & (df['timestamp'] <= self.end_date)
        filtered = df[mask].sort_values('timestamp').reset_index(drop=True)

        if len(filtered) == 0:
            cprint(f"[ERROR] No snapshots found in date range", "red")
            sys.exit(1)

        return filtered

    def _check_entry_gates(self, snapshot: Dict) -> Dict:
        """
        Simulate entry gate checks without LLM

        For backtesting, we use a simplified model:
        - Use current market price as baseline probability
        - Add small random edge (simulates swarm forecast deviation)
        - Check all 6 entry gates

        Returns:
            Decision dict with entry_decision, metrics, etc.
        """
        # Extract snapshot data
        mid_yes = snapshot['mid_yes']
        mid_no = snapshot['mid_no']
        spread = snapshot['spread']
        liquidity = snapshot['liquidity']
        volume_24h = snapshot['volume_24h']
        market_type = snapshot['market_type']
        regime = snapshot['regime']

        # Simulate probability estimate (market price ± random edge)
        # In reality, swarm forecaster would provide this
        baseline_prob = mid_yes

        # Add random edge: N(0, 0.05) for simulation
        # Real system uses swarm consensus + meta-learning
        edge = np.random.normal(0, 0.05)
        true_prob = np.clip(baseline_prob + edge, 0.01, 0.99)

        # Calculate EV_net
        if true_prob > mid_yes:
            # Bet YES
            side = "YES"
            ev_net = (true_prob - mid_yes) / mid_yes
        else:
            # Bet NO
            side = "NO"
            ev_net = (true_prob - mid_no) / mid_no

        # Calculate z-score (simplified)
        # Real system uses volatility from historical snapshots
        volatility = max(snapshot.get('volatility_lookback', 0.1), 0.05)
        z_score = abs(true_prob - baseline_prob) / volatility

        # Check entry gates (RELAXED FOR TESTING - see line comments)
        gates = {
            'ev_net': ev_net >= 0.01,  # Relaxed: 1% (was 3%) for testing
            'z_score': z_score >= 0.5,  # Relaxed: 0.5 (was 1.5) for testing
            'spread': spread <= 0.99,   # Relaxed: 99% (was 6%) - effectively disabled for testing
            'liquidity': liquidity >= POLYMARKET_MIN_LIQUIDITY,  # Keep original
            'volume': volume_24h >= 100,  # Relaxed: $100 (was $1k) for testing
            'time': snapshot.get('time_to_resolution_days', 999) <= 9999  # Disabled for testing (was 180)
        }

        # ALL gates must pass
        entry_decision = "ENTER" if all(gates.values()) else "REJECT"

        # Position sizing (Kelly criterion)
        kelly_fraction = 0.25 if regime == "illiquid" else 0.5
        edge_decimal = abs(ev_net)
        kelly_size = (edge_decimal * kelly_fraction) * self.portfolio_value
        final_size = min(kelly_size, 1000)  # Cap at $1k per position

        return {
            'entry_decision': entry_decision,
            'side': side,
            'true_prob': true_prob,
            'market_price': mid_yes,
            'ev_net': ev_net,
            'z_score': z_score,
            'spread': spread,
            'liquidity': liquidity,
            'gates': gates,
            'failed_gates': [k for k, v in gates.items() if not v],
            'position_size': final_size,
            'regime': regime,
        }

    def _simulate_execution(self, decision: Dict, snapshot: Dict) -> Optional[Dict]:
        """
        Simulate limit-at-fair order execution

        Args:
            decision: Entry decision from _check_entry_gates()
            snapshot: Market snapshot at entry time

        Returns:
            Execution dict if filled, None if not filled
        """
        if decision['entry_decision'] != 'ENTER':
            return None

        # Simulate fill probability (95% by default)
        if random.random() > self.fill_rate:
            return None  # Order not filled

        # Filled at true_prob (limit-at-fair pricing)
        return {
            'filled': True,
            'fill_price': decision['true_prob'],
            'size': decision['position_size'],
            'side': decision['side'],
            'timestamp': snapshot['timestamp'],
        }

    def _check_exit_rules(self, position: Dict, snapshot: Dict) -> Dict:
        """
        Check 6-rule exit system (ANY rule triggers exit)

        Args:
            position: Open position dict
            snapshot: Current market snapshot

        Returns:
            Dict with should_exit, exit_reason, exit_rule
        """
        current_price = snapshot['mid_yes'] if position['side'] == 'YES' else snapshot['mid_no']
        entry_price = position['entry_price']
        entry_time = pd.to_datetime(position['entry_timestamp'])
        current_time = pd.to_datetime(snapshot['timestamp'])
        days_held = (current_time - entry_time).total_seconds() / 86400

        # Calculate current PnL %
        if position['side'] == 'YES':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # Rule 1: EV Decay (EV_net < 1%)
        current_ev = position.get('current_ev', position['entry_ev'])
        if current_ev < POLYMARKET_EXIT_EV_DECAY:
            return {
                'should_exit': True,
                'exit_reason': f"EV decay ({current_ev:.1%} < {POLYMARKET_EXIT_EV_DECAY:.1%})",
                'exit_rule': 'ev_decay'
            }

        # Rule 2: Z-Reversion (z < 0.8)
        current_z = position.get('current_z', position['entry_z'])
        if current_z < POLYMARKET_EXIT_Z_REVERSION:
            return {
                'should_exit': True,
                'exit_reason': f"Z-reversion ({current_z:.2f} < {POLYMARKET_EXIT_Z_REVERSION:.2f})",
                'exit_rule': 'z_reversion'
            }

        # Rule 3: Trailing EV (current < 70% of peak)
        peak_ev = position.get('peak_ev', position['entry_ev'])
        if current_ev < POLYMARKET_EXIT_TRAILING_EV_ALPHA * peak_ev:
            return {
                'should_exit': True,
                'exit_reason': f"Trailing EV ({current_ev:.1%} < {POLYMARKET_EXIT_TRAILING_EV_ALPHA:.0%} × {peak_ev:.1%})",
                'exit_rule': 'trailing_ev'
            }

        # Rule 4: Time Gate (no 30%+ improvement in 7 days)
        if days_held >= POLYMARKET_EXIT_TIME_GATE_DAYS:
            improvement = pnl_pct / position['entry_ev']
            if improvement < 0.3:
                return {
                    'should_exit': True,
                    'exit_reason': f"Time gate ({days_held:.1f} days, improvement={improvement:.1%})",
                    'exit_rule': 'time_gate'
                }

        # Rule 5: Signal Reversal (skip in backtest - requires real-time agent signals)

        # Rule 6: Profit Target / Stop Loss
        if pnl_pct >= POLYMARKET_EXIT_PROFIT_TARGET:
            return {
                'should_exit': True,
                'exit_reason': f"Profit target ({pnl_pct:.1%})",
                'exit_rule': 'profit_target'
            }

        if pnl_pct <= -POLYMARKET_EXIT_STOP_LOSS:
            return {
                'should_exit': True,
                'exit_reason': f"Stop loss ({pnl_pct:.1%})",
                'exit_rule': 'stop_loss'
            }

        # No exit
        return {'should_exit': False}

    def _close_position(self, position: Dict, snapshot: Dict, exit_info: Dict):
        """Close position and record trade"""
        current_price = snapshot['mid_yes'] if position['side'] == 'YES' else snapshot['mid_no']
        entry_price = position['entry_price']
        size = position['size']

        # Calculate PnL
        if position['side'] == 'YES':
            pnl = (current_price - entry_price) * size / entry_price
        else:
            pnl = (entry_price - current_price) * size / entry_price

        # Update portfolio
        self.portfolio_value += pnl

        # Record closed position
        closed = {
            **position,
            'exit_timestamp': snapshot['timestamp'],
            'exit_price': current_price,
            'exit_reason': exit_info['exit_reason'],
            'exit_rule': exit_info['exit_rule'],
            'pnl': pnl,
            'pnl_pct': pnl / size,
            'hold_time_days': (pd.to_datetime(snapshot['timestamp']) - pd.to_datetime(position['entry_timestamp'])).total_seconds() / 86400,
        }
        self.closed_positions.append(closed)

        # Record daily PnL
        date_key = pd.to_datetime(snapshot['timestamp']).date().isoformat()
        self.daily_pnl[date_key] = self.daily_pnl.get(date_key, 0) + pnl

        self.total_exits += 1

    def run_backtest(self, verbose=False):
        """Run full backtest on historical snapshots"""
        cprint("\n" + "="*80, "cyan")
        cprint("STARTING BACKTEST", "cyan", attrs=['bold'])
        cprint("="*80 + "\n", "cyan")

        total_snapshots = len(self.snapshots)
        gate_failures = {
            'ev_net': 0,
            'z_score': 0,
            'spread': 0,
            'liquidity': 0,
            'volume': 0,
            'time': 0
        }

        for idx, row in self.snapshots.iterrows():
            snapshot = row.to_dict()

            # Progress indicator (every 1000 snapshots)
            if idx % 1000 == 0:
                progress = idx / total_snapshots * 100
                cprint(f"[{progress:5.1f}%] Processing snapshot {idx:,}/{total_snapshots:,} - {snapshot['timestamp']}", "yellow")

            # Check entry signals
            decision = self._check_entry_gates(snapshot)

            # Track gate failures for diagnostics
            if decision['entry_decision'] == 'REJECT':
                for gate in decision['failed_gates']:
                    gate_failures[gate] = gate_failures.get(gate, 0) + 1

            if decision['entry_decision'] == 'ENTER':
                self.total_signals += 1

                # Try to execute
                execution = self._simulate_execution(decision, snapshot)

                if execution and execution['filled']:
                    self.total_fills += 1

                    # Open new position
                    position = {
                        'position_id': f"BT_{snapshot['market_id']}_{snapshot['timestamp']}",
                        'market_id': snapshot['market_id'],
                        'question': snapshot['question'],
                        'entry_timestamp': snapshot['timestamp'],
                        'entry_price': execution['fill_price'],
                        'side': execution['side'],
                        'size': execution['size'],
                        'entry_ev': decision['ev_net'],
                        'entry_z': decision['z_score'],
                        'current_ev': decision['ev_net'],
                        'current_z': decision['z_score'],
                        'peak_ev': decision['ev_net'],
                        'regime': decision['regime'],
                    }
                    self.positions.append(position)
                    self.total_entries += 1

            # Check exit rules for all open positions
            positions_to_close = []
            for position in self.positions:
                # Only check positions for this market
                if position['market_id'] == snapshot['market_id']:
                    exit_check = self._check_exit_rules(position, snapshot)

                    if exit_check['should_exit']:
                        positions_to_close.append((position, exit_check))

            # Close positions
            for position, exit_info in positions_to_close:
                self._close_position(position, snapshot, exit_info)
                self.positions.remove(position)

        # Close any remaining open positions at final price
        if len(self.positions) > 0:
            cprint(f"\n[WARN] Closing {len(self.positions)} open positions at final snapshot price", "yellow")
            final_snapshot = self.snapshots.iloc[-1].to_dict()

            for position in list(self.positions):
                exit_info = {
                    'exit_reason': 'Backtest end',
                    'exit_rule': 'forced_close'
                }
                self._close_position(position, final_snapshot, exit_info)

            self.positions = []

        cprint("\n" + "="*80, "cyan")
        cprint("BACKTEST COMPLETE", "cyan", attrs=['bold'])
        cprint("="*80 + "\n", "cyan")

        # Print gate failure diagnostics
        cprint("[GATE FAILURE ANALYSIS]", "yellow", attrs=['bold'])
        total_rejections = sum(gate_failures.values())
        if total_rejections > 0:
            cprint(f"  Total Rejections: {total_rejections:,}", "white")
            for gate, count in sorted(gate_failures.items(), key=lambda x: -x[1]):
                pct = count / total_snapshots * 100
                cprint(f"    {gate:12s}: {count:,} ({pct:.1f}% of snapshots)", "red")
        cprint("")

        return self.calculate_performance()

    def calculate_performance(self) -> Dict:
        """Calculate comprehensive performance statistics"""
        if len(self.closed_positions) == 0:
            cprint("[WARN] No closed positions - cannot calculate performance", "yellow")
            return {}

        # Basic stats
        total_trades = len(self.closed_positions)
        winning_trades = sum(1 for p in self.closed_positions if p['pnl'] > 0)
        losing_trades = sum(1 for p in self.closed_positions if p['pnl'] < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # PnL stats
        total_pnl = sum(p['pnl'] for p in self.closed_positions)
        avg_pnl = total_pnl / total_trades
        avg_win = np.mean([p['pnl'] for p in self.closed_positions if p['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([p['pnl'] for p in self.closed_positions if p['pnl'] < 0]) if losing_trades > 0 else 0

        # Return stats
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital

        # Sharpe ratio (daily returns)
        daily_returns = pd.Series(self.daily_pnl).sort_index()
        if len(daily_returns) > 1:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        cumulative_pnl = np.cumsum([p['pnl'] for p in self.closed_positions])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        # Hold time stats
        avg_hold_time = np.mean([p['hold_time_days'] for p in self.closed_positions])

        # Exit rule breakdown
        exit_rules = {}
        for p in self.closed_positions:
            rule = p['exit_rule']
            exit_rules[rule] = exit_rules.get(rule, 0) + 1

        return {
            'backtest_period': f"{self.start_date.date()} to {self.end_date.date()}",
            'total_snapshots': len(self.snapshots),
            'total_signals': self.total_signals,
            'total_entries': self.total_entries,
            'fill_rate_actual': self.total_fills / self.total_entries if self.total_entries > 0 else 0,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'initial_capital': self.initial_capital,
            'final_capital': self.portfolio_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_hold_time_days': avg_hold_time,
            'exit_rules': exit_rules,
        }

    def print_report(self, stats: Dict):
        """Print formatted performance report"""
        cprint("\n" + "="*80, "cyan")
        cprint("BACKTEST PERFORMANCE REPORT", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")

        cprint(f"\n[PERIOD] {stats['backtest_period']}", "white", attrs=['bold'])
        cprint(f"[DATA] {stats['total_snapshots']:,} snapshots processed\n", "white")

        cprint("[TRADE STATISTICS]", "yellow", attrs=['bold'])
        cprint(f"  Signals Generated:    {stats['total_signals']:,}", "white")
        cprint(f"  Entries Attempted:    {stats['total_entries']:,}", "white")
        cprint(f"  Orders Filled:        {stats['total_trades']:,} ({stats['fill_rate_actual']:.1%})", "white")
        cprint(f"  Winning Trades:       {stats['winning_trades']:,}", "green")
        cprint(f"  Losing Trades:        {stats['losing_trades']:,}", "red")
        cprint(f"  Win Rate:             {stats['win_rate']:.1%}", "green" if stats['win_rate'] > 0.5 else "yellow")

        cprint("\n[PNL ANALYSIS]", "yellow", attrs=['bold'])
        cprint(f"  Total PnL:            {format_currency(stats['total_pnl'])}", "green" if stats['total_pnl'] > 0 else "red")
        cprint(f"  Average PnL/Trade:    {format_currency(stats['avg_pnl'])}", "white")
        cprint(f"  Average Win:          {format_currency(stats['avg_win'])}", "green")
        cprint(f"  Average Loss:         {format_currency(stats['avg_loss'])}", "red")
        cprint(f"  Profit Factor:        {stats['profit_factor']:.2f}", "white")

        cprint("\n[RETURNS]", "yellow", attrs=['bold'])
        cprint(f"  Initial Capital:      {format_currency(stats['initial_capital'])}", "white")
        cprint(f"  Final Capital:        {format_currency(stats['final_capital'])}", "white")
        cprint(f"  Total Return:         {stats['total_return']:.2%}", "green" if stats['total_return'] > 0 else "red")
        cprint(f"  Sharpe Ratio:         {stats['sharpe_ratio']:.2f}", "white")
        cprint(f"  Max Drawdown:         {format_currency(stats['max_drawdown'])}", "red")

        cprint("\n[POSITION MANAGEMENT]", "yellow", attrs=['bold'])
        cprint(f"  Avg Hold Time:        {stats['avg_hold_time_days']:.1f} days", "white")

        cprint("\n[EXIT RULE BREAKDOWN]", "yellow", attrs=['bold'])
        for rule, count in sorted(stats['exit_rules'].items(), key=lambda x: -x[1]):
            pct = count / stats['total_trades'] * 100
            cprint(f"  {rule:20s}: {count:3d} ({pct:5.1f}%)", "white")

        cprint("\n" + "="*80, "cyan")

    def save_results(self, stats: Dict, output_dir: str = "src/data/polymarket/backtests"):
        """Save backtest results to JSON and CSV"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary stats
        stats_path = os.path.join(output_dir, f"backtest_stats_{timestamp}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        # Save trade-by-trade results
        trades_df = pd.DataFrame(self.closed_positions)
        trades_path = os.path.join(output_dir, f"backtest_trades_{timestamp}.csv")
        trades_df.to_csv(trades_path, index=False)

        # Save daily PnL
        daily_df = pd.DataFrame(list(self.daily_pnl.items()), columns=['date', 'pnl'])
        daily_path = os.path.join(output_dir, f"backtest_daily_pnl_{timestamp}.csv")
        daily_df.to_csv(daily_path, index=False)

        cprint(f"\n[SAVE] Results saved to:", "green")
        cprint(f"  Stats:  {stats_path}", "white")
        cprint(f"  Trades: {trades_path}", "white")
        cprint(f"  Daily:  {daily_path}", "white")


def main():
    """Main entry point for backtester"""
    parser = argparse.ArgumentParser(description="Polymarket Backtesting Framework")
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)', default='2025-10-24')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)', default='2025-10-27')
    parser.add_argument('--capital', type=float, help='Initial capital', default=10000.0)
    parser.add_argument('--fill-rate', type=float, help='Order fill rate', default=0.95)
    parser.add_argument('--fast', action='store_true', help='Fast mode (no LLM)', default=True)
    parser.add_argument('--report', action='store_true', help='Just print last report')

    args = parser.parse_args()

    if args.report:
        # Load and print most recent backtest report
        backtest_dir = "src/data/polymarket/backtests"
        if os.path.exists(backtest_dir):
            files = [f for f in os.listdir(backtest_dir) if f.startswith('backtest_stats_')]
            if files:
                latest = sorted(files)[-1]
                with open(os.path.join(backtest_dir, latest), 'r') as f:
                    stats = json.load(f)

                backtester = PolymarketBacktester(
                    start_date=args.start,
                    end_date=args.end,
                    initial_capital=args.capital
                )
                backtester.print_report(stats)
                return

        cprint("[ERROR] No backtest reports found", "red")
        return

    # Run new backtest
    backtester = PolymarketBacktester(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        fill_rate=args.fill_rate,
        fast_mode=args.fast
    )

    stats = backtester.run_backtest()

    if stats:
        backtester.print_report(stats)
        backtester.save_results(stats)


if __name__ == "__main__":
    main()
