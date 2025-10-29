"""
[EMOJI] Moon Dev's Polymarket Exit Manager
6-rule exit system for disciplined position management
Built with love by Moon Dev [EMOJI]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from termcolor import cprint

# Add project root to path
import sys
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from src.polymarket_utils import PolymarketUtils, format_currency, format_probability
from src.config import (
    POLYMARKET_EXIT_EV_DECAY,
    POLYMARKET_EXIT_Z_REVERSION,
    POLYMARKET_EXIT_TRAILING_EV_ALPHA,
    POLYMARKET_EXIT_TIME_GATE_DAYS,
    POLYMARKET_EXIT_SIGNAL_REVERSAL,
    POLYMARKET_EXIT_PROFIT_TARGET,
    POLYMARKET_EXIT_STOP_LOSS,
    POLYMARKET_DATA_DIR,
    POLYMARKET_VERBOSE_LOGGING
)


class PolymarketExitManager:
    """
    Polymarket Exit Manager - 6-Rule Exit System

    Monitors open positions and triggers exits when ANY rule fires.

    Exit Rules (ANY triggers exit):
    1. EV Decay: EV_net < 1%
    2. Z-Score Reversion: z < 0.8
    3. Trailing EV: current_EV < 0.7 x peak_EV
    4. Time Gate: No 30%+ EV improvement in 7 days
    5. Signal Reversal: >=3 agents flip bearish
    6. Profit/Stop: +/-8% / -3%

    Philosophy: Exit on convergence, not resolution
    """

    def __init__(self):
        """Initialize Exit Manager"""
        self.data_dir = Path(POLYMARKET_DATA_DIR) / 'positions'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.utils = PolymarketUtils()

        # Data files
        self.positions_file = self.data_dir / 'open_positions.csv'
        self.closed_positions_file = self.data_dir / 'closed_positions.csv'

        # Load positions
        self.open_positions = self._load_or_create_positions()
        self.closed_positions = self._load_or_create_closed_positions()

        cprint(f"\n[EMOJI] Polymarket Exit Manager Initialized", "cyan", attrs=['bold'])
        cprint(f"[EMOJI] Open Positions: {len(self.open_positions)}", "green")
        cprint(f"[EMOJI] Closed Positions: {len(self.closed_positions)}", "cyan")
        cprint(f"\n[SYM]  EXIT RULES:", "yellow")
        cprint(f"   1. EV Decay: EV_net < {POLYMARKET_EXIT_EV_DECAY:.2%}", "cyan")
        cprint(f"   2. Z-Reversion: z < {POLYMARKET_EXIT_Z_REVERSION}", "cyan")
        cprint(f"   3. Trailing EV: current < {POLYMARKET_EXIT_TRAILING_EV_ALPHA:.0%} x peak", "cyan")
        cprint(f"   4. Time Gate: No 30%+ improvement in {POLYMARKET_EXIT_TIME_GATE_DAYS} days", "cyan")
        cprint(f"   5. Signal Reversal: >=3 agents flip", "cyan")
        cprint(f"   6. Profit/Stop: +{POLYMARKET_EXIT_PROFIT_TARGET:.0%} / {POLYMARKET_EXIT_STOP_LOSS:.0%}", "cyan")

    def _load_or_create_positions(self) -> pd.DataFrame:
        """Load or create open positions database"""
        if self.positions_file.exists():
            df = pd.read_csv(self.positions_file)
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['last_check_time'] = pd.to_datetime(df['last_check_time'])
            cprint(f"[SYM] Loaded {len(df)} open positions", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'position_id',
                'market_id',
                'question',
                'entry_time',
                'side',  # YES or NO
                'position_size',
                'entry_price',
                'current_price',
                'entry_ev_net',
                'current_ev_net',
                'peak_ev_net',
                'entry_z_score',
                'current_z_score',
                'last_check_time',
                'last_ev_improvement_time',
                'pnl_pct',
                'bullish_signals_count',  # For signal reversal
                'bearish_signals_count'
            ])
            cprint(f"[EMOJI] Created new open positions database", "yellow")
            return df

    def _load_or_create_closed_positions(self) -> pd.DataFrame:
        """Load or create closed positions database"""
        if self.closed_positions_file.exists():
            df = pd.read_csv(self.closed_positions_file)
            cprint(f"[SYM] Loaded {len(df)} closed positions", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'position_id',
                'market_id',
                'question',
                'entry_time',
                'exit_time',
                'hold_time_hours',
                'side',
                'position_size',
                'entry_price',
                'exit_price',
                'entry_ev_net',
                'exit_ev_net',
                'peak_ev_net',
                'entry_z_score',
                'exit_z_score',
                'pnl_pct',
                'pnl_usd',
                'exit_rule_triggered',
                'exit_reason'
            ])
            cprint(f"[EMOJI] Created new closed positions database", "yellow")
            return df

    def open_position(
        self,
        market_id: str,
        question: str,
        side: str,
        position_size: float,
        entry_price: float,
        entry_ev_net: float,
        entry_z_score: float
    ) -> str:
        """
        Open new position

        Args:
            market_id: Market ID
            question: Market question
            side: 'YES' or 'NO'
            position_size: Position size in USD
            entry_price: Entry price (0-1)
            entry_ev_net: EV_net at entry
            entry_z_score: Z-score at entry

        Returns:
            position_id
        """
        position_id = f"{market_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        cprint(f"\n[EMOJI] OPENING POSITION", "green", attrs=['bold'])
        cprint(f"ID: {position_id}", "cyan")
        cprint(f"Market: {question[:50]}...", "white")
        cprint(f"Side: {side} @ {format_probability(entry_price)}", "yellow")
        cprint(f"Size: {format_currency(position_size)}", "green")
        cprint(f"Entry EV: {entry_ev_net:.3f}", "cyan")
        cprint(f"Entry Z: {entry_z_score:.2f}sigma", "cyan")

        position_data = {
            'position_id': position_id,
            'market_id': market_id,
            'question': question,
            'entry_time': datetime.now(),
            'side': side,
            'position_size': position_size,
            'entry_price': entry_price,
            'current_price': entry_price,
            'entry_ev_net': entry_ev_net,
            'current_ev_net': entry_ev_net,
            'peak_ev_net': entry_ev_net,
            'entry_z_score': entry_z_score,
            'current_z_score': entry_z_score,
            'last_check_time': datetime.now(),
            'last_ev_improvement_time': datetime.now(),
            'pnl_pct': 0.0,
            'bullish_signals_count': 0,
            'bearish_signals_count': 0
        }

        self.open_positions = pd.concat([
            self.open_positions,
            pd.DataFrame([position_data])
        ], ignore_index=True)

        self._save_positions()

        cprint(f"[SYM] Position opened successfully", "green")

        return position_id

    def check_position(
        self,
        position_id: str,
        current_price: float,
        current_ev_net: float,
        current_z_score: float,
        bullish_signals_count: int = 0,
        bearish_signals_count: int = 0
    ) -> Dict:
        """
        Check position against all exit rules

        Args:
            position_id: Position ID
            current_price: Current market price
            current_ev_net: Current EV_net
            current_z_score: Current z-score
            bullish_signals_count: Number of bullish signals
            bearish_signals_count: Number of bearish signals

        Returns:
            {
                'should_exit': bool,
                'exit_rules_triggered': list,
                'position_update': dict
            }
        """
        # Get position
        pos_mask = self.open_positions['position_id'] == position_id

        if not pos_mask.any():
            cprint(f"[SYM] Position {position_id} not found", "red")
            return {'should_exit': False, 'exit_rules_triggered': [], 'position_update': {}}

        position = self.open_positions[pos_mask].iloc[0]

        if POLYMARKET_VERBOSE_LOGGING:
            cprint(f"\n[EMOJI] Checking position: {position_id}", "cyan")

        # Calculate PnL
        if position['side'] == 'YES':
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        else:  # NO
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

        # Update peak EV
        peak_ev_net = max(position['peak_ev_net'], current_ev_net)

        # Check if EV improved significantly
        ev_improvement = (current_ev_net - position['entry_ev_net']) / abs(position['entry_ev_net']) if position['entry_ev_net'] != 0 else 0
        last_ev_improvement_time = position['last_ev_improvement_time']

        if ev_improvement >= 0.30:  # 30%+ improvement
            last_ev_improvement_time = datetime.now()

        # Update position data
        self.open_positions.loc[pos_mask, 'current_price'] = current_price
        self.open_positions.loc[pos_mask, 'current_ev_net'] = current_ev_net
        self.open_positions.loc[pos_mask, 'peak_ev_net'] = peak_ev_net
        self.open_positions.loc[pos_mask, 'current_z_score'] = current_z_score
        self.open_positions.loc[pos_mask, 'last_check_time'] = datetime.now()
        self.open_positions.loc[pos_mask, 'last_ev_improvement_time'] = last_ev_improvement_time
        self.open_positions.loc[pos_mask, 'pnl_pct'] = pnl_pct
        self.open_positions.loc[pos_mask, 'bullish_signals_count'] = bullish_signals_count
        self.open_positions.loc[pos_mask, 'bearish_signals_count'] = bearish_signals_count

        # Check exit rules
        exit_rules_triggered = []

        # Rule 1: EV Decay
        if current_ev_net < POLYMARKET_EXIT_EV_DECAY:
            exit_rules_triggered.append('EV_DECAY')
            if POLYMARKET_VERBOSE_LOGGING:
                cprint(f"   [SYM] Rule 1: EV Decay ({current_ev_net:.3f} < {POLYMARKET_EXIT_EV_DECAY:.3f})", "red")

        # Rule 2: Z-Score Reversion
        if current_z_score < POLYMARKET_EXIT_Z_REVERSION:
            exit_rules_triggered.append('Z_REVERSION')
            if POLYMARKET_VERBOSE_LOGGING:
                cprint(f"   [SYM] Rule 2: Z-Reversion ({current_z_score:.2f} < {POLYMARKET_EXIT_Z_REVERSION})", "red")

        # Rule 3: Trailing EV
        trailing_ev_threshold = POLYMARKET_EXIT_TRAILING_EV_ALPHA * peak_ev_net
        if current_ev_net < trailing_ev_threshold:
            exit_rules_triggered.append('TRAILING_EV')
            if POLYMARKET_VERBOSE_LOGGING:
                cprint(f"   [SYM] Rule 3: Trailing EV ({current_ev_net:.3f} < {trailing_ev_threshold:.3f})", "red")

        # Rule 4: Time Gate
        last_improvement = pd.to_datetime(last_ev_improvement_time)
        days_since_improvement = (datetime.now() - last_improvement).total_seconds() / 86400

        if days_since_improvement > POLYMARKET_EXIT_TIME_GATE_DAYS:
            exit_rules_triggered.append('TIME_GATE')
            if POLYMARKET_VERBOSE_LOGGING:
                cprint(f"   [SYM] Rule 4: Time Gate ({days_since_improvement:.1f} days since improvement)", "red")

        # Rule 5: Signal Reversal
        if POLYMARKET_EXIT_SIGNAL_REVERSAL:
            # Check if originally bullish (YES position with positive entry EV)
            # or bearish (NO position)
            originally_bullish = (position['side'] == 'YES' and position['entry_ev_net'] > 0) or \
                                 (position['side'] == 'NO' and position['entry_ev_net'] > 0)

            if originally_bullish:
                # Exit if >=3 bearish signals
                if bearish_signals_count >= 3:
                    exit_rules_triggered.append('SIGNAL_REVERSAL')
                    if POLYMARKET_VERBOSE_LOGGING:
                        cprint(f"   [SYM] Rule 5: Signal Reversal (>=3 bearish signals)", "red")
            else:
                # Exit if >=3 bullish signals (reversal from bearish)
                if bullish_signals_count >= 3:
                    exit_rules_triggered.append('SIGNAL_REVERSAL')
                    if POLYMARKET_VERBOSE_LOGGING:
                        cprint(f"   [SYM] Rule 5: Signal Reversal (>=3 bullish signals)", "red")

        # Rule 6: Profit Target / Stop Loss
        if pnl_pct >= POLYMARKET_EXIT_PROFIT_TARGET:
            exit_rules_triggered.append('PROFIT_TARGET')
            if POLYMARKET_VERBOSE_LOGGING:
                cprint(f"   [SYM] Rule 6: Profit Target ({pnl_pct:.1%} >= {POLYMARKET_EXIT_PROFIT_TARGET:.1%})", "green")

        if pnl_pct <= POLYMARKET_EXIT_STOP_LOSS:
            exit_rules_triggered.append('STOP_LOSS')
            if POLYMARKET_VERBOSE_LOGGING:
                cprint(f"   [SYM] Rule 6: Stop Loss ({pnl_pct:.1%} <= {POLYMARKET_EXIT_STOP_LOSS:.1%})", "red")

        # Determine if should exit
        should_exit = len(exit_rules_triggered) > 0

        if should_exit:
            cprint(f"\n[EMOJI] EXIT TRIGGERED: {', '.join(exit_rules_triggered)}", "red", attrs=['bold'])

        # Save updates
        self._save_positions()

        return {
            'should_exit': should_exit,
            'exit_rules_triggered': exit_rules_triggered,
            'position_update': {
                'current_price': current_price,
                'current_ev_net': current_ev_net,
                'current_z_score': current_z_score,
                'pnl_pct': pnl_pct,
                'peak_ev_net': peak_ev_net
            }
        }

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_rules_triggered: List[str]
    ) -> Dict:
        """
        Close position

        Args:
            position_id: Position ID
            exit_price: Exit price
            exit_rules_triggered: List of exit rules that triggered

        Returns:
            Closed position dict
        """
        # Get position
        pos_mask = self.open_positions['position_id'] == position_id

        if not pos_mask.any():
            cprint(f"[SYM] Position {position_id} not found", "red")
            return {}

        position = self.open_positions[pos_mask].iloc[0]

        # Calculate final PnL
        if position['side'] == 'YES':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

        pnl_usd = pnl_pct * position['position_size']

        # Calculate hold time
        hold_time_hours = (datetime.now() - position['entry_time']).total_seconds() / 3600

        cprint(f"\n[EMOJI] CLOSING POSITION", "red", attrs=['bold'])
        cprint(f"ID: {position_id}", "cyan")
        cprint(f"Market: {position['question'][:50]}...", "white")
        cprint(f"Entry: {format_probability(position['entry_price'])} -> Exit: {format_probability(exit_price)}", "yellow")
        cprint(f"Hold Time: {hold_time_hours:.1f} hours", "cyan")
        cprint(f"PnL: {pnl_pct:+.2%} ({format_currency(pnl_usd):+})", "green" if pnl_pct > 0 else "red", attrs=['bold'])
        cprint(f"Exit Rules: {', '.join(exit_rules_triggered)}", "yellow")

        # Build closed position record
        closed_position = {
            'position_id': position_id,
            'market_id': position['market_id'],
            'question': position['question'],
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'hold_time_hours': hold_time_hours,
            'side': position['side'],
            'position_size': position['position_size'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_ev_net': position['entry_ev_net'],
            'exit_ev_net': position['current_ev_net'],
            'peak_ev_net': position['peak_ev_net'],
            'entry_z_score': position['entry_z_score'],
            'exit_z_score': position['current_z_score'],
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'exit_rule_triggered': ','.join(exit_rules_triggered),
            'exit_reason': self._format_exit_reason(exit_rules_triggered, position, exit_price)
        }

        # Add to closed positions
        self.closed_positions = pd.concat([
            self.closed_positions,
            pd.DataFrame([closed_position])
        ], ignore_index=True)

        # Remove from open positions
        self.open_positions = self.open_positions[~pos_mask].copy()

        # Save
        self._save_positions()
        self._save_closed_positions()

        cprint(f"[SYM] Position closed successfully", "green")

        return closed_position

    def _format_exit_reason(
        self,
        exit_rules: List[str],
        position: pd.Series,
        exit_price: float
    ) -> str:
        """Format detailed exit reason"""

        reasons = []

        if 'EV_DECAY' in exit_rules:
            reasons.append(f"EV decayed to {position['current_ev_net']:.3f}")

        if 'Z_REVERSION' in exit_rules:
            reasons.append(f"Z-score reverted to {position['current_z_score']:.2f}")

        if 'TRAILING_EV' in exit_rules:
            threshold = POLYMARKET_EXIT_TRAILING_EV_ALPHA * position['peak_ev_net']
            reasons.append(f"EV fell below {POLYMARKET_EXIT_TRAILING_EV_ALPHA:.0%} of peak ({position['current_ev_net']:.3f} < {threshold:.3f})")

        if 'TIME_GATE' in exit_rules:
            reasons.append(f"No EV improvement for {POLYMARKET_EXIT_TIME_GATE_DAYS} days")

        if 'SIGNAL_REVERSAL' in exit_rules:
            reasons.append("Signal reversal (>=3 agents flipped)")

        if 'PROFIT_TARGET' in exit_rules:
            reasons.append(f"Profit target hit ({position['pnl_pct']:.1%})")

        if 'STOP_LOSS' in exit_rules:
            reasons.append(f"Stop loss hit ({position['pnl_pct']:.1%})")

        return '; '.join(reasons)

    def get_open_positions_summary(self) -> pd.DataFrame:
        """Get summary of open positions"""
        if self.open_positions.empty:
            return pd.DataFrame()

        summary = self.open_positions[[
            'position_id',
            'question',
            'side',
            'position_size',
            'entry_price',
            'current_price',
            'pnl_pct',
            'current_ev_net',
            'current_z_score'
        ]].copy()

        return summary

    def get_performance_statistics(self) -> Dict:
        """Get performance statistics from closed positions"""

        if self.closed_positions.empty:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl_pct': 0.0,
                'total_pnl_usd': 0.0
            }

        winning_trades = self.closed_positions[self.closed_positions['pnl_pct'] > 0]
        losing_trades = self.closed_positions[self.closed_positions['pnl_pct'] <= 0]

        stats = {
            'total_trades': len(self.closed_positions),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.closed_positions),
            'avg_pnl_pct': self.closed_positions['pnl_pct'].mean(),
            'avg_winning_pnl_pct': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
            'avg_losing_pnl_pct': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
            'total_pnl_usd': self.closed_positions['pnl_usd'].sum(),
            'avg_hold_time_hours': self.closed_positions['hold_time_hours'].mean(),
            'exit_rule_distribution': self.closed_positions['exit_rule_triggered'].value_counts().to_dict()
        }

        return stats

    def _save_positions(self):
        """Save open positions to CSV"""
        self.open_positions.to_csv(self.positions_file, index=False)

    def _save_closed_positions(self):
        """Save closed positions to CSV"""
        self.closed_positions.to_csv(self.closed_positions_file, index=False)

    def print_summary(self):
        """Print exit manager summary"""

        stats = self.get_performance_statistics()

        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] EXIT MANAGER SUMMARY", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        cprint(f"\n[EMOJI] POSITIONS:", "yellow")
        cprint(f"   Open Positions: {len(self.open_positions)}", "green")
        cprint(f"   Closed Positions: {len(self.closed_positions)}", "cyan")

        if stats['total_trades'] > 0:
            cprint(f"\n[EMOJI] PERFORMANCE:", "yellow")
            cprint(f"   Win Rate: {stats['win_rate']:.1%}", "green" if stats['win_rate'] > 0.5 else "red")
            cprint(f"   Avg PnL: {stats['avg_pnl_pct']:+.2%}", "green" if stats['avg_pnl_pct'] > 0 else "red")
            pnl_sign = "+" if stats['total_pnl_usd'] > 0 else ""
            cprint(f"   Total PnL: {pnl_sign}{format_currency(abs(stats['total_pnl_usd']))}", "green" if stats['total_pnl_usd'] > 0 else "red", attrs=['bold'])
            cprint(f"   Avg Hold Time: {stats['avg_hold_time_hours']:.1f} hours", "cyan")

            if stats['winning_trades'] > 0:
                cprint(f"\n[EMOJI] WINNING TRADES:", "green")
                cprint(f"   Count: {stats['winning_trades']}", "green")
                cprint(f"   Avg PnL: {stats['avg_winning_pnl_pct']:+.2%}", "green")

            if stats['losing_trades'] > 0:
                cprint(f"\n[SYM]  LOSING TRADES:", "red")
                cprint(f"   Count: {stats['losing_trades']}", "red")
                cprint(f"   Avg PnL: {stats['avg_losing_pnl_pct']:+.2%}", "red")

            if stats['exit_rule_distribution']:
                cprint(f"\n[EMOJI] EXIT RULE DISTRIBUTION:", "yellow")
                for rule, count in stats['exit_rule_distribution'].items():
                    cprint(f"   {rule}: {count}", "cyan")

        cprint(f"\n{'='*80}\n", "cyan")


def main():
    """Test Exit Manager"""

    exit_mgr = PolymarketExitManager()

    # Example: Open position and check exit rules
    cprint("\n" + "="*80, "magenta")
    cprint("TEST: Position Lifecycle", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    # Open position
    position_id = exit_mgr.open_position(
        market_id='test_btc',
        question='Will Bitcoin hit $100k by EOY?',
        side='YES',
        position_size=500,
        entry_price=0.42,
        entry_ev_net=0.05,
        entry_z_score=2.1
    )

    # Check 1: EV still good
    result1 = exit_mgr.check_position(
        position_id=position_id,
        current_price=0.44,
        current_ev_net=0.04,
        current_z_score=1.8,
        bullish_signals_count=4,
        bearish_signals_count=1
    )

    # Check 2: EV decayed below threshold
    result2 = exit_mgr.check_position(
        position_id=position_id,
        current_price=0.48,
        current_ev_net=0.008,  # Below 1% threshold
        current_z_score=0.6,  # Below 0.8 threshold
        bullish_signals_count=2,
        bearish_signals_count=4  # Signal reversal
    )

    if result2['should_exit']:
        closed = exit_mgr.close_position(
            position_id=position_id,
            exit_price=0.48,
            exit_rules_triggered=result2['exit_rules_triggered']
        )

    # Print summary
    exit_mgr.print_summary()

    cprint("\n[SYM] Exit Manager Tests Complete!\n", "green", attrs=['bold'])


if __name__ == "__main__":
    main()
