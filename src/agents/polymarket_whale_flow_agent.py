"""
[EMOJI] Moon Dev's Polymarket Whale Flow Agent
Track high-value wallets and detect large orders for insider trading opportunities
Built with love by Moon Dev [EMOJI]
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from termcolor import cprint

# Add project root to path
import sys
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from src.polymarket_utils import PolymarketUtils, format_currency, format_probability
from src.config import (
    POLYMARKET_WHALE_MIN_BET_SIZE,
    POLYMARKET_WHALE_MIN_WIN_RATE,
    POLYMARKET_WHALE_MIN_SAMPLE_SIZE,
    POLYMARKET_WHALE_COPY_THRESHOLD,
    POLYMARKET_WHALE_DECAY_HOURS,
    POLYMARKET_DATA_DIR,
    POLYMARKET_VERBOSE_LOGGING
)


class PolymarketWhaleFlowAgent:
    """
    Polymarket Whale Flow Agent - Smart Money Tracking

    Monitors high-volume wallets and detects large orders that may signal
    insider information or smart money positioning.

    Key Features:
    1. Track top 100 wallets by total volume
    2. Monitor all bets ≥ $10,000
    3. Calculate historical win rates (≥20 bets required)
    4. Only signal wallets with ≥60% win rate
    5. Real-time large order detection
    6. Time-decay scoring (24h decay)

    Signal Strength = wallet_win_rate × bet_size_percentile × time_decay
    """

    def __init__(self):
        """Initialize Whale Flow Agent"""
        self.data_dir = Path(POLYMARKET_DATA_DIR) / 'whale_flow'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.utils = PolymarketUtils()

        # Data files
        self.whale_bets_file = self.data_dir / 'whale_bets.csv'
        self.wallet_stats_file = self.data_dir / 'wallet_stats.csv'
        self.active_signals_file = self.data_dir / 'active_signals.csv'

        # Initialize data storage
        self.whale_bets = self._load_or_create_whale_bets()
        self.wallet_stats = self._load_or_create_wallet_stats()
        self.active_signals = self._load_or_create_active_signals()

        cprint(f"\n[EMOJI] Polymarket Whale Flow Agent Initialized", "cyan", attrs=['bold'])
        cprint(f"[EMOJI] Tracked Wallets: {len(self.wallet_stats)}", "green")
        cprint(f"[EMOJI] Historical Bets: {len(self.whale_bets)}", "green")
        cprint(f"[EMOJI] Active Signals: {len(self.active_signals)}", "yellow")

    def _load_or_create_whale_bets(self) -> pd.DataFrame:
        """Load or create whale bets tracking database"""
        if self.whale_bets_file.exists():
            df = pd.read_csv(self.whale_bets_file)
            cprint(f"[SYM] Loaded {len(df)} whale bets from history", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'wallet_address',
                'market_id',
                'question',
                'bet_side',  # YES or NO
                'bet_size',
                'odds_at_bet',
                'market_resolved',
                'outcome',  # YES or NO (after resolution)
                'profit_loss',
                'win'  # True/False (after resolution)
            ])
            cprint(f"[EMOJI] Created new whale bets database", "yellow")
            return df

    def _load_or_create_wallet_stats(self) -> pd.DataFrame:
        """Load or create wallet statistics database"""
        if self.wallet_stats_file.exists():
            df = pd.read_csv(self.wallet_stats_file)
            cprint(f"[SYM] Loaded stats for {len(df)} wallets", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'wallet_address',
                'total_bets',
                'resolved_bets',
                'wins',
                'losses',
                'win_rate',
                'total_volume',
                'total_profit_loss',
                'roi',
                'last_bet_time',
                'is_whale'  # True if meets whale criteria
            ])
            cprint(f"[EMOJI] Created new wallet stats database", "yellow")
            return df

    def _load_or_create_active_signals(self) -> pd.DataFrame:
        """Load or create active signals database"""
        if self.active_signals_file.exists():
            df = pd.read_csv(self.active_signals_file)
            # Filter out expired signals
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = datetime.now() - timedelta(hours=POLYMARKET_WHALE_DECAY_HOURS)
            df = df[df['timestamp'] > cutoff].copy()
            cprint(f"[SYM] Loaded {len(df)} active signals", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'wallet_address',
                'market_id',
                'question',
                'bet_side',
                'bet_size',
                'odds_at_bet',
                'wallet_win_rate',
                'signal_strength',
                'time_decay'
            ])
            cprint(f"[EMOJI] Created new active signals database", "yellow")
            return df

    def track_wallet(self, wallet_address: str, bets: List[Dict]) -> Dict:
        """
        Track a wallet and update statistics

        Args:
            wallet_address: Wallet address to track
            bets: List of bet dicts with keys:
                  {timestamp, market_id, question, bet_side, bet_size, odds_at_bet,
                   market_resolved, outcome, profit_loss}

        Returns:
            Wallet statistics dict
        """
        cprint(f"\n[EMOJI] Tracking Wallet: {wallet_address[:8]}...", "cyan")

        # Add bets to database
        for bet in bets:
            bet['wallet_address'] = wallet_address
            self.whale_bets = pd.concat([
                self.whale_bets,
                pd.DataFrame([bet])
            ], ignore_index=True)

        # Calculate wallet statistics
        wallet_bets = self.whale_bets[
            self.whale_bets['wallet_address'] == wallet_address
        ].copy()

        total_bets = len(wallet_bets)
        resolved_bets = len(wallet_bets[wallet_bets['market_resolved'] == True])

        if resolved_bets > 0:
            wins = wallet_bets[wallet_bets['win'] == True].shape[0]
            losses = resolved_bets - wins
            win_rate = wins / resolved_bets
        else:
            wins = 0
            losses = 0
            win_rate = 0.0

        total_volume = wallet_bets['bet_size'].sum()
        total_profit_loss = wallet_bets['profit_loss'].sum() if 'profit_loss' in wallet_bets.columns else 0
        roi = (total_profit_loss / total_volume) if total_volume > 0 else 0

        last_bet_time = wallet_bets['timestamp'].max()

        # Determine if whale status
        is_whale = (
            resolved_bets >= POLYMARKET_WHALE_MIN_SAMPLE_SIZE and
            win_rate >= POLYMARKET_WHALE_MIN_WIN_RATE
        )

        stats = {
            'wallet_address': wallet_address,
            'total_bets': total_bets,
            'resolved_bets': resolved_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_volume': total_volume,
            'total_profit_loss': total_profit_loss,
            'roi': roi,
            'last_bet_time': last_bet_time,
            'is_whale': is_whale
        }

        # Update wallet stats database
        if wallet_address in self.wallet_stats['wallet_address'].values:
            # Update existing
            mask = self.wallet_stats['wallet_address'] == wallet_address
            for key, value in stats.items():
                self.wallet_stats.loc[mask, key] = value
        else:
            # Add new
            self.wallet_stats = pd.concat([
                self.wallet_stats,
                pd.DataFrame([stats])
            ], ignore_index=True)

        # Print summary
        cprint(f"[EMOJI] Total Bets: {total_bets} | Resolved: {resolved_bets}", "blue")
        if resolved_bets >= POLYMARKET_WHALE_MIN_SAMPLE_SIZE:
            cprint(f"[EMOJI] Win Rate: {win_rate:.1%} ({wins}W/{losses}L)", "cyan")
            cprint(f"[EMOJI] Total Volume: {format_currency(total_volume)}", "cyan")
            cprint(f"[EMOJI] ROI: {roi:.1%}", "green" if roi > 0 else "red")

            if is_whale:
                cprint(f"[EMOJI] WHALE STATUS: Confirmed (≥{POLYMARKET_WHALE_MIN_WIN_RATE:.0%} win rate)", "green", attrs=['bold'])
            else:
                cprint(f"[SYM]  Below whale threshold ({POLYMARKET_WHALE_MIN_WIN_RATE:.0%})", "yellow")
        else:
            cprint(f"⏳ Need {POLYMARKET_WHALE_MIN_SAMPLE_SIZE - resolved_bets} more resolved bets for whale status", "yellow")

        # Save updated data
        self._save_data()

        return stats

    def detect_large_order(
        self,
        wallet_address: str,
        market_id: str,
        question: str,
        bet_side: str,
        bet_size: float,
        odds_at_bet: float
    ) -> Dict:
        """
        Detect and process large order

        Args:
            wallet_address: Wallet placing bet
            market_id: Market ID
            question: Market question
            bet_side: 'YES' or 'NO'
            bet_size: Bet size in USD
            odds_at_bet: Odds at time of bet (0-1)

        Returns:
            Signal dict or None if not significant
        """
        # Check if bet meets minimum size threshold
        if bet_size < POLYMARKET_WHALE_MIN_BET_SIZE:
            if POLYMARKET_VERBOSE_LOGGING:
                cprint(f"⏭  Bet size ${bet_size:,.0f} below threshold ${POLYMARKET_WHALE_MIN_BET_SIZE:,.0f}", "yellow")
            return None

        cprint(f"\n[EMOJI] LARGE ORDER DETECTED", "red", attrs=['bold'])
        cprint(f"[EMOJI] Size: {format_currency(bet_size)}", "yellow")
        cprint(f"[EMOJI] Market: {question[:60]}...", "cyan")
        cprint(f"[EMOJI] Side: {bet_side} @ {format_probability(odds_at_bet)}", "cyan")

        # Get wallet statistics
        wallet_stats = self.wallet_stats[
            self.wallet_stats['wallet_address'] == wallet_address
        ]

        if wallet_stats.empty:
            cprint(f"[SYM]  Unknown wallet - no historical data", "yellow")
            wallet_win_rate = 0.0
            is_whale = False
        else:
            stats = wallet_stats.iloc[0]
            wallet_win_rate = stats['win_rate']
            is_whale = stats['is_whale']
            resolved_bets = stats['resolved_bets']

            cprint(f"[EMOJI] Wallet Stats:", "cyan")
            cprint(f"   Win Rate: {wallet_win_rate:.1%} ({resolved_bets} bets)", "cyan")
            cprint(f"   Whale Status: {'[SYM] YES' if is_whale else '[SYM] NO'}", "green" if is_whale else "red")

        # Calculate signal strength
        signal = self._calculate_signal_strength(
            wallet_address=wallet_address,
            wallet_win_rate=wallet_win_rate,
            is_whale=is_whale,
            bet_size=bet_size,
            bet_side=bet_side,
            odds_at_bet=odds_at_bet
        )

        # Add to active signals if strong enough
        if signal and signal['signal_strength'] >= 0.5:
            signal_data = {
                'timestamp': datetime.now(),
                'wallet_address': wallet_address,
                'market_id': market_id,
                'question': question,
                'bet_side': bet_side,
                'bet_size': bet_size,
                'odds_at_bet': odds_at_bet,
                'wallet_win_rate': wallet_win_rate,
                'signal_strength': signal['signal_strength'],
                'time_decay': 1.0
            }

            self.active_signals = pd.concat([
                self.active_signals,
                pd.DataFrame([signal_data])
            ], ignore_index=True)

            cprint(f"\n[SYM] SIGNAL ADDED (Strength: {signal['signal_strength']:.2f})", "green", attrs=['bold'])

            # Save
            self._save_data()

            return signal_data

        else:
            cprint(f"\n[SYM]  Signal too weak (strength: {signal['signal_strength']:.2f})", "yellow")
            return None

    def _calculate_signal_strength(
        self,
        wallet_address: str,
        wallet_win_rate: float,
        is_whale: bool,
        bet_size: float,
        bet_side: str,
        odds_at_bet: float
    ) -> Optional[Dict]:
        """
        Calculate signal strength for large order

        Formula:
        signal_strength = wallet_win_rate × bet_size_percentile × time_decay

        Returns:
            {signal_strength, components}
        """
        if not is_whale:
            # Wallet doesn't meet whale criteria
            return {
                'signal_strength': 0.0,
                'reason': 'Wallet does not meet whale criteria',
                'wallet_win_rate': wallet_win_rate,
                'is_whale': False
            }

        # Calculate bet size percentile (relative to all whale bets)
        all_bet_sizes = self.whale_bets['bet_size'].values
        if len(all_bet_sizes) > 10:
            percentile = np.sum(all_bet_sizes < bet_size) / len(all_bet_sizes)
        else:
            # Default to high percentile for large bets when not enough data
            percentile = 0.8 if bet_size >= POLYMARKET_WHALE_MIN_BET_SIZE * 2 else 0.6

        # Time decay = 1.0 (fresh signal)
        time_decay = 1.0

        # Calculate signal strength
        signal_strength = wallet_win_rate * percentile * time_decay

        return {
            'signal_strength': signal_strength,
            'components': {
                'wallet_win_rate': wallet_win_rate,
                'bet_size_percentile': percentile,
                'time_decay': time_decay
            },
            'is_whale': True
        }

    def get_active_signals(self, market_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get active whale signals with time decay applied

        Args:
            market_id: Optional market ID to filter by

        Returns:
            DataFrame of active signals with updated time_decay
        """
        if self.active_signals.empty:
            return pd.DataFrame()

        # Apply time decay
        now = datetime.now()
        signals = self.active_signals.copy()
        signals['timestamp'] = pd.to_datetime(signals['timestamp'])

        # Calculate hours elapsed
        signals['hours_elapsed'] = (now - signals['timestamp']).dt.total_seconds() / 3600

        # Linear decay over POLYMARKET_WHALE_DECAY_HOURS
        signals['time_decay'] = np.maximum(
            0,
            1 - (signals['hours_elapsed'] / POLYMARKET_WHALE_DECAY_HOURS)
        )

        # Recalculate signal strength with decay
        signals['signal_strength'] = (
            signals['wallet_win_rate'] *
            signals['time_decay']
        )

        # Filter out expired signals
        signals = signals[signals['time_decay'] > 0].copy()

        # Filter by market if specified
        if market_id:
            signals = signals[signals['market_id'] == market_id].copy()

        # Sort by signal strength
        signals = signals.sort_values('signal_strength', ascending=False)

        return signals

    def get_whale_consensus(self, market_id: str) -> Dict:
        """
        Calculate whale consensus for a specific market

        Args:
            market_id: Market ID

        Returns:
            {
                'has_consensus': bool,
                'consensus_side': 'YES' or 'NO',
                'signal_count': int,
                'total_volume': float,
                'avg_signal_strength': float
            }
        """
        signals = self.get_active_signals(market_id)

        if signals.empty:
            return {
                'has_consensus': False,
                'consensus_side': None,
                'signal_count': 0,
                'total_volume': 0,
                'avg_signal_strength': 0
            }

        # Count signals by side
        yes_signals = signals[signals['bet_side'] == 'YES']
        no_signals = signals[signals['bet_side'] == 'NO']

        yes_count = len(yes_signals)
        no_count = len(no_signals)

        yes_volume = yes_signals['bet_size'].sum()
        no_volume = no_signals['bet_size'].sum()

        yes_strength = yes_signals['signal_strength'].mean() if yes_count > 0 else 0
        no_strength = no_signals['signal_strength'].mean() if no_count > 0 else 0

        # Determine consensus
        # Require: stronger side has ≥75% of total volume or signal strength
        total_volume = yes_volume + no_volume
        total_strength = yes_strength + no_strength

        if total_volume > 0:
            yes_volume_pct = yes_volume / total_volume
            consensus_by_volume = yes_volume_pct >= POLYMARKET_WHALE_COPY_THRESHOLD

            if consensus_by_volume:
                consensus_side = 'YES' if yes_volume_pct >= POLYMARKET_WHALE_COPY_THRESHOLD else 'NO'
                has_consensus = True
            else:
                consensus_side = None
                has_consensus = False
        else:
            consensus_side = None
            has_consensus = False

        return {
            'has_consensus': has_consensus,
            'consensus_side': consensus_side,
            'signal_count': len(signals),
            'yes_count': yes_count,
            'no_count': no_count,
            'yes_volume': yes_volume,
            'no_volume': no_volume,
            'yes_strength': yes_strength,
            'no_strength': no_strength,
            'total_volume': total_volume,
            'avg_signal_strength': signals['signal_strength'].mean()
        }

    def get_top_whales(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N whales by win rate and volume

        Args:
            n: Number of top whales to return

        Returns:
            DataFrame of top whales
        """
        whales = self.wallet_stats[
            self.wallet_stats['is_whale'] == True
        ].copy()

        if whales.empty:
            return pd.DataFrame()

        # Sort by win rate (primary) and total volume (secondary)
        whales = whales.sort_values(['win_rate', 'total_volume'], ascending=False)

        return whales.head(n)

    def _save_data(self):
        """Save all data to CSV files"""
        self.whale_bets.to_csv(self.whale_bets_file, index=False)
        self.wallet_stats.to_csv(self.wallet_stats_file, index=False)
        self.active_signals.to_csv(self.active_signals_file, index=False)

        if POLYMARKET_VERBOSE_LOGGING:
            cprint(f"[EMOJI] Data saved to {self.data_dir.name}/", "blue")

    def print_summary(self):
        """Print agent summary"""
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] WHALE FLOW AGENT SUMMARY", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        # Whale stats
        whales = self.wallet_stats[self.wallet_stats['is_whale'] == True]
        cprint(f"\n[EMOJI] TRACKED WALLETS:", "yellow")
        cprint(f"   Total Wallets: {len(self.wallet_stats)}", "cyan")
        cprint(f"   Confirmed Whales: {len(whales)}", "green")
        cprint(f"   Total Bets: {len(self.whale_bets)}", "cyan")

        # Active signals
        active = self.get_active_signals()
        cprint(f"\n[EMOJI] ACTIVE SIGNALS:", "yellow")
        cprint(f"   Active Signals: {len(active)}", "green" if len(active) > 0 else "yellow")

        if not active.empty:
            cprint(f"   Avg Signal Strength: {active['signal_strength'].mean():.2f}", "cyan")
            cprint(f"   Total Volume: {format_currency(active['bet_size'].sum())}", "cyan")

        # Top whales
        if not whales.empty:
            top_whales = self.get_top_whales(5)
            cprint(f"\n[EMOJI] TOP 5 WHALES:", "yellow")
            for i, (_, whale) in enumerate(top_whales.iterrows(), 1):
                cprint(
                    f"   {i}. {whale['wallet_address'][:8]}... | "
                    f"WR: {whale['win_rate']:.1%} | "
                    f"Vol: {format_currency(whale['total_volume'])} | "
                    f"ROI: {whale['roi']:.1%}",
                    "cyan"
                )

        cprint(f"\n{'='*80}\n", "cyan")


def main():
    """Test Whale Flow Agent"""

    agent = PolymarketWhaleFlowAgent()

    # Example 1: Track a whale wallet
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 1: Track Whale Wallet", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    # Simulate historical bets for a wallet
    test_wallet = "0x742d35Cc6634C0532925a3b8D57d9f8b2Ae6Cda6"

    test_bets = [
        {
            'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
            'market_id': f'market_{i}',
            'question': f'Test Market {i}',
            'bet_side': 'YES' if i % 3 != 0 else 'NO',
            'bet_size': 15000 + (i * 1000),
            'odds_at_bet': 0.55 + (i * 0.01),
            'market_resolved': True,
            'outcome': 'YES' if i % 3 != 0 else 'NO',  # 67% win rate
            'profit_loss': (15000 + i*1000) * 0.45 if i % 3 != 0 else -(15000 + i*1000),
            'win': i % 3 != 0
        }
        for i in range(25)  # 25 bets with ~67% win rate
    ]

    stats = agent.track_wallet(test_wallet, test_bets)

    # Example 2: Detect large order
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 2: Detect Large Order", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    signal = agent.detect_large_order(
        wallet_address=test_wallet,
        market_id='market_test_large',
        question='Will Bitcoin hit $100k by EOY 2024?',
        bet_side='YES',
        bet_size=50000,
        odds_at_bet=0.42
    )

    # Example 3: Get active signals
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 3: Active Signals", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    active_signals = agent.get_active_signals()
    if not active_signals.empty:
        cprint(f"Found {len(active_signals)} active signals:", "green")
        for _, sig in active_signals.iterrows():
            cprint(
                f"  [EMOJI] {format_currency(sig['bet_size'])} on {sig['bet_side']} | "
                f"Strength: {sig['signal_strength']:.2f} | "
                f"Decay: {sig['time_decay']:.2f}",
                "cyan"
            )

    # Print summary
    agent.print_summary()

    cprint("\n[SYM] Whale Flow Agent Tests Complete!\n", "green", attrs=['bold'])


if __name__ == "__main__":
    main()
