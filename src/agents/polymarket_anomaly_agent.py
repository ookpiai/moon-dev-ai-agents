"""
[EMOJI] Moon Dev's Polymarket Anomaly Agent
Statistical detection of unusual market behavior using z-scores
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
    POLYMARKET_ANOMALY_Z_THRESHOLD,
    POLYMARKET_ANOMALY_WINDOW_HOURS,
    POLYMARKET_ANOMALY_MIN_DATA_POINTS,
    POLYMARKET_ANOMALY_VOLUME_Z_THRESHOLD,
    POLYMARKET_ANOMALY_CHECK_INTERVAL_SEC,
    POLYMARKET_DATA_DIR,
    POLYMARKET_VERBOSE_LOGGING
)


class PolymarketAnomalyAgent:
    """
    Polymarket Anomaly Agent - Statistical Anomaly Detection

    Detects unusual market behavior using rolling z-score analysis.
    Monitors price, volume, and liquidity for statistical outliers.

    Key Features:
    1. Rolling z-score calculation (24h window)
    2. Multi-dimensional anomaly detection (price, volume, liquidity)
    3. Minimum data requirement (≥20 points)
    4. Configurable thresholds (2.0σ price, 2.5σ volume)
    5. Severity classification (mild, moderate, severe, extreme)
    6. Real-time monitoring

    Anomaly Severity = max(price_z, volume_z, liquidity_z) / threshold
    """

    def __init__(self):
        """Initialize Anomaly Agent"""
        self.data_dir = Path(POLYMARKET_DATA_DIR) / 'anomaly'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.utils = PolymarketUtils()

        # Data files
        self.market_history_file = self.data_dir / 'market_history.csv'
        self.anomalies_file = self.data_dir / 'anomalies.csv'
        self.anomaly_signals_file = self.data_dir / 'anomaly_signals.csv'

        # Initialize data storage
        self.market_history = self._load_or_create_market_history()
        self.anomalies = self._load_or_create_anomalies()
        self.anomaly_signals = self._load_or_create_anomaly_signals()

        cprint(f"\n[EMOJI] Polymarket Anomaly Agent Initialized", "cyan", attrs=['bold'])
        cprint(f"[EMOJI] Tracked Markets: {self.market_history['market_id'].nunique() if not self.market_history.empty else 0}", "green")
        cprint(f"[EMOJI] Historical Data Points: {len(self.market_history)}", "green")
        cprint(f"[EMOJI] Anomalies Detected: {len(self.anomalies)}", "yellow")

    def _load_or_create_market_history(self) -> pd.DataFrame:
        """Load or create market history tracking database"""
        if self.market_history_file.exists():
            df = pd.read_csv(self.market_history_file)
            # Keep last 7 days
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = datetime.now() - timedelta(days=7)
            df = df[df['timestamp'] > cutoff].copy()
            cprint(f"[SYM] Loaded {len(df)} historical data points", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'market_id',
                'yes_price',
                'no_price',
                'volume_24h',
                'liquidity',
                'spread'
            ])
            cprint(f"[EMOJI] Created new market history database", "yellow")
            return df

    def _load_or_create_anomalies(self) -> pd.DataFrame:
        """Load or create anomalies database"""
        if self.anomalies_file.exists():
            df = pd.read_csv(self.anomalies_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = datetime.now() - timedelta(days=7)
            df = df[df['timestamp'] > cutoff].copy()
            cprint(f"[SYM] Loaded {len(df)} anomalies", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'market_id',
                'question',
                'anomaly_type',  # price, volume, liquidity
                'z_score',
                'severity',  # mild, moderate, severe, extreme
                'current_value',
                'mean_value',
                'std_value'
            ])
            cprint(f"[EMOJI] Created new anomalies database", "yellow")
            return df

    def _load_or_create_anomaly_signals(self) -> pd.DataFrame:
        """Load or create anomaly signals database"""
        if self.anomaly_signals_file.exists():
            df = pd.read_csv(self.anomaly_signals_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = datetime.now() - timedelta(hours=24)
            df = df[df['timestamp'] > cutoff].copy()
            cprint(f"[SYM] Loaded {len(df)} anomaly signals", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'market_id',
                'question',
                'anomaly_types',  # comma-separated
                'max_z_score',
                'severity',
                'signal_strength',
                'price_change_pct',
                'volume_change_pct'
            ])
            cprint(f"[EMOJI] Created new anomaly signals database", "yellow")
            return df

    def record_market_snapshot(
        self,
        market_id: str,
        yes_price: float,
        no_price: float,
        volume_24h: float,
        liquidity: float,
        spread: float
    ):
        """
        Record market snapshot for historical tracking

        Args:
            market_id: Market ID
            yes_price: YES outcome price (0-1)
            no_price: NO outcome price (0-1)
            volume_24h: 24h volume in USD
            liquidity: Market liquidity in USD
            spread: Bid-ask spread
        """
        snapshot = {
            'timestamp': datetime.now(),
            'market_id': market_id,
            'yes_price': yes_price,
            'no_price': no_price,
            'volume_24h': volume_24h,
            'liquidity': liquidity,
            'spread': spread
        }

        self.market_history = pd.concat([
            self.market_history,
            pd.DataFrame([snapshot])
        ], ignore_index=True)

        if POLYMARKET_VERBOSE_LOGGING:
            cprint(f"[EMOJI] Recorded snapshot for {market_id}", "blue")

    def detect_anomalies(self, market_id: str) -> List[Dict]:
        """
        Detect anomalies for a specific market using rolling z-scores

        Args:
            market_id: Market ID to analyze

        Returns:
            List of anomaly dicts
        """
        # Get market history
        history = self.market_history[
            self.market_history['market_id'] == market_id
        ].copy()

        if len(history) < POLYMARKET_ANOMALY_MIN_DATA_POINTS:
            if POLYMARKET_VERBOSE_LOGGING:
                cprint(f"⏳ Market {market_id}: Need {POLYMARKET_ANOMALY_MIN_DATA_POINTS - len(history)} more data points", "yellow")
            return []

        # Sort by timestamp
        history = history.sort_values('timestamp')

        # Get recent window (last N hours)
        cutoff = datetime.now() - timedelta(hours=POLYMARKET_ANOMALY_WINDOW_HOURS)
        window = history[history['timestamp'] > cutoff].copy()

        if len(window) < POLYMARKET_ANOMALY_MIN_DATA_POINTS:
            return []

        # Calculate rolling statistics
        anomalies = []

        # Current values (most recent)
        latest = history.iloc[-1]

        # 1. Price Anomaly (YES price)
        price_mean = window['yes_price'].mean()
        price_std = window['yes_price'].std()

        if price_std > 0:
            price_z = abs((latest['yes_price'] - price_mean) / price_std)

            if price_z >= POLYMARKET_ANOMALY_Z_THRESHOLD:
                severity = self._classify_severity(price_z, POLYMARKET_ANOMALY_Z_THRESHOLD)

                anomalies.append({
                    'timestamp': datetime.now(),
                    'market_id': market_id,
                    'anomaly_type': 'price',
                    'z_score': price_z,
                    'severity': severity,
                    'current_value': latest['yes_price'],
                    'mean_value': price_mean,
                    'std_value': price_std
                })

                if POLYMARKET_VERBOSE_LOGGING:
                    cprint(f"[EMOJI] PRICE ANOMALY: z={price_z:.2f} ({severity})", "red")

        # 2. Volume Anomaly
        volume_mean = window['volume_24h'].mean()
        volume_std = window['volume_24h'].std()

        if volume_std > 0:
            volume_z = abs((latest['volume_24h'] - volume_mean) / volume_std)

            if volume_z >= POLYMARKET_ANOMALY_VOLUME_Z_THRESHOLD:
                severity = self._classify_severity(volume_z, POLYMARKET_ANOMALY_VOLUME_Z_THRESHOLD)

                anomalies.append({
                    'timestamp': datetime.now(),
                    'market_id': market_id,
                    'anomaly_type': 'volume',
                    'z_score': volume_z,
                    'severity': severity,
                    'current_value': latest['volume_24h'],
                    'mean_value': volume_mean,
                    'std_value': volume_std
                })

                if POLYMARKET_VERBOSE_LOGGING:
                    cprint(f"[EMOJI] VOLUME ANOMALY: z={volume_z:.2f} ({severity})", "red")

        # 3. Liquidity Anomaly
        liquidity_mean = window['liquidity'].mean()
        liquidity_std = window['liquidity'].std()

        if liquidity_std > 0:
            liquidity_z = abs((latest['liquidity'] - liquidity_mean) / liquidity_std)

            if liquidity_z >= POLYMARKET_ANOMALY_Z_THRESHOLD:
                severity = self._classify_severity(liquidity_z, POLYMARKET_ANOMALY_Z_THRESHOLD)

                anomalies.append({
                    'timestamp': datetime.now(),
                    'market_id': market_id,
                    'anomaly_type': 'liquidity',
                    'z_score': liquidity_z,
                    'severity': severity,
                    'current_value': latest['liquidity'],
                    'mean_value': liquidity_mean,
                    'std_value': liquidity_std
                })

                if POLYMARKET_VERBOSE_LOGGING:
                    cprint(f"[EMOJI] LIQUIDITY ANOMALY: z={liquidity_z:.2f} ({severity})", "red")

        return anomalies

    def _classify_severity(self, z_score: float, threshold: float) -> str:
        """
        Classify anomaly severity based on z-score

        Args:
            z_score: Z-score value
            threshold: Detection threshold

        Returns:
            Severity level: 'mild', 'moderate', 'severe', 'extreme'
        """
        ratio = z_score / threshold

        if ratio >= 3.0:
            return 'extreme'
        elif ratio >= 2.0:
            return 'severe'
        elif ratio >= 1.5:
            return 'moderate'
        else:
            return 'mild'

    def analyze_market(self, market_id: str, question: str) -> Optional[Dict]:
        """
        Analyze market for anomalies and generate signal

        Args:
            market_id: Market ID
            question: Market question

        Returns:
            Signal dict or None if no significant anomalies
        """
        cprint(f"\n[EMOJI] ANALYZING MARKET: {market_id}", "cyan")
        cprint(f"[SYM] {question[:60]}...", "white")

        # Detect anomalies
        anomalies = self.detect_anomalies(market_id)

        if not anomalies:
            if POLYMARKET_VERBOSE_LOGGING:
                cprint(f"[SYM] No anomalies detected", "green")
            return None

        cprint(f"[EMOJI] ANOMALIES DETECTED: {len(anomalies)}", "red", attrs=['bold'])

        # Store anomalies
        for anom in anomalies:
            anom['question'] = question
            self.anomalies = pd.concat([
                self.anomalies,
                pd.DataFrame([anom])
            ], ignore_index=True)

            cprint(
                f"   [EMOJI] {anom['anomaly_type'].upper()}: z={anom['z_score']:.2f} ({anom['severity']})",
                "yellow"
            )

        # Generate signal if anomalies are significant
        signal = self._generate_anomaly_signal(market_id, question, anomalies)

        if signal:
            self.anomaly_signals = pd.concat([
                self.anomaly_signals,
                pd.DataFrame([signal])
            ], ignore_index=True)

            cprint(f"[SYM] SIGNAL GENERATED (Strength: {signal['signal_strength']:.2f})", "green", attrs=['bold'])

        # Save data
        self._save_data()

        return signal

    def _generate_anomaly_signal(
        self,
        market_id: str,
        question: str,
        anomalies: List[Dict]
    ) -> Optional[Dict]:
        """
        Generate trading signal from anomalies

        Args:
            market_id: Market ID
            question: Market question
            anomalies: List of detected anomalies

        Returns:
            Signal dict or None
        """
        if not anomalies:
            return None

        # Find max z-score
        max_z = max(a['z_score'] for a in anomalies)
        max_severity = max(
            (a for a in anomalies if a['z_score'] == max_z),
            key=lambda x: x['z_score']
        )['severity']

        # Get anomaly types
        anomaly_types = ','.join(sorted(set(a['anomaly_type'] for a in anomalies)))

        # Calculate price change
        history = self.market_history[
            self.market_history['market_id'] == market_id
        ].sort_values('timestamp')

        if len(history) >= 2:
            price_change_pct = (
                (history.iloc[-1]['yes_price'] - history.iloc[-2]['yes_price'])
                / history.iloc[-2]['yes_price']
            ) * 100
            volume_change_pct = (
                (history.iloc[-1]['volume_24h'] - history.iloc[-2]['volume_24h'])
                / history.iloc[-2]['volume_24h']
            ) * 100 if history.iloc[-2]['volume_24h'] > 0 else 0
        else:
            price_change_pct = 0
            volume_change_pct = 0

        # Calculate signal strength
        # Strength based on: max_z / threshold × number of anomalies
        severity_multiplier = {
            'mild': 1.0,
            'moderate': 1.5,
            'severe': 2.0,
            'extreme': 3.0
        }

        signal_strength = (
            (max_z / POLYMARKET_ANOMALY_Z_THRESHOLD) *
            severity_multiplier.get(max_severity, 1.0) *
            (1 + 0.1 * (len(anomalies) - 1))  # Bonus for multiple anomalies
        )

        signal = {
            'timestamp': datetime.now(),
            'market_id': market_id,
            'question': question,
            'anomaly_types': anomaly_types,
            'max_z_score': max_z,
            'severity': max_severity,
            'signal_strength': signal_strength,
            'price_change_pct': price_change_pct,
            'volume_change_pct': volume_change_pct
        }

        return signal

    def get_active_signals(
        self,
        market_id: Optional[str] = None,
        min_severity: str = 'mild'
    ) -> pd.DataFrame:
        """
        Get active anomaly signals

        Args:
            market_id: Optional market ID to filter by
            min_severity: Minimum severity ('mild', 'moderate', 'severe', 'extreme')

        Returns:
            DataFrame of active signals
        """
        if self.anomaly_signals.empty:
            return pd.DataFrame()

        signals = self.anomaly_signals.copy()
        signals['timestamp'] = pd.to_datetime(signals['timestamp'])

        # Filter by age (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        signals = signals[signals['timestamp'] > cutoff].copy()

        # Filter by market if specified
        if market_id:
            signals = signals[signals['market_id'] == market_id].copy()

        # Filter by severity
        severity_order = ['mild', 'moderate', 'severe', 'extreme']
        min_idx = severity_order.index(min_severity)
        signals = signals[
            signals['severity'].apply(lambda x: severity_order.index(x) >= min_idx)
        ].copy()

        # Sort by signal strength
        signals = signals.sort_values('signal_strength', ascending=False)

        return signals

    def get_market_statistics(self, market_id: str) -> Optional[Dict]:
        """
        Get statistical summary for a market

        Args:
            market_id: Market ID

        Returns:
            Statistics dict or None
        """
        history = self.market_history[
            self.market_history['market_id'] == market_id
        ].copy()

        if history.empty:
            return None

        # Get window
        cutoff = datetime.now() - timedelta(hours=POLYMARKET_ANOMALY_WINDOW_HOURS)
        window = history[history['timestamp'] > cutoff].copy()

        if len(window) < 2:
            return None

        stats = {
            'data_points': len(window),
            'time_span_hours': (window['timestamp'].max() - window['timestamp'].min()).total_seconds() / 3600,
            'price': {
                'current': window['yes_price'].iloc[-1],
                'mean': window['yes_price'].mean(),
                'std': window['yes_price'].std(),
                'min': window['yes_price'].min(),
                'max': window['yes_price'].max()
            },
            'volume': {
                'current': window['volume_24h'].iloc[-1],
                'mean': window['volume_24h'].mean(),
                'std': window['volume_24h'].std(),
                'min': window['volume_24h'].min(),
                'max': window['volume_24h'].max()
            },
            'liquidity': {
                'current': window['liquidity'].iloc[-1],
                'mean': window['liquidity'].mean(),
                'std': window['liquidity'].std(),
                'min': window['liquidity'].min(),
                'max': window['liquidity'].max()
            }
        }

        return stats

    def _save_data(self):
        """Save all data to CSV files"""
        self.market_history.to_csv(self.market_history_file, index=False)
        self.anomalies.to_csv(self.anomalies_file, index=False)
        self.anomaly_signals.to_csv(self.anomaly_signals_file, index=False)

        if POLYMARKET_VERBOSE_LOGGING:
            cprint(f"[EMOJI] Data saved to {self.data_dir.name}/", "blue")

    def print_summary(self):
        """Print agent summary"""
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] ANOMALY AGENT SUMMARY", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        # Data points
        cprint(f"\n[EMOJI] DATA TRACKING:", "yellow")
        if not self.market_history.empty:
            markets = self.market_history['market_id'].nunique()
            data_points = len(self.market_history)
            cprint(f"   Markets Tracked: {markets}", "cyan")
            cprint(f"   Total Data Points: {data_points}", "cyan")
            cprint(f"   Avg Points/Market: {data_points/markets:.1f}", "cyan")

        # Anomalies
        cprint(f"\n[EMOJI] ANOMALIES:", "yellow")
        if not self.anomalies.empty:
            recent = self.anomalies[
                self.anomalies['timestamp'] > (datetime.now() - timedelta(hours=24))
            ]
            cprint(f"   Total Anomalies (7d): {len(self.anomalies)}", "cyan")
            cprint(f"   Recent Anomalies (24h): {len(recent)}", "green" if len(recent) > 0 else "cyan")

            # By type
            for anom_type in ['price', 'volume', 'liquidity']:
                count = len(self.anomalies[self.anomalies['anomaly_type'] == anom_type])
                cprint(f"   {anom_type.capitalize()}: {count}", "cyan")

            # By severity
            cprint(f"\n   Severity Distribution:", "yellow")
            for severity in ['mild', 'moderate', 'severe', 'extreme']:
                count = len(self.anomalies[self.anomalies['severity'] == severity])
                cprint(f"   {severity.capitalize()}: {count}", "cyan")

        # Active signals
        active = self.get_active_signals()
        cprint(f"\n[EMOJI] ACTIVE SIGNALS:", "yellow")
        cprint(f"   Active Signals (24h): {len(active)}", "green" if len(active) > 0 else "yellow")

        if not active.empty:
            cprint(f"   Avg Signal Strength: {active['signal_strength'].mean():.2f}", "cyan")
            cprint(f"   Max Z-Score: {active['max_z_score'].max():.2f}", "cyan")

        cprint(f"\n{'='*80}\n", "cyan")


def main():
    """Test Anomaly Agent"""

    agent = PolymarketAnomalyAgent()

    test_market_id = 'test_market_123'
    test_question = 'Will Bitcoin hit $100k by EOY 2024?'

    # Simulate normal market data (25 data points)
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 1: Record Normal Market Data", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    base_price = 0.45
    base_volume = 50000
    base_liquidity = 200000

    for i in range(25):
        # Add small random variations (normal)
        agent.record_market_snapshot(
            market_id=test_market_id,
            yes_price=base_price + np.random.normal(0, 0.02),
            no_price=0.55 + np.random.normal(0, 0.02),
            volume_24h=base_volume + np.random.normal(0, 5000),
            liquidity=base_liquidity + np.random.normal(0, 10000),
            spread=0.02 + np.random.normal(0, 0.005)
        )
        time.sleep(0.01)  # Small delay

    cprint(f"[SYM] Recorded 25 normal data points", "green")

    # Analyze (should find no anomalies)
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 2: Analyze Normal Data (No Anomalies Expected)", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    result1 = agent.analyze_market(test_market_id, test_question)

    # Add anomalous data point (price spike)
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 3: Record Anomalous Data (Price Spike)", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    agent.record_market_snapshot(
        market_id=test_market_id,
        yes_price=0.65,  # Large price jump (4+ sigma)
        no_price=0.35,
        volume_24h=base_volume,
        liquidity=base_liquidity,
        spread=0.02
    )

    result2 = agent.analyze_market(test_market_id, test_question)

    # Add volume spike
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 4: Record Volume Spike", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    agent.record_market_snapshot(
        market_id=test_market_id,
        yes_price=0.46,
        no_price=0.54,
        volume_24h=200000,  # 4x volume spike
        liquidity=base_liquidity,
        spread=0.02
    )

    result3 = agent.analyze_market(test_market_id, test_question)

    # Get statistics
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 5: Market Statistics", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    stats = agent.get_market_statistics(test_market_id)
    if stats:
        cprint(f"Data Points: {stats['data_points']}", "cyan")
        cprint(f"Price: {format_probability(stats['price']['current'])} (μ={format_probability(stats['price']['mean'])}, σ={stats['price']['std']:.4f})", "cyan")
        cprint(f"Volume: {format_currency(stats['volume']['current'])} (μ={format_currency(stats['volume']['mean'])})", "cyan")

    # Get active signals
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 6: Active Signals", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    active = agent.get_active_signals()
    if not active.empty:
        cprint(f"Found {len(active)} active signals:", "green")
        for _, sig in active.iterrows():
            cprint(
                f"  {sig['anomaly_types']:20s} | "
                f"z={sig['max_z_score']:5.2f} | "
                f"{sig['severity']:8s} | "
                f"Strength: {sig['signal_strength']:.2f}",
                "cyan"
            )

    # Print summary
    agent.print_summary()

    cprint("\n[SYM] Anomaly Agent Tests Complete!\n", "green", attrs=['bold'])


if __name__ == "__main__":
    main()
