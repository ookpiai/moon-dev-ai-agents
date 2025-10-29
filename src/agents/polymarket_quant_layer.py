"""
[EMOJI] Moon Dev's Polymarket Quant Layer
Core quantitative decision engine for probability arbitrage trading
Built with love by Moon Dev [EMOJI]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from termcolor import cprint

# Add project root to path
import sys
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from src.polymarket_utils import PolymarketUtils, format_currency, format_probability
from src.config import (
    POLYMARKET_EV_MIN,
    POLYMARKET_Z_MIN,
    POLYMARKET_MAX_SPREAD,
    POLYMARKET_MIN_LIQUIDITY,
    POLYMARKET_MIN_VOLUME_24H,
    POLYMARKET_MAX_DAYS_TO_RESOLUTION,
    POLYMARKET_KELLY_FRACTION,
    POLYMARKET_MAX_PER_MARKET,
    POLYMARKET_MAX_PER_THEME,
    POLYMARKET_MIN_POSITION_SIZE,
    POLYMARKET_MAX_POSITION_SIZE,
    POLYMARKET_REGIME_INFORMATION_MULT,
    POLYMARKET_REGIME_ILLIQUID_MULT,
    POLYMARKET_REGIME_EMOTION_MULT,
    POLYMARKET_DATA_DIR,
    POLYMARKET_VERBOSE_LOGGING
)


class PolymarketQuantLayer:
    """
    Polymarket Quant Layer - Quantitative Decision Engine

    Converts forecasts and signals into objective trade decisions using:
    1. EV_net calculation (expected value - costs)
    2. Z-score significance testing
    3. Multi-gate entry system (ALL must pass)
    4. Regime classification (Information/Illiquid/Emotion)
    5. Kelly position sizing with constraints

    Entry Requirements (ALL):
    - EV_net >= 0.03 (3% edge after costs)
    - z >= 1.5 (1.5 sigma significance)
    - spread <= 0.06 (6% max)
    - liquidity >= $10,000
    """

    def __init__(self, portfolio_value: float = 10000):
        """
        Initialize Quant Layer

        Args:
            portfolio_value: Total portfolio value for position sizing
        """
        self.data_dir = Path(POLYMARKET_DATA_DIR) / 'quant_layer'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.utils = PolymarketUtils()

        # Portfolio
        self.portfolio_value = portfolio_value

        # Data file
        self.decisions_file = self.data_dir / 'trade_decisions.csv'
        self.decisions = self._load_or_create_decisions()

        cprint(f"\n[EMOJI] Polymarket Quant Layer Initialized", "cyan", attrs=['bold'])
        cprint(f"[EMOJI] Portfolio Value: {format_currency(portfolio_value)}", "green")
        cprint(f"[EMOJI] Historical Decisions: {len(self.decisions)}", "green")
        cprint(f"\n[SYM]  ENTRY GATES:", "yellow")
        cprint(f"   EV_net >= {POLYMARKET_EV_MIN:.2%}", "cyan")
        cprint(f"   Z-score >= {POLYMARKET_Z_MIN}", "cyan")
        cprint(f"   Spread <= {POLYMARKET_MAX_SPREAD:.2%}", "cyan")
        cprint(f"   Liquidity >= {format_currency(POLYMARKET_MIN_LIQUIDITY)}", "cyan")

    def _load_or_create_decisions(self) -> pd.DataFrame:
        """Load or create trade decisions database"""
        if self.decisions_file.exists():
            df = pd.read_csv(self.decisions_file)
            cprint(f"[SYM] Loaded {len(df)} historical decisions", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'market_id',
                'question',
                'true_prob',
                'market_price',
                'edge',
                'ev_net',
                'z_score',
                'spread',
                'liquidity',
                'volume_24h',
                'regime',
                'regime_multiplier',
                'kelly_size',
                'final_position_size',
                'side',  # YES or NO
                'entry_decision',  # ENTER or REJECT
                'rejection_reasons',  # comma-separated gates that failed
                'confidence'
            ])
            cprint(f"[EMOJI] Created new trade decisions database", "yellow")
            return df

    def analyze_opportunity(
        self,
        market_id: str,
        question: str,
        true_prob: float,
        market_price: float,
        spread: float,
        liquidity: float,
        volume_24h: float,
        days_to_resolution: float,
        confidence: float = 0.75,
        forecast_std: Optional[float] = None
    ) -> Dict:
        """
        Analyze trading opportunity using full quant framework

        Args:
            market_id: Market ID
            question: Market question
            true_prob: TRUE probability from forecasters (0-1)
            market_price: Current market price (0-1)
            spread: Bid-ask spread
            liquidity: Market liquidity in USD
            volume_24h: 24h volume in USD
            days_to_resolution: Days until market resolution
            confidence: Forecast confidence (0-1)
            forecast_std: Optional standard deviation of forecast

        Returns:
            Complete decision dict
        """
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] QUANT ANALYSIS", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")
        cprint(f"[SYM] {question[:60]}...", "white", attrs=['bold'])

        # Calculate all metrics
        edge = self._calculate_edge(true_prob, market_price)
        ev_net = self._calculate_ev_net(true_prob, market_price, spread)
        z_score = self._calculate_z_score(true_prob, market_price, forecast_std)
        regime = self._classify_regime(spread, volume_24h, liquidity, market_price)
        position_size = self._calculate_position_size(
            edge=edge,
            confidence=confidence,
            regime_multiplier=regime['multiplier']
        )

        # Determine side (trade YES or NO?)
        side = 'YES' if true_prob > market_price else 'NO'

        # Entry gates check
        entry_decision, rejection_reasons = self._check_entry_gates(
            ev_net=ev_net,
            z_score=z_score,
            spread=spread,
            liquidity=liquidity,
            volume_24h=volume_24h,
            days_to_resolution=days_to_resolution
        )

        # Print metrics
        self._print_analysis(
            true_prob=true_prob,
            market_price=market_price,
            edge=edge,
            ev_net=ev_net,
            z_score=z_score,
            spread=spread,
            liquidity=liquidity,
            volume_24h=volume_24h,
            regime=regime,
            position_size=position_size,
            side=side,
            entry_decision=entry_decision,
            rejection_reasons=rejection_reasons
        )

        # Build result
        result = {
            'market_id': market_id,
            'question': question,
            'true_prob': true_prob,
            'market_price': market_price,
            'edge': edge,
            'ev_net': ev_net,
            'z_score': z_score,
            'spread': spread,
            'liquidity': liquidity,
            'volume_24h': volume_24h,
            'regime': regime['name'],
            'regime_multiplier': regime['multiplier'],
            'kelly_size': position_size['kelly_size'],
            'final_position_size': position_size['final_size'],
            'side': side,
            'entry_decision': entry_decision,
            'rejection_reasons': ','.join(rejection_reasons) if rejection_reasons else '',
            'confidence': confidence
        }

        # Store decision
        self._store_decision(result)

        return result

    def _calculate_edge(self, true_prob: float, market_price: float) -> float:
        """
        Calculate edge (mispricing)

        edge = TRUE_PROB - MARKET_PRICE
        """
        return true_prob - market_price

    def _calculate_ev_net(
        self,
        true_prob: float,
        market_price: float,
        spread: float,
        slippage: float = 0.005,  # 0.5%
        fees: float = 0.02  # 2%
    ) -> float:
        """
        Calculate EV_net (expected value after costs)

        EV_net = EDGE - COSTS
        COSTS = spread/2 + slippage + fees

        Returns:
            EV_net as decimal (e.g., 0.03 = 3%)
        """
        edge = true_prob - market_price
        costs = (spread / 2) + slippage + fees

        ev_net = edge - costs

        return ev_net

    def _calculate_z_score(
        self,
        true_prob: float,
        market_price: float,
        forecast_std: Optional[float] = None
    ) -> float:
        """
        Calculate z-score (statistical significance of mispricing)

        z = (TRUE_PROB - MARKET_PRICE) / sigma

        If forecast_std not provided, estimates from typical forecast uncertainty

        Returns:
            Z-score (number of standard deviations)
        """
        edge = true_prob - market_price

        if forecast_std is None:
            # Estimate std based on price level (prices near 0.5 have higher uncertainty)
            # Typical forecast std ranges from 0.05 to 0.15
            forecast_std = 0.05 + 0.1 * (1 - abs(true_prob - 0.5) * 2)

        if forecast_std == 0:
            return 0.0

        z_score = edge / forecast_std

        return z_score

    def _classify_regime(
        self,
        spread: float,
        volume_24h: float,
        liquidity: float,
        current_price: float,
        price_1h_ago: Optional[float] = None
    ) -> Dict:
        """
        Classify market regime for adaptive position sizing

        Three regimes:
        - Information: Tight spreads, high volume/liquidity (normal sizing)
        - Illiquid: Wide spreads, low volume/liquidity (reduce sizing)
        - Emotion: Wide spreads + high volume + large price movement (increase sizing)

        Returns:
            {name, multiplier, characteristics}
        """
        # Calculate price change if available
        if price_1h_ago:
            price_change_pct = abs(current_price - price_1h_ago)
        else:
            price_change_pct = 0

        # Information Regime: Efficient market
        if (spread <= 0.04 and
            volume_24h >= 5000 and
            liquidity >= 15000):
            return {
                'name': 'INFORMATION',
                'multiplier': POLYMARKET_REGIME_INFORMATION_MULT,
                'characteristics': 'Tight spreads, high liquidity, efficient'
            }

        # Illiquid Regime: Reduce size
        if (spread > 0.06 or
            volume_24h < 2000 or
            liquidity < 8000):
            return {
                'name': 'ILLIQUID',
                'multiplier': POLYMARKET_REGIME_ILLIQUID_MULT,
                'characteristics': 'Wide spreads, low volume/liquidity'
            }

        # Emotion Regime: Panic or euphoria
        if (spread > 0.05 and
            volume_24h >= 10000 and
            price_change_pct > 0.10):
            return {
                'name': 'EMOTION',
                'multiplier': POLYMARKET_REGIME_EMOTION_MULT,
                'characteristics': 'Wide spreads, high volume, large price movement'
            }

        # Default to Information
        return {
            'name': 'INFORMATION',
            'multiplier': POLYMARKET_REGIME_INFORMATION_MULT,
            'characteristics': 'Standard conditions'
        }

    def _calculate_position_size(
        self,
        edge: float,
        confidence: float,
        regime_multiplier: float
    ) -> Dict:
        """
        Calculate position size using fractional Kelly criterion

        f = KELLY_FRACTION × |edge| × confidence × regime_multiplier

        Constrained by:
        - MIN_POSITION_SIZE <= f <= MAX_POSITION_SIZE
        - f <= MAX_PER_MARKET × portfolio_value
        - f <= MAX_PER_THEME × portfolio_value

        Returns:
            {kelly_size, final_size, constraints_applied}
        """
        # Base Kelly size
        kelly_size = (
            POLYMARKET_KELLY_FRACTION *
            abs(edge) *
            confidence *
            regime_multiplier *
            self.portfolio_value
        )

        # Apply constraints
        constraints_applied = []

        final_size = kelly_size

        # Min/Max position size
        if final_size < POLYMARKET_MIN_POSITION_SIZE:
            final_size = POLYMARKET_MIN_POSITION_SIZE
            constraints_applied.append('min_position_size')

        if final_size > POLYMARKET_MAX_POSITION_SIZE:
            final_size = POLYMARKET_MAX_POSITION_SIZE
            constraints_applied.append('max_position_size')

        # Max per market
        max_per_market = POLYMARKET_MAX_PER_MARKET * self.portfolio_value
        if final_size > max_per_market:
            final_size = max_per_market
            constraints_applied.append('max_per_market')

        # Note: MAX_PER_THEME would require tracking theme allocations
        # Implement in orchestrator layer

        return {
            'kelly_size': kelly_size,
            'final_size': final_size,
            'constraints_applied': constraints_applied
        }

    def _check_entry_gates(
        self,
        ev_net: float,
        z_score: float,
        spread: float,
        liquidity: float,
        volume_24h: float,
        days_to_resolution: float
    ) -> Tuple[str, list]:
        """
        Check all entry gates

        ALL gates must pass for ENTER decision

        Returns:
            Tuple of (decision, rejection_reasons)
            - decision: 'ENTER' or 'REJECT'
            - rejection_reasons: List of failed gates
        """
        rejection_reasons = []

        # Gate 1: EV_net
        if ev_net < POLYMARKET_EV_MIN:
            rejection_reasons.append(f'ev_net={ev_net:.3f}<{POLYMARKET_EV_MIN:.3f}')

        # Gate 2: Z-score
        if z_score < POLYMARKET_Z_MIN:
            rejection_reasons.append(f'z={z_score:.2f}<{POLYMARKET_Z_MIN}')

        # Gate 3: Spread
        if spread > POLYMARKET_MAX_SPREAD:
            rejection_reasons.append(f'spread={spread:.3f}>{POLYMARKET_MAX_SPREAD:.3f}')

        # Gate 4: Liquidity
        if liquidity < POLYMARKET_MIN_LIQUIDITY:
            rejection_reasons.append(f'liq=${liquidity:.0f}<${POLYMARKET_MIN_LIQUIDITY}')

        # Gate 5: Volume
        if volume_24h < POLYMARKET_MIN_VOLUME_24H:
            rejection_reasons.append(f'vol=${volume_24h:.0f}<${POLYMARKET_MIN_VOLUME_24H}')

        # Gate 6: Time to resolution
        if days_to_resolution > POLYMARKET_MAX_DAYS_TO_RESOLUTION:
            rejection_reasons.append(f'days={days_to_resolution:.0f}>{POLYMARKET_MAX_DAYS_TO_RESOLUTION}')

        # Decision
        if not rejection_reasons:
            decision = 'ENTER'
        else:
            decision = 'REJECT'

        return decision, rejection_reasons

    def _print_analysis(
        self,
        true_prob: float,
        market_price: float,
        edge: float,
        ev_net: float,
        z_score: float,
        spread: float,
        liquidity: float,
        volume_24h: float,
        regime: Dict,
        position_size: Dict,
        side: str,
        entry_decision: str,
        rejection_reasons: list
    ):
        """Print detailed quantitative analysis"""

        cprint(f"\n[EMOJI] PROBABILITY ASSESSMENT:", "yellow")
        cprint(f"   TRUE Probability:  {format_probability(true_prob)}", "green", attrs=['bold'])
        cprint(f"   Market Price:      {format_probability(market_price)}", "cyan")
        cprint(f"   Edge:              {edge:+.3f} ({edge*100:+.1f}%)", "yellow" if edge > 0 else "red")

        cprint(f"\n[EMOJI] EXPECTED VALUE:", "yellow")
        ev_color = "green" if ev_net >= POLYMARKET_EV_MIN else "red"
        cprint(f"   EV_net:            {ev_net:.3f} ({ev_net*100:.1f}%)", ev_color, attrs=['bold'])
        cprint(f"   Threshold:         {POLYMARKET_EV_MIN:.3f} ({POLYMARKET_EV_MIN*100:.1f}%)", "cyan")
        if ev_net >= POLYMARKET_EV_MIN:
            cprint(f"   [SYM] EV GATE PASSED", "green")
        else:
            cprint(f"   [SYM] EV GATE FAILED", "red")

        cprint(f"\n[EMOJI] STATISTICAL SIGNIFICANCE:", "yellow")
        z_color = "green" if z_score >= POLYMARKET_Z_MIN else "red"
        cprint(f"   Z-score:           {z_score:.2f}sigma", z_color, attrs=['bold'])
        cprint(f"   Threshold:         {POLYMARKET_Z_MIN:.2f}sigma", "cyan")
        if z_score >= POLYMARKET_Z_MIN:
            cprint(f"   [SYM] Z-SCORE GATE PASSED", "green")
        else:
            cprint(f"   [SYM] Z-SCORE GATE FAILED", "red")

        cprint(f"\n[EMOJI] MARKET CONDITIONS:", "yellow")
        spread_color = "green" if spread <= POLYMARKET_MAX_SPREAD else "red"
        cprint(f"   Spread:            {spread:.3f} ({spread*100:.1f}%)", spread_color)
        liq_color = "green" if liquidity >= POLYMARKET_MIN_LIQUIDITY else "red"
        cprint(f"   Liquidity:         {format_currency(liquidity)}", liq_color)
        vol_color = "green" if volume_24h >= POLYMARKET_MIN_VOLUME_24H else "red"
        cprint(f"   Volume 24h:        {format_currency(volume_24h)}", vol_color)

        cprint(f"\n[EMOJI] REGIME CLASSIFICATION:", "yellow")
        cprint(f"   Regime:            {regime['name']}", "cyan", attrs=['bold'])
        cprint(f"   Multiplier:        {regime['multiplier']:.2f}x", "cyan")
        cprint(f"   Characteristics:   {regime['characteristics']}", "blue")

        cprint(f"\n[EMOJI] POSITION SIZING:", "yellow")
        cprint(f"   Kelly Size:        {format_currency(position_size['kelly_size'])}", "cyan")
        cprint(f"   Final Size:        {format_currency(position_size['final_size'])}", "green", attrs=['bold'])
        if position_size['constraints_applied']:
            cprint(f"   Constraints:       {', '.join(position_size['constraints_applied'])}", "yellow")

        cprint(f"\n[EMOJI] TRADE SPECIFICATION:", "yellow")
        cprint(f"   Side:              {side}", "cyan", attrs=['bold'])

        cprint(f"\n[EMOJI] ENTRY DECISION:", "yellow")
        if entry_decision == 'ENTER':
            cprint(f"   [SYM] ENTER TRADE", "green", attrs=['bold'])
            cprint(f"   All entry gates passed!", "green")
        else:
            cprint(f"   [SYM] REJECT TRADE", "red", attrs=['bold'])
            cprint(f"   Failed gates:", "red")
            for reason in rejection_reasons:
                cprint(f"      - {reason}", "red")

        cprint(f"\n{'='*80}\n", "cyan")

    def _store_decision(self, result: Dict):
        """Store trade decision to database"""

        decision_data = {
            'timestamp': datetime.now(),
            **result
        }

        self.decisions = pd.concat([
            self.decisions,
            pd.DataFrame([decision_data])
        ], ignore_index=True)

        # Save
        self.decisions.to_csv(self.decisions_file, index=False)

        if POLYMARKET_VERBOSE_LOGGING:
            cprint(f"[EMOJI] Decision saved", "blue")

    def get_decision_statistics(self) -> Dict:
        """Get statistics on historical decisions"""

        if self.decisions.empty:
            return {
                'total_decisions': 0,
                'enter_decisions': 0,
                'reject_decisions': 0
            }

        enter_decisions = self.decisions[self.decisions['entry_decision'] == 'ENTER']
        reject_decisions = self.decisions[self.decisions['entry_decision'] == 'REJECT']

        stats = {
            'total_decisions': len(self.decisions),
            'enter_decisions': len(enter_decisions),
            'reject_decisions': len(reject_decisions),
            'entry_rate': len(enter_decisions) / len(self.decisions) if len(self.decisions) > 0 else 0,
            'avg_ev_net': self.decisions['ev_net'].mean(),
            'avg_z_score': self.decisions['z_score'].mean(),
            'avg_edge': self.decisions['edge'].mean(),
            'avg_position_size': enter_decisions['final_position_size'].mean() if len(enter_decisions) > 0 else 0
        }

        # Rejection reasons
        if len(reject_decisions) > 0:
            all_reasons = []
            for reasons_str in reject_decisions['rejection_reasons']:
                if reasons_str:
                    all_reasons.extend(reasons_str.split(','))
            stats['common_rejection_reasons'] = pd.Series(all_reasons).value_counts().to_dict()
        else:
            stats['common_rejection_reasons'] = {}

        return stats

    def print_summary(self):
        """Print quant layer summary"""

        stats = self.get_decision_statistics()

        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] QUANT LAYER SUMMARY", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        cprint(f"\n[EMOJI] DECISION STATISTICS:", "yellow")
        cprint(f"   Total Decisions: {stats['total_decisions']}", "cyan")
        cprint(f"   Enter: {stats['enter_decisions']}", "green")
        cprint(f"   Reject: {stats['reject_decisions']}", "red")
        cprint(f"   Entry Rate: {stats['entry_rate']:.1%}", "cyan")

        if stats['total_decisions'] > 0:
            cprint(f"\n[EMOJI] AVERAGE METRICS:", "yellow")
            cprint(f"   Avg EV_net: {stats['avg_ev_net']:.3f} ({stats['avg_ev_net']*100:.1f}%)", "cyan")
            cprint(f"   Avg Z-score: {stats['avg_z_score']:.2f}sigma", "cyan")
            cprint(f"   Avg Edge: {stats['avg_edge']:+.3f} ({stats['avg_edge']*100:+.1f}%)", "cyan")
            if stats['enter_decisions'] > 0:
                cprint(f"   Avg Position Size: {format_currency(stats['avg_position_size'])}", "green")

        if stats['common_rejection_reasons']:
            cprint(f"\n[SYM] COMMON REJECTION REASONS:", "yellow")
            for reason, count in list(stats['common_rejection_reasons'].items())[:5]:
                cprint(f"   {reason}: {count}", "red")

        cprint(f"\n[SYM]  CONFIGURATION:", "yellow")
        cprint(f"   Portfolio Value: {format_currency(self.portfolio_value)}", "cyan")
        cprint(f"   Kelly Fraction: {POLYMARKET_KELLY_FRACTION:.2f}", "cyan")
        cprint(f"   Min Position: {format_currency(POLYMARKET_MIN_POSITION_SIZE)}", "cyan")
        cprint(f"   Max Position: {format_currency(POLYMARKET_MAX_POSITION_SIZE)}", "cyan")

        cprint(f"\n{'='*80}\n", "cyan")


def main():
    """Test Quant Layer"""

    quant = PolymarketQuantLayer(portfolio_value=10000)

    # Example 1: Strong edge, all gates pass
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 1: Strong Opportunity (Should ENTER)", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    result1 = quant.analyze_opportunity(
        market_id='test_strong',
        question='Will Bitcoin hit $100k by EOY 2024?',
        true_prob=0.55,  # TRUE probability from forecasters
        market_price=0.42,  # Current market price
        spread=0.04,  # 4% spread
        liquidity=250000,
        volume_24h=80000,
        days_to_resolution=60,
        confidence=0.8,
        forecast_std=0.08
    )

    # Example 2: Good edge but fails spread gate
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 2: Good Edge, Wide Spread (Should REJECT)", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    result2 = quant.analyze_opportunity(
        market_id='test_wide_spread',
        question='Will Trump win 2024 election?',
        true_prob=0.72,
        market_price=0.65,
        spread=0.08,  # 8% spread (fails gate)
        liquidity=500000,
        volume_24h=200000,
        days_to_resolution=30,
        confidence=0.75
    )

    # Example 3: Small edge, fails EV gate
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 3: Small Edge (Should REJECT)", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    result3 = quant.analyze_opportunity(
        market_id='test_small_edge',
        question='Will Fed cut rates in March?',
        true_prob=0.57,
        market_price=0.55,  # Only 2% edge
        spread=0.03,
        liquidity=150000,
        volume_24h=50000,
        days_to_resolution=45,
        confidence=0.7
    )

    # Print summary
    quant.print_summary()

    cprint("\n[SYM] Quant Layer Tests Complete!\n", "green", attrs=['bold'])


if __name__ == "__main__":
    main()
