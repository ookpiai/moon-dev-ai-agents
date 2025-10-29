"""
[EMOJI] Moon Dev's Polymarket Orchestrator
Master control system - complete probability arbitrage trading pipeline
Built with love by Moon Dev [EMOJI]
"""

import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from termcolor import cprint
import pandas as pd

# Add project root to path
import sys
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from src.polymarket_utils import PolymarketUtils, format_currency, format_probability
from src.agents.polymarket_swarm_forecaster import PolymarketSwarmForecaster
from src.agents.polymarket_llm_forecaster import PolymarketLLMForecaster
from src.agents.polymarket_whale_flow_agent import PolymarketWhaleFlowAgent
from src.agents.polymarket_event_catalyst_agent import PolymarketEventCatalystAgent
from src.agents.polymarket_anomaly_agent import PolymarketAnomalyAgent
from src.agents.polymarket_quant_layer import PolymarketQuantLayer
from src.agents.polymarket_exit_manager import PolymarketExitManager
from src.config import (
    POLYMARKET_ENABLED,
    POLYMARKET_PAPER_TRADING,
    POLYMARKET_USE_SWARM,
    POLYMARKET_USE_LLM_ADJUSTMENT,
    POLYMARKET_USE_WHALE_FLOW,
    POLYMARKET_USE_EVENT_CATALYST,
    POLYMARKET_USE_ANOMALY_DETECTION,
    POLYMARKET_MAX_OPEN_POSITIONS,
    POLYMARKET_MAX_DAILY_TRADES,
    POLYMARKET_MAX_DAILY_LOSS_USD,
    POLYMARKET_MIN_ACCOUNT_BALANCE_USD,
    POLYMARKET_DATA_DIR,
    POLYMARKET_VERBOSE_LOGGING
)


class PolymarketOrchestrator:
    """
    Polymarket Orchestrator - Master Trading Pipeline

    Complete probability arbitrage system:
    1. SENSE: Gather signals (Whale, Event, Anomaly)
    2. THINK: Generate forecasts (Swarm + LLM)
    3. DECIDE: Quant analysis (EV, z-scores, Kelly)
    4. TRADE: Execute limit-at-fair
    5. EXIT: Monitor 6 exit rules
    6. LEARN: Log and analyze

    Architecture:
    - 5 Sensing Agents -> Signals
    - 2 Forecasting Agents -> Probabilities
    - 1 Quant Layer -> Decisions
    - 1 Exit Manager -> Position Management
    """

    def __init__(self, portfolio_value: float = 10000):
        """
        Initialize Orchestrator

        Args:
            portfolio_value: Total portfolio value
        """
        self.portfolio_value = portfolio_value

        # Initialize utilities
        self.utils = PolymarketUtils()

        # Initialize agents (conditional on feature flags)
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] POLYMARKET ORCHESTRATOR INITIALIZATION", "cyan", attrs=['bold'])
        cprint(f"{'='*80}\n", "cyan")

        cprint(f"[EMOJI] Portfolio Value: {format_currency(portfolio_value)}", "green", attrs=['bold'])
        cprint(f"[EMOJI] Paper Trading: {'ENABLED' if POLYMARKET_PAPER_TRADING else 'DISABLED'}", "yellow", attrs=['bold'])

        # Forecasting agents
        self.swarm_forecaster = PolymarketSwarmForecaster() if POLYMARKET_USE_SWARM else None
        self.llm_forecaster = PolymarketLLMForecaster() if POLYMARKET_USE_LLM_ADJUSTMENT else None

        # Sensing agents
        self.whale_agent = PolymarketWhaleFlowAgent() if POLYMARKET_USE_WHALE_FLOW else None
        self.event_agent = PolymarketEventCatalystAgent() if POLYMARKET_USE_EVENT_CATALYST else None
        self.anomaly_agent = PolymarketAnomalyAgent() if POLYMARKET_USE_ANOMALY_DETECTION else None

        # Decision and execution
        self.quant_layer = PolymarketQuantLayer(portfolio_value=portfolio_value)
        self.exit_manager = PolymarketExitManager()

        # Session tracking
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.session_start_time = datetime.now()

        cprint(f"\n[SYM] Orchestrator initialized successfully!", "green", attrs=['bold'])
        cprint(f"{'='*80}\n", "cyan")

    def analyze_market(
        self,
        market_id: str,
        question: str,
        description: str = "",
        current_yes_price: float = 0.5,
        current_no_price: float = 0.5,
        volume_24h: float = 0,
        liquidity: float = 0,
        spread: float = 0.05,
        days_to_resolution: float = 30
    ) -> Dict:
        """
        Complete analysis pipeline for a single market

        Args:
            market_id: Market ID
            question: Market question
            description: Market description
            current_yes_price: Current YES price
            current_no_price: Current NO price
            volume_24h: 24h volume in USD
            liquidity: Market liquidity in USD
            spread: Bid-ask spread
            days_to_resolution: Days until resolution

        Returns:
            Complete analysis result dict
        """
        cprint(f"\n{'='*80}", "magenta")
        cprint(f"[EMOJI] ANALYZING MARKET", "magenta", attrs=['bold'])
        cprint(f"{'='*80}", "magenta")
        cprint(f"[SYM] {question}", "white", attrs=['bold'])
        cprint(f"[EMOJI] Market ID: {market_id}", "cyan")
        cprint(f"[EMOJI] Current: YES={format_probability(current_yes_price)}, NO={format_probability(current_no_price)}", "yellow")
        cprint(f"[EMOJI] Volume: {format_currency(volume_24h)} | Liquidity: {format_currency(liquidity)}", "cyan")

        result = {
            'market_id': market_id,
            'question': question,
            'analysis_time': datetime.now(),
            'current_yes_price': current_yes_price,
            'current_no_price': current_no_price
        }

        # ===== PHASE 1: SENSE =====
        cprint(f"\n{'-'*80}", "cyan")
        cprint(f"[EMOJI] PHASE 1: SENSING", "cyan", attrs=['bold'])
        cprint(f"{'-'*80}", "cyan")

        signals = self._gather_signals(market_id, question, volume_24h, liquidity, current_yes_price)
        result['signals'] = signals

        # ===== PHASE 2: THINK =====
        cprint(f"\n{'-'*80}", "cyan")
        cprint(f"[EMOJI] PHASE 2: FORECASTING", "cyan", attrs=['bold'])
        cprint(f"{'-'*80}", "cyan")

        forecast = self._generate_forecast(
            question=question,
            description=description,
            context={
                'current_odds': {'yes': current_yes_price, 'no': current_no_price},
                'volume_24h': volume_24h,
                'liquidity': liquidity,
                'days_to_resolution': days_to_resolution
            },
            market_id=market_id
        )
        result['forecast'] = forecast

        # ===== PHASE 3: DECIDE =====
        cprint(f"\n{'-'*80}", "cyan")
        cprint(f"[EMOJI] PHASE 3: QUANTITATIVE ANALYSIS", "cyan", attrs=['bold'])
        cprint(f"{'-'*80}", "cyan")

        if forecast and forecast.get('final_probability'):
            decision = self._make_decision(
                market_id=market_id,
                question=question,
                true_prob=forecast['final_probability'],
                market_price=current_yes_price,
                spread=spread,
                liquidity=liquidity,
                volume_24h=volume_24h,
                days_to_resolution=days_to_resolution,
                confidence=forecast.get('confidence', 0.75),
                forecast_std=forecast.get('forecast_std')
            )
            result['decision'] = decision
        else:
            cprint(f"[SYM]  No valid forecast - skipping decision", "yellow")
            result['decision'] = {'entry_decision': 'REJECT', 'rejection_reasons': ['no_forecast']}

        # ===== PHASE 4: TRADE =====
        if result['decision'].get('entry_decision') == 'ENTER':
            cprint(f"\n{'-'*80}", "green")
            cprint(f"[EMOJI] PHASE 4: TRADE EXECUTION", "green", attrs=['bold'])
            cprint(f"{'-'*80}", "green")

            execution = self._execute_trade(result['decision'], current_yes_price)
            result['execution'] = execution
        else:
            result['execution'] = {'status': 'REJECTED'}

        return result

    def _gather_signals(
        self,
        market_id: str,
        question: str,
        volume_24h: float,
        liquidity: float,
        current_price: float
    ) -> Dict:
        """
        Gather signals from all sensing agents

        Returns:
            {whale, event, anomaly, signal_count}
        """
        signals = {
            'whale': None,
            'event': None,
            'anomaly': None,
            'signal_count': 0
        }

        # Whale Flow signals
        if self.whale_agent:
            cprint(f"[EMOJI] Checking Whale Flow...", "cyan")
            whale_signals = self.whale_agent.get_active_signals(market_id)
            if not whale_signals.empty:
                signals['whale'] = {
                    'signal_count': len(whale_signals),
                    'max_strength': whale_signals['signal_strength'].max(),
                    'avg_strength': whale_signals['signal_strength'].mean()
                }
                signals['signal_count'] += len(whale_signals)
                cprint(f"   [SYM] {len(whale_signals)} whale signals (avg strength: {signals['whale']['avg_strength']:.2f})", "green")
            else:
                cprint(f"   [SYM] No whale signals", "white")

        # Event Catalyst signals
        if self.event_agent:
            cprint(f"[EMOJI] Checking Event Catalyst...", "cyan")
            event_signals = self.event_agent.get_active_signals(market_id)
            if not event_signals.empty:
                signals['event'] = {
                    'signal_count': len(event_signals),
                    'max_strength': event_signals['signal_strength'].max(),
                    'directions': event_signals['direction'].value_counts().to_dict()
                }
                signals['signal_count'] += len(event_signals)
                cprint(f"   [SYM] {len(event_signals)} event signals", "green")
            else:
                cprint(f"   [SYM] No event signals", "white")

        # Anomaly signals
        if self.anomaly_agent:
            cprint(f"[EMOJI] Checking Anomaly Detection...", "cyan")
            anomaly_signals = self.anomaly_agent.get_active_signals(market_id)
            if not anomaly_signals.empty:
                signals['anomaly'] = {
                    'signal_count': len(anomaly_signals),
                    'max_z_score': anomaly_signals['max_z_score'].max(),
                    'severities': anomaly_signals['severity'].value_counts().to_dict()
                }
                signals['signal_count'] += len(anomaly_signals)
                cprint(f"   [SYM] {len(anomaly_signals)} anomaly signals", "green")
            else:
                cprint(f"   [SYM] No anomalies detected", "white")

        cprint(f"\n[EMOJI] Total Signals: {signals['signal_count']}", "yellow", attrs=['bold'])

        return signals

    def _generate_forecast(
        self,
        question: str,
        description: str,
        context: Dict,
        market_id: str
    ) -> Optional[Dict]:
        """
        Generate probability forecast using Swarm + LLM

        Returns:
            {
                'swarm_prior': float,
                'llm_adjustment': float,
                'final_probability': float,
                'confidence': float,
                'forecast_std': float
            }
        """
        forecast = {}

        # Step 1: Swarm consensus
        if self.swarm_forecaster:
            cprint(f"[EMOJI] Running Swarm Forecaster (6 models)...", "cyan")
            swarm_result = self.swarm_forecaster.forecast(
                question=question,
                description=description,
                context=context
            )

            forecast['swarm_prior'] = swarm_result['consensus_prob']
            forecast['forecast_std'] = swarm_result['disagreement']  # Use disagreement as std
            forecast['model_probs'] = swarm_result['model_probs']
            forecast['high_disagreement'] = swarm_result['high_disagreement']

            cprint(f"   [SYM] Swarm Consensus: {format_probability(forecast['swarm_prior'])}", "green", attrs=['bold'])

            if swarm_result['high_disagreement']:
                cprint(f"   [SYM]  HIGH DISAGREEMENT detected ({swarm_result['disagreement']*100:.1f}%)", "yellow")

        else:
            cprint(f"[SYM]  Swarm forecaster disabled - using simple estimate", "yellow")
            forecast['swarm_prior'] = context.get('current_odds', {}).get('yes', 0.5)
            forecast['forecast_std'] = 0.10

        # Step 2: LLM bounded adjustment
        if self.llm_forecaster and POLYMARKET_USE_LLM_ADJUSTMENT:
            cprint(f"[EMOJI] Running LLM Forecaster (bounded adjustment)...", "cyan")
            llm_result = self.llm_forecaster.forecast(
                question=question,
                swarm_prior=forecast['swarm_prior'],
                description=description,
                context=context,
                market_id=market_id
            )

            if llm_result['accepted']:
                forecast['llm_adjustment'] = llm_result['llm_adjustment']
                forecast['final_probability'] = llm_result['final_probability']
                forecast['confidence'] = llm_result['confidence']
                forecast['reasoning'] = llm_result['reasoning']

                cprint(f"   [SYM] LLM Adjustment: {forecast['llm_adjustment']:+.3f}", "green")
                cprint(f"   [SYM] Final Forecast: {format_probability(forecast['final_probability'])}", "green", attrs=['bold'])
            else:
                cprint(f"   [SYM] LLM forecast rejected - using swarm prior", "red")
                forecast['llm_adjustment'] = 0.0
                forecast['final_probability'] = forecast['swarm_prior']
                forecast['confidence'] = 0.7
        else:
            # Use swarm consensus as final
            forecast['llm_adjustment'] = 0.0
            forecast['final_probability'] = forecast['swarm_prior']
            forecast['confidence'] = 0.75

        return forecast

    def _make_decision(
        self,
        market_id: str,
        question: str,
        true_prob: float,
        market_price: float,
        spread: float,
        liquidity: float,
        volume_24h: float,
        days_to_resolution: float,
        confidence: float,
        forecast_std: Optional[float]
    ) -> Dict:
        """
        Run quantitative analysis and make entry decision

        Returns:
            Decision dict from quant layer
        """
        decision = self.quant_layer.analyze_opportunity(
            market_id=market_id,
            question=question,
            true_prob=true_prob,
            market_price=market_price,
            spread=spread,
            liquidity=liquidity,
            volume_24h=volume_24h,
            days_to_resolution=days_to_resolution,
            confidence=confidence,
            forecast_std=forecast_std
        )

        return decision

    def _execute_trade(self, decision: Dict, current_price: float) -> Dict:
        """
        Execute trade (paper trading or real)

        Returns:
            Execution result
        """
        if POLYMARKET_PAPER_TRADING:
            cprint(f"[EMOJI] PAPER TRADING MODE", "yellow", attrs=['bold'])
            cprint(f"   Would execute: {decision['side']} @ {format_probability(current_price)}", "yellow")
            cprint(f"   Position Size: {format_currency(decision['final_position_size'])}", "yellow")

            # Open paper position
            position_id = self.exit_manager.open_position(
                market_id=decision['market_id'],
                question=decision['question'],
                side=decision['side'],
                position_size=decision['final_position_size'],
                entry_price=current_price,
                entry_ev_net=decision['ev_net'],
                entry_z_score=decision['z_score']
            )

            self.trades_today += 1

            return {
                'status': 'PAPER_TRADE',
                'position_id': position_id,
                'side': decision['side'],
                'entry_price': current_price,
                'position_size': decision['final_position_size']
            }

        else:
            cprint(f"[EMOJI] REAL TRADING MODE", "green", attrs=['bold'])
            cprint(f"   Executing: {decision['side']} @ limit {format_probability(decision['true_prob'])}", "green")
            cprint(f"   Position Size: {format_currency(decision['final_position_size'])}", "green")

            # TODO: Implement real Polymarket API execution
            # This would use the Polymarket SDK to place limit order

            return {
                'status': 'REAL_TRADE',
                'message': 'Real trading not yet implemented'
            }

    def monitor_positions(self):
        """
        Monitor all open positions and check exit rules
        """
        if self.exit_manager.open_positions.empty:
            cprint(f"\n[SYM] No open positions to monitor", "white")
            return

        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] MONITORING OPEN POSITIONS", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        positions_to_close = []

        for _, position in self.exit_manager.open_positions.iterrows():
            position_id = position['position_id']
            market_id = position['market_id']

            cprint(f"\n[EMOJI] Checking: {position_id}", "cyan")

            # Get current market data (in real system, fetch from Polymarket API)
            # For now, simulate some changes
            current_price = position['current_price'] * (1 + (0.02 if position['side'] == 'YES' else -0.02))
            current_ev_net = position['current_ev_net'] * 0.8  # Simulate EV decay
            current_z_score = position['current_z_score'] * 0.9  # Simulate z-score reversion

            # Check exit rules
            check_result = self.exit_manager.check_position(
                position_id=position_id,
                current_price=current_price,
                current_ev_net=current_ev_net,
                current_z_score=current_z_score,
                bullish_signals_count=2,
                bearish_signals_count=1
            )

            if check_result['should_exit']:
                positions_to_close.append({
                    'position_id': position_id,
                    'exit_price': current_price,
                    'exit_rules': check_result['exit_rules_triggered']
                })

        # Close positions that triggered exits
        for exit_info in positions_to_close:
            closed = self.exit_manager.close_position(
                position_id=exit_info['position_id'],
                exit_price=exit_info['exit_price'],
                exit_rules_triggered=exit_info['exit_rules']
            )

            # Update daily PnL
            self.daily_pnl += closed.get('pnl_usd', 0)

    def check_risk_limits(self) -> bool:
        """
        Check portfolio-level risk limits

        Returns:
            True if safe to continue, False if limits breached
        """
        # Max positions
        if len(self.exit_manager.open_positions) >= POLYMARKET_MAX_OPEN_POSITIONS:
            cprint(f"[SYM]  Max open positions reached ({POLYMARKET_MAX_OPEN_POSITIONS})", "red")
            return False

        # Max daily trades
        if self.trades_today >= POLYMARKET_MAX_DAILY_TRADES:
            cprint(f"[SYM]  Max daily trades reached ({POLYMARKET_MAX_DAILY_TRADES})", "red")
            return False

        # Max daily loss
        if self.daily_pnl <= -POLYMARKET_MAX_DAILY_LOSS_USD:
            cprint(f"[EMOJI] DAILY LOSS LIMIT BREACHED: {format_currency(self.daily_pnl)}", "red", attrs=['bold'])
            return False

        # Min account balance (would check real balance in production)
        # For now, simulate based on portfolio value
        simulated_balance = self.portfolio_value + self.daily_pnl
        if simulated_balance < POLYMARKET_MIN_ACCOUNT_BALANCE_USD:
            cprint(f"[EMOJI] ACCOUNT BALANCE TOO LOW: {format_currency(simulated_balance)}", "red", attrs=['bold'])
            return False

        return True

    def print_session_summary(self):
        """Print session summary"""

        uptime = datetime.now() - self.session_start_time
        uptime_str = str(uptime).split('.')[0]

        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] SESSION SUMMARY", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        cprint(f"\n[TIMER] RUNTIME:", "yellow")
        cprint(f"   Uptime: {uptime_str}", "cyan")

        cprint(f"\n[EMOJI] PORTFOLIO:", "yellow")
        cprint(f"   Starting Value: {format_currency(self.portfolio_value)}", "cyan")
        pnl_sign = "+" if self.daily_pnl > 0 else ""
        cprint(f"   Daily PnL: {pnl_sign}{format_currency(abs(self.daily_pnl))}", "green" if self.daily_pnl > 0 else "red")

        cprint(f"\n[EMOJI] TRADING:", "yellow")
        cprint(f"   Trades Today: {self.trades_today}", "cyan")
        cprint(f"   Open Positions: {len(self.exit_manager.open_positions)}", "green")

        # Get performance stats
        stats = self.exit_manager.get_performance_statistics()
        if stats['total_trades'] > 0:
            cprint(f"\n[EMOJI] PERFORMANCE:", "yellow")
            cprint(f"   Total Trades: {stats['total_trades']}", "cyan")
            cprint(f"   Win Rate: {stats['win_rate']:.1%}", "green" if stats['win_rate'] > 0.5 else "red")
            cprint(f"   Total PnL: {format_currency(stats['total_pnl_usd']):+}", "green" if stats['total_pnl_usd'] > 0 else "red", attrs=['bold'])

        cprint(f"\n{'='*80}\n", "cyan")


def main():
    """Run Polymarket Orchestrator"""

    cprint(f"\n{'='*80}", "magenta")
    cprint(f"[EMOJI] POLYMARKET ORCHESTRATOR - DEMONSTRATION", "magenta", attrs=['bold'])
    cprint(f"{'='*80}\n", "magenta")

    # Initialize orchestrator
    orchestrator = PolymarketOrchestrator(portfolio_value=10000)

    # Analyze a test market
    result = orchestrator.analyze_market(
        market_id='test_btc_100k',
        question='Will Bitcoin hit $100k by end of 2024?',
        description='Market resolves YES if Bitcoin closes above $100,000 on December 31, 2024.',
        current_yes_price=0.42,
        current_no_price=0.58,
        volume_24h=125000,
        liquidity=450000,
        spread=0.04,
        days_to_resolution=60
    )

    # Monitor positions
    orchestrator.monitor_positions()

    # Print summary
    orchestrator.print_session_summary()

    cprint(f"\n[SYM] Orchestrator Demonstration Complete!\n", "green", attrs=['bold'])


if __name__ == "__main__":
    if not POLYMARKET_ENABLED:
        cprint("\n[SYM]  Polymarket trading is DISABLED in config.py", "red", attrs=['bold'])
        cprint("Set POLYMARKET_ENABLED = True to activate\n", "yellow")
    else:
        main()
