"""
[EMOJI] Moon Dev's Polymarket LLM Forecaster
Bounded probability adjustment with explicit reasoning
Built with love by Moon Dev [EMOJI]
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from termcolor import cprint

# Add project root to path
import sys
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from src.models.model_factory import ModelFactory
from src.polymarket_utils import PolymarketUtils, format_probability
from src.config import (
    POLYMARKET_LLM_BASE_MODEL,
    POLYMARKET_LLM_MAX_ADJUSTMENT,
    POLYMARKET_LLM_MIN_CONFIDENCE,
    POLYMARKET_LLM_REASONING_REQUIRED,
    POLYMARKET_DATA_DIR,
    POLYMARKET_VERBOSE_LOGGING
)

model_factory = ModelFactory()


class PolymarketLLMForecaster:
    """
    Polymarket LLM Forecaster - Bounded Probability Adjustment

    Takes swarm consensus as prior and makes bounded adjustments with
    explicit reasoning. Prevents overconfidence by limiting adjustments.

    Key Features:
    1. Anchored to swarm consensus (multi-model prior)
    2. Bounded adjustment (±10-15% maximum)
    3. Explicit reasoning required
    4. Confidence threshold (≥0.6)
    5. Prevents extreme predictions

    Formula: final_prob = swarm_prior + bounded_adjustment
    Where: |adjustment| ≤ MAX_ADJUSTMENT
    """

    def __init__(self):
        """Initialize LLM Forecaster"""
        self.data_dir = Path(POLYMARKET_DATA_DIR) / 'llm_forecasts'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.utils = PolymarketUtils()

        # Data file
        self.forecasts_file = self.data_dir / 'llm_forecasts.csv'
        self.forecasts = self._load_or_create_forecasts()

        # Initialize LLM
        self.model = model_factory.get_model(POLYMARKET_LLM_BASE_MODEL)

        if not self.model:
            cprint(f"[SYM] Failed to load model: {POLYMARKET_LLM_BASE_MODEL}", "red")

        cprint(f"\n[EMOJI] Polymarket LLM Forecaster Initialized", "cyan", attrs=['bold'])
        cprint(f"[EMOJI] Model: {POLYMARKET_LLM_BASE_MODEL}", "green")
        cprint(f"[EMOJI] Max Adjustment: ±{POLYMARKET_LLM_MAX_ADJUSTMENT * 100:.0f}%", "cyan")
        cprint(f"[EMOJI] Min Confidence: {POLYMARKET_LLM_MIN_CONFIDENCE}", "cyan")
        cprint(f"[EMOJI] Historical Forecasts: {len(self.forecasts)}", "green")

    def _load_or_create_forecasts(self) -> pd.DataFrame:
        """Load or create forecasts database"""
        if self.forecasts_file.exists():
            df = pd.read_csv(self.forecasts_file)
            cprint(f"[SYM] Loaded {len(df)} historical forecasts", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'market_id',
                'question',
                'swarm_prior',
                'llm_adjustment',
                'final_probability',
                'confidence',
                'reasoning',
                'model_used',
                'accepted'  # True if confidence met threshold
            ])
            cprint(f"[EMOJI] Created new forecasts database", "yellow")
            return df

    def forecast(
        self,
        question: str,
        swarm_prior: float,
        description: str = "",
        context: Optional[Dict] = None,
        market_id: Optional[str] = None
    ) -> Dict:
        """
        Generate bounded forecast adjustment

        Args:
            question: Polymarket question
            swarm_prior: Swarm consensus probability (0-1)
            description: Market description
            context: Additional context dict
            market_id: Optional market ID

        Returns:
            {
                'swarm_prior': float,
                'llm_adjustment': float,
                'final_probability': float,
                'confidence': float,
                'reasoning': str,
                'accepted': bool,
                'rejection_reason': str (if rejected)
            }
        """
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] LLM BOUNDED FORECAST", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")
        cprint(f"[SYM] Question: {question}", "white", attrs=['bold'])
        cprint(f"[EMOJI] Swarm Prior: {format_probability(swarm_prior)}", "blue", attrs=['bold'])

        if description:
            cprint(f"[EMOJI] Description: {description[:100]}{'...' if len(description) > 100 else ''}", "cyan")

        # Build prompt
        prompt = self._build_adjustment_prompt(
            question=question,
            swarm_prior=swarm_prior,
            description=description,
            context=context
        )

        # Get LLM response
        try:
            system_prompt = """You are a professional forecaster providing bounded probability adjustments.
You must be calibrated and provide explicit reasoning for your adjustments.
NEVER make extreme adjustments - stay anchored to the prior."""

            response = self.model.generate_response(
                system_prompt=system_prompt,
                user_content=prompt,
                temperature=0.7,
                max_tokens=600
            )

            # Extract text
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            if POLYMARKET_VERBOSE_LOGGING:
                cprint(f"\n[EMOJI] LLM Response:\n{response_text}\n", "blue")

            # Parse response
            parsed = self._parse_llm_response(response_text, swarm_prior)

            # Validate and bound adjustment
            result = self._validate_and_bound(parsed, swarm_prior)

            # Store forecast
            self._store_forecast(
                market_id=market_id or 'unknown',
                question=question,
                swarm_prior=swarm_prior,
                result=result
            )

            # Print result
            self._print_result(result)

            return result

        except Exception as e:
            cprint(f"[SYM] LLM Forecasting Error: {e}", "red")
            return {
                'swarm_prior': swarm_prior,
                'llm_adjustment': 0.0,
                'final_probability': swarm_prior,
                'confidence': 0.0,
                'reasoning': f"Error: {str(e)}",
                'accepted': False,
                'rejection_reason': 'LLM error'
            }

    def _build_adjustment_prompt(
        self,
        question: str,
        swarm_prior: float,
        description: str,
        context: Optional[Dict]
    ) -> str:
        """Build prompt for LLM adjustment"""

        prompt = f"""You are adjusting a probability forecast based on a multi-model consensus.

QUESTION:
{question}

SWARM CONSENSUS (6-MODEL PRIOR):
The consensus from 6 LLMs (Claude, GPT-5, DeepSeek, Grok, Gemini, Ollama) is: {format_probability(swarm_prior)}

"""

        if description:
            prompt += f"""DESCRIPTION:
{description}

"""

        if context:
            prompt += "ADDITIONAL CONTEXT:\n"
            if 'current_odds' in context:
                prompt += f"- Current Market Odds: {format_probability(context['current_odds'].get('yes', 0.5))}\n"
            if 'volume_24h' in context:
                prompt += f"- 24h Volume: ${context['volume_24h']:,.0f}\n"
            if 'liquidity' in context:
                prompt += f"- Liquidity: ${context['liquidity']:,.0f}\n"
            if 'days_to_resolution' in context:
                prompt += f"- Days to Resolution: {context['days_to_resolution']:.1f}\n"
            prompt += "\n"

        prompt += f"""YOUR TASK:
Provide a BOUNDED adjustment to the swarm consensus.

CRITICAL RULES:
1. You MUST stay within ±{POLYMARKET_LLM_MAX_ADJUSTMENT * 100:.0f}% of the swarm prior
2. The swarm prior is {format_probability(swarm_prior)} - respect this multi-model consensus
3. Only adjust if you have STRONG reasoning
4. Small adjustments (±5-10%) are preferred
5. Express your confidence (0.0 to 1.0)

OUTPUT FORMAT (REQUIRED):
ADJUSTMENT: +0.XX or -0.XX
CONFIDENCE: 0.X
REASONING:
[Your detailed reasoning explaining why you're adjusting up or down from the swarm consensus]

EXAMPLE:
ADJUSTMENT: +0.08
CONFIDENCE: 0.75
REASONING: The swarm consensus of 65% seems slightly conservative given recent polling data showing a 3-point shift in favor of YES. However, the swarm correctly captures the base rate uncertainty, so I'm only adjusting up by 8% rather than making a dramatic change.

Now provide your bounded adjustment:
"""

        return prompt

    def _parse_llm_response(self, response_text: str, swarm_prior: float) -> Dict:
        """
        Parse LLM response for adjustment, confidence, and reasoning

        Returns:
            {adjustment, confidence, reasoning}
        """
        # Extract ADJUSTMENT
        adj_match = re.search(r'ADJUSTMENT:\s*([\+\-]?[0-9.]+)', response_text, re.IGNORECASE)
        if adj_match:
            adjustment = float(adj_match.group(1))
        else:
            # Try to find percentage format
            adj_match = re.search(r'ADJUSTMENT:\s*([\+\-]?[0-9.]+)\s*%', response_text, re.IGNORECASE)
            if adj_match:
                adjustment = float(adj_match.group(1)) / 100
            else:
                cprint("[SYM] Could not parse ADJUSTMENT, using 0", "yellow")
                adjustment = 0.0

        # Extract CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response_text, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))
            # Normalize to 0-1 if given as percentage
            if confidence > 1:
                confidence = confidence / 100
        else:
            cprint("[SYM] Could not parse CONFIDENCE, using 0.5", "yellow")
            confidence = 0.5

        # Extract REASONING
        reasoning_match = re.search(r'REASONING:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            # Truncate to first 500 chars
            reasoning = reasoning[:500]
        else:
            reasoning = "No explicit reasoning provided"

        return {
            'adjustment': adjustment,
            'confidence': confidence,
            'reasoning': reasoning
        }

    def _validate_and_bound(self, parsed: Dict, swarm_prior: float) -> Dict:
        """
        Validate and bound the LLM adjustment

        Returns:
            Complete result dict
        """
        adjustment = parsed['adjustment']
        confidence = parsed['confidence']
        reasoning = parsed['reasoning']

        # Bound adjustment to max allowed
        original_adjustment = adjustment
        adjustment = max(-POLYMARKET_LLM_MAX_ADJUSTMENT, min(POLYMARKET_LLM_MAX_ADJUSTMENT, adjustment))

        if abs(original_adjustment) > POLYMARKET_LLM_MAX_ADJUSTMENT:
            cprint(f"[SYM]  Adjustment clamped: {original_adjustment:+.3f} → {adjustment:+.3f}", "yellow")

        # Calculate final probability
        final_probability = swarm_prior + adjustment

        # Clamp to [0, 1]
        final_probability = max(0.0, min(1.0, final_probability))

        # Check confidence threshold
        accepted = confidence >= POLYMARKET_LLM_MIN_CONFIDENCE

        rejection_reason = None
        if not accepted:
            rejection_reason = f"Confidence {confidence:.2f} below threshold {POLYMARKET_LLM_MIN_CONFIDENCE}"
            cprint(f"[SYM] FORECAST REJECTED: {rejection_reason}", "red")

        # Check reasoning requirement
        if POLYMARKET_LLM_REASONING_REQUIRED and len(reasoning) < 50:
            accepted = False
            rejection_reason = "Insufficient reasoning provided"
            cprint(f"[SYM] FORECAST REJECTED: {rejection_reason}", "red")

        result = {
            'swarm_prior': swarm_prior,
            'llm_adjustment': adjustment,
            'final_probability': final_probability,
            'confidence': confidence,
            'reasoning': reasoning,
            'accepted': accepted,
            'rejection_reason': rejection_reason
        }

        return result

    def _store_forecast(
        self,
        market_id: str,
        question: str,
        swarm_prior: float,
        result: Dict
    ):
        """Store forecast to database"""

        forecast_data = {
            'timestamp': datetime.now(),
            'market_id': market_id,
            'question': question,
            'swarm_prior': swarm_prior,
            'llm_adjustment': result['llm_adjustment'],
            'final_probability': result['final_probability'],
            'confidence': result['confidence'],
            'reasoning': result['reasoning'],
            'model_used': POLYMARKET_LLM_BASE_MODEL,
            'accepted': result['accepted']
        }

        self.forecasts = pd.concat([
            self.forecasts,
            pd.DataFrame([forecast_data])
        ], ignore_index=True)

        # Save
        self.forecasts.to_csv(self.forecasts_file, index=False)

        if POLYMARKET_VERBOSE_LOGGING:
            cprint(f"[EMOJI] Forecast saved", "blue")

    def _print_result(self, result: Dict):
        """Print forecast result"""

        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] FORECAST RESULT", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        cprint(f"\n[EMOJI] Swarm Prior:       {format_probability(result['swarm_prior'])}", "blue")
        cprint(f"[EMOJI] LLM Adjustment:    {result['llm_adjustment']:+.3f} ({result['llm_adjustment']*100:+.1f}%)", "yellow")
        cprint(f"[EMOJI] Final Probability: {format_probability(result['final_probability'])}", "green", attrs=['bold'])
        cprint(f"[EMOJI] Confidence:        {result['confidence']:.2f}", "cyan")

        if result['accepted']:
            cprint(f"\n[SYM] FORECAST ACCEPTED", "green", attrs=['bold'])
        else:
            cprint(f"\n[SYM] FORECAST REJECTED: {result['rejection_reason']}", "red", attrs=['bold'])

        cprint(f"\n[EMOJI] REASONING:", "yellow")
        cprint(f"{result['reasoning']}", "white")

        cprint(f"\n{'='*80}\n", "cyan")

    def get_forecast_statistics(self) -> Dict:
        """Get statistics on historical forecasts"""

        if self.forecasts.empty:
            return {
                'total_forecasts': 0,
                'accepted_forecasts': 0,
                'acceptance_rate': 0.0
            }

        accepted = self.forecasts[self.forecasts['accepted'] == True]

        stats = {
            'total_forecasts': len(self.forecasts),
            'accepted_forecasts': len(accepted),
            'acceptance_rate': len(accepted) / len(self.forecasts) if len(self.forecasts) > 0 else 0,
            'avg_confidence': self.forecasts['confidence'].mean(),
            'avg_adjustment': self.forecasts['llm_adjustment'].mean(),
            'avg_abs_adjustment': self.forecasts['llm_adjustment'].abs().mean(),
            'positive_adjustments': len(self.forecasts[self.forecasts['llm_adjustment'] > 0]),
            'negative_adjustments': len(self.forecasts[self.forecasts['llm_adjustment'] < 0])
        }

        return stats

    def print_summary(self):
        """Print forecaster summary"""

        stats = self.get_forecast_statistics()

        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] LLM FORECASTER SUMMARY", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        cprint(f"\n[EMOJI] FORECAST STATISTICS:", "yellow")
        cprint(f"   Total Forecasts: {stats['total_forecasts']}", "cyan")
        cprint(f"   Accepted: {stats['accepted_forecasts']}", "green")
        cprint(f"   Acceptance Rate: {stats['acceptance_rate']:.1%}", "cyan")

        if stats['total_forecasts'] > 0:
            cprint(f"\n[EMOJI] ADJUSTMENT PATTERNS:", "yellow")
            cprint(f"   Avg Confidence: {stats['avg_confidence']:.2f}", "cyan")
            cprint(f"   Avg Adjustment: {stats['avg_adjustment']:+.3f}", "cyan")
            cprint(f"   Avg Absolute Adjustment: {stats['avg_abs_adjustment']:.3f}", "cyan")
            cprint(f"   Positive Adjustments: {stats['positive_adjustments']}", "green")
            cprint(f"   Negative Adjustments: {stats['negative_adjustments']}", "red")

        cprint(f"\n[SYM]  CONFIGURATION:", "yellow")
        cprint(f"   Model: {POLYMARKET_LLM_BASE_MODEL}", "cyan")
        cprint(f"   Max Adjustment: ±{POLYMARKET_LLM_MAX_ADJUSTMENT * 100:.0f}%", "cyan")
        cprint(f"   Min Confidence: {POLYMARKET_LLM_MIN_CONFIDENCE}", "cyan")

        cprint(f"\n{'='*80}\n", "cyan")


def main():
    """Test LLM Forecaster"""

    forecaster = PolymarketLLMForecaster()

    # Example 1: Adjust swarm consensus upward
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 1: Bounded Adjustment (Upward)", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    result1 = forecaster.forecast(
        question="Will Bitcoin hit $100k by EOY 2024?",
        swarm_prior=0.42,  # 42% from swarm
        description="Market resolves YES if Bitcoin closes above $100,000 on December 31, 2024.",
        context={
            'current_odds': {'yes': 0.45, 'no': 0.55},
            'volume_24h': 125000,
            'liquidity': 350000,
            'days_to_resolution': 60
        },
        market_id='test_btc_100k'
    )

    # Example 2: Adjust swarm consensus downward
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 2: Bounded Adjustment (Downward)", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    result2 = forecaster.forecast(
        question="Will Trump win the 2024 election?",
        swarm_prior=0.68,  # 68% from swarm
        description="Market resolves YES if Trump wins electoral college majority.",
        context={
            'current_odds': {'yes': 0.65, 'no': 0.35},
            'volume_24h': 500000,
            'liquidity': 1200000,
            'days_to_resolution': 30
        },
        market_id='test_trump_2024'
    )

    # Example 3: Small adjustment with high confidence
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 3: Small Adjustment (Anchored)", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    result3 = forecaster.forecast(
        question="Will the Federal Reserve cut rates in March 2024?",
        swarm_prior=0.55,
        description="Market resolves YES if Fed cuts rates by ≥25 basis points in March.",
        market_id='test_fed_march'
    )

    # Print summary
    forecaster.print_summary()

    cprint("\n[SYM] LLM Forecaster Tests Complete!\n", "green", attrs=['bold'])


if __name__ == "__main__":
    main()
