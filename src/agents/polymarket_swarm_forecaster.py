"""
[EMOJI] Moon Dev's Polymarket Swarm Forecaster
6-model parallel consensus for probability forecasting
Built with love by Moon Dev [EMOJI]
"""

import json
import time
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from termcolor import cprint

# Add project root to path
import sys
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from src.models.model_factory import ModelFactory
from src.polymarket_utils import PolymarketUtils, format_probability
from src.config import (
    POLYMARKET_SWARM_MODELS,
    POLYMARKET_SWARM_TIMEOUT_SEC,
    POLYMARKET_SWARM_CONSENSUS_MIN,
    POLYMARKET_SWARM_DISAGREEMENT_THRESHOLD,
    POLYMARKET_DATA_DIR,
    POLYMARKET_VERBOSE_LOGGING
)

model_factory = ModelFactory()


class PolymarketSwarmForecaster:
    """
    Polymarket Swarm Forecaster - 6-Model Probability Consensus

    Queries 6 LLMs in parallel to forecast probabilities for Polymarket questions.
    Uses statistical consensus (median) and flags high disagreement.

    Architecture:
    - Parallel execution (ThreadPoolExecutor)
    - Timeout handling (45 seconds per model)
    - Numerical probability extraction
    - Statistical consensus (median)
    - Disagreement detection (spread ≥15%)
    """

    def __init__(self):
        """Initialize Swarm Forecaster with configured models"""
        self.results_dir = Path(POLYMARKET_DATA_DIR) / 'swarm_forecasts'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize active models from config
        self.active_models = self._initialize_models()

        cprint(f"\n[EMOJI] Polymarket Swarm Forecaster Initialized", "cyan", attrs=['bold'])
        cprint(f"[EMOJI] Active Models: {len(self.active_models)}", "green")
        for provider in self.active_models.keys():
            cprint(f"   -> {provider.upper()}", "cyan")

    def _initialize_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize models from config

        Returns:
            Dict of active models {provider: {name, instance}}
        """
        active = {}

        # Model name mapping
        model_names = {
            'claude': 'claude-sonnet-4-5',        # Claude 4.5 Sonnet - Balanced
            'openai': 'gpt-5',                    # GPT-5 - Most advanced
            'deepseek': 'deepseek-chat',          # DeepSeek Chat - Cheapest
            'groq': 'llama-3.1-8b-instant',       # Llama 3.1 8B - Cheapest Groq model
            'gemini': 'gemini-2.5-flash',         # Gemini 2.5 Flash - Cheapest
            'xai': 'grok-4-fast-reasoning',       # Grok-4 Fast Reasoning (disabled)
            'ollama': 'DeepSeek-R1:latest'        # Local DeepSeek-R1 (disabled)
        }

        for provider, enabled in POLYMARKET_SWARM_MODELS.items():
            if enabled:
                try:
                    model = model_factory.get_model(provider, model_names.get(provider))
                    if model:
                        active[provider] = {
                            'name': model_names.get(provider, 'default'),
                            'instance': model
                        }
                        if POLYMARKET_VERBOSE_LOGGING:
                            cprint(f"[SYM] Loaded {provider}: {model_names.get(provider)}", "green")
                except Exception as e:
                    cprint(f"[SYM] Failed to load {provider}: {e}", "yellow")

        if len(active) < 2:
            cprint(f"[SYM] Warning: Only {len(active)} models active. Need at least 2 for consensus.", "red")

        return active

    def forecast(
        self,
        question: str,
        description: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate probability forecast using swarm consensus

        Args:
            question: Polymarket question (e.g., "Will Trump win 2024 election?")
            description: Additional market description/details
            context: Optional context dict (volume, liquidity, current_odds, etc.)

        Returns:
            {
                'consensus_prob': float,  # Median probability (0-1)
                'model_probs': dict,      # Individual model probabilities
                'disagreement': float,    # Spread (max - min)
                'high_disagreement': bool, # True if spread ≥ threshold
                'reasoning': dict,        # Individual model reasoning
                'metadata': dict          # Timing, success rates, etc.
            }
        """
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] SWARM FORECAST", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")
        cprint(f"[SYM] Question: {question}", "white", attrs=['bold'])

        if description:
            cprint(f"[EMOJI] Description: {description[:150]}{'...' if len(description) > 150 else ''}", "blue")

        # Build comprehensive prompt
        prompt = self._build_forecast_prompt(question, description, context)

        # Query all models in parallel
        start_time = time.time()
        all_responses = self._query_swarm(prompt)

        # Parse probabilities from responses
        model_probs, reasoning = self._parse_probabilities(all_responses)

        # Calculate consensus and disagreement
        consensus_result = self._calculate_consensus(model_probs)

        # Build result
        result = {
            'question': question,
            'description': description,
            'consensus_prob': consensus_result['consensus'],
            'model_probs': model_probs,
            'disagreement': consensus_result['disagreement'],
            'high_disagreement': consensus_result['high_disagreement'],
            'reasoning': reasoning,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_models': len(self.active_models),
                'successful_responses': len(model_probs),
                'failed_responses': len(self.active_models) - len(model_probs),
                'total_time': round(time.time() - start_time, 2),
                'disagreement_threshold': POLYMARKET_SWARM_DISAGREEMENT_THRESHOLD
            }
        }

        # Save results
        self._save_forecast(result)

        # Print summary
        self._print_summary(result)

        return result

    def _build_forecast_prompt(
        self,
        question: str,
        description: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build comprehensive forecast prompt

        Returns structured prompt for probability forecasting
        """
        prompt = f"""You are a professional forecaster predicting probabilities for prediction markets.

QUESTION:
{question}

"""

        if description:
            prompt += f"""DESCRIPTION:
{description}

"""

        if context:
            prompt += "MARKET CONTEXT:\n"
            if 'current_odds' in context:
                prompt += f"- Current Market Odds: YES={format_probability(context['current_odds'].get('yes', 0.5))}, NO={format_probability(context['current_odds'].get('no', 0.5))}\n"
            if 'volume_24h' in context:
                prompt += f"- 24h Volume: ${context['volume_24h']:,.0f}\n"
            if 'liquidity' in context:
                prompt += f"- Liquidity: ${context['liquidity']:,.0f}\n"
            if 'days_to_resolution' in context:
                prompt += f"- Days to Resolution: {context['days_to_resolution']:.1f}\n"
            prompt += "\n"

        prompt += """YOUR TASK:
Provide your best probability estimate for the YES outcome (0.00 to 1.00).

IMPORTANT GUIDELINES:
1. Output your probability first on its own line as: PROBABILITY: 0.XX
2. Then provide your reasoning
3. Consider base rates, recent evidence, and time to resolution
4. Be calibrated - avoid extreme probabilities unless highly confident
5. Ignore current market odds - provide YOUR independent estimate

EXAMPLE OUTPUT FORMAT:
PROBABILITY: 0.67

[Your reasoning here explaining why you believe there is a 67% chance of YES]

Now provide your forecast:
"""

        return prompt

    def _query_swarm(self, prompt: str) -> Dict[str, Dict[str, Any]]:
        """
        Query all models in parallel

        Args:
            prompt: Forecast prompt

        Returns:
            Dict of responses {provider: {success, response, error, time}}
        """
        cprint(f"\n[EMOJI] Querying {len(self.active_models)} models in parallel...", "yellow", attrs=['bold'])

        all_responses = {}

        with ThreadPoolExecutor(max_workers=len(self.active_models)) as executor:
            # Submit all queries
            futures = {
                executor.submit(
                    self._query_single_model,
                    provider,
                    model_info,
                    prompt
                ): provider
                for provider, model_info in self.active_models.items()
            }

            # Collect results as they complete
            completed = 0
            total = len(futures)

            try:
                for future in as_completed(futures, timeout=POLYMARKET_SWARM_TIMEOUT_SEC + 10):
                    provider = futures[future]
                    completed += 1

                    cprint(f"[TIMER] {completed}/{total} completed...", "yellow")

                    try:
                        result = future.result(timeout=5)
                        all_responses[provider] = result

                        if result['success']:
                            cprint(f"[SYM] {provider.upper()}: {result['response_time']:.2f}s", "green")
                        else:
                            cprint(f"[SYM] {provider.upper()}: {result['error']}", "red")

                    except Exception as e:
                        cprint(f"[EMOJI] {provider.upper()} error: {e}", "red")
                        all_responses[provider] = {
                            'success': False,
                            'response': None,
                            'error': str(e),
                            'response_time': 0
                        }

            except TimeoutError:
                cprint(f"[ALARM] Overall timeout reached", "yellow")
                # Mark missing responses as timeout
                for future, provider in futures.items():
                    if provider not in all_responses:
                        all_responses[provider] = {
                            'success': False,
                            'response': None,
                            'error': 'Global timeout',
                            'response_time': POLYMARKET_SWARM_TIMEOUT_SEC
                        }

        return all_responses

    def _query_single_model(
        self,
        provider: str,
        model_info: Dict[str, Any],
        prompt: str
    ) -> Dict[str, Any]:
        """
        Query a single model with timeout handling

        Args:
            provider: Provider name (claude, openai, etc.)
            model_info: Model configuration dict
            prompt: Forecast prompt

        Returns:
            {success, response, error, response_time}
        """
        start = time.time()

        try:
            model = model_info['instance']

            system_prompt = "You are a professional forecaster. Provide calibrated probability estimates with clear reasoning."

            response = model.generate_response(
                system_prompt=system_prompt,
                user_content=prompt,
                temperature=0.7,
                max_tokens=800
            )

            # Extract text from response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            elapsed = time.time() - start

            return {
                'success': True,
                'response': response_text,
                'error': None,
                'response_time': round(elapsed, 2)
            }

        except Exception as e:
            elapsed = time.time() - start
            return {
                'success': False,
                'response': None,
                'error': str(e),
                'response_time': round(elapsed, 2)
            }

    def _parse_probabilities(
        self,
        responses: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Extract probabilities and reasoning from model responses

        Args:
            responses: Raw model responses

        Returns:
            Tuple of (model_probs, reasoning)
            - model_probs: {provider: probability}
            - reasoning: {provider: reasoning_text}
        """
        model_probs = {}
        reasoning = {}

        for provider, data in responses.items():
            if not data['success'] or not data['response']:
                continue

            response_text = data['response']

            # Extract probability using regex
            # Look for patterns like: PROBABILITY: 0.67 or PROBABILITY: 67%
            prob = self._extract_probability(response_text)

            if prob is not None:
                model_probs[provider] = prob
                reasoning[provider] = response_text

                if POLYMARKET_VERBOSE_LOGGING:
                    cprint(f"[EMOJI] {provider.upper()}: {format_probability(prob)}", "cyan")
            else:
                cprint(f"[SYM] {provider.upper()}: Could not parse probability", "yellow")

        return model_probs, reasoning

    def _extract_probability(self, text: str) -> Optional[float]:
        """
        Extract probability from model response text

        Looks for patterns like:
        - PROBABILITY: 0.67
        - PROBABILITY: 67%
        - My estimate is 0.67
        - I estimate 67%

        Returns:
            Probability as float (0-1) or None if not found
        """
        # Pattern 1: PROBABILITY: 0.XX or PROBABILITY: XX%
        match = re.search(r'PROBABILITY:\s*([0-9.]+)%?', text, re.IGNORECASE)
        if match:
            prob = float(match.group(1))
            # Convert percentage to decimal if needed
            if prob > 1:
                prob = prob / 100
            return max(0.0, min(1.0, prob))  # Clamp to [0, 1]

        # Pattern 2: estimate is X.XX or estimate XX%
        match = re.search(r'estimate\s+(?:is\s+)?([0-9.]+)%?', text, re.IGNORECASE)
        if match:
            prob = float(match.group(1))
            if prob > 1:
                prob = prob / 100
            return max(0.0, min(1.0, prob))

        # Pattern 3: probability of X.XX or probability XX%
        match = re.search(r'probability\s+of\s+([0-9.]+)%?', text, re.IGNORECASE)
        if match:
            prob = float(match.group(1))
            if prob > 1:
                prob = prob / 100
            return max(0.0, min(1.0, prob))

        # Pattern 4: Standalone percentage in first 200 chars
        # (common in direct responses)
        first_part = text[:200]
        match = re.search(r'\b([0-9]{1,2}(?:\.[0-9]+)?)\s*%', first_part)
        if match:
            prob = float(match.group(1)) / 100
            return max(0.0, min(1.0, prob))

        # Pattern 5: Decimal probability 0.XX in first 200 chars
        match = re.search(r'\b(0\.[0-9]+)\b', first_part)
        if match:
            prob = float(match.group(1))
            return max(0.0, min(1.0, prob))

        return None

    def _calculate_consensus(self, model_probs: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate consensus probability and disagreement metrics

        Args:
            model_probs: {provider: probability}

        Returns:
            {consensus, disagreement, high_disagreement}
        """
        if not model_probs:
            return {
                'consensus': 0.5,
                'disagreement': 0.0,
                'high_disagreement': True
            }

        probs = list(model_probs.values())

        # Consensus = median
        consensus = float(np.median(probs))

        # Disagreement = spread (max - min)
        disagreement = float(np.max(probs) - np.min(probs))

        # Flag if disagreement exceeds threshold
        high_disagreement = disagreement >= POLYMARKET_SWARM_DISAGREEMENT_THRESHOLD

        return {
            'consensus': consensus,
            'disagreement': disagreement,
            'high_disagreement': high_disagreement
        }

    def _save_forecast(self, result: Dict[str, Any]):
        """Save forecast to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"swarm_forecast_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        if POLYMARKET_VERBOSE_LOGGING:
            cprint(f"\n[EMOJI] Forecast saved: {filename.name}", "blue")

    def _print_summary(self, result: Dict[str, Any]):
        """Print forecast summary"""
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] SWARM FORECAST RESULTS", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        # Consensus
        consensus = result['consensus_prob']
        cprint(f"\n[EMOJI] CONSENSUS: {format_probability(consensus)}", "green", attrs=['bold'])

        # Individual models
        cprint(f"\n[EMOJI] INDIVIDUAL FORECASTS:", "yellow")
        for provider, prob in sorted(result['model_probs'].items(), key=lambda x: x[1], reverse=True):
            cprint(f"   {provider.upper():12s}: {format_probability(prob)}", "cyan")

        # Disagreement
        disagreement = result['disagreement']
        disagreement_pct = disagreement * 100

        if result['high_disagreement']:
            cprint(f"\n[SYM]  HIGH DISAGREEMENT: {disagreement_pct:.1f}% spread (threshold: {POLYMARKET_SWARM_DISAGREEMENT_THRESHOLD*100:.0f}%)", "red", attrs=['bold'])
        else:
            cprint(f"\n[SYM] Low Disagreement: {disagreement_pct:.1f}% spread", "green")

        # Metadata
        metadata = result['metadata']
        cprint(f"\n[EMOJI] METADATA:", "blue")
        cprint(f"   Successful Models: {metadata['successful_responses']}/{metadata['total_models']}", "cyan")
        cprint(f"   Total Time: {metadata['total_time']:.2f}s", "cyan")
        cprint(f"   Timestamp: {metadata['timestamp']}", "cyan")

        cprint(f"\n{'='*80}\n", "cyan")


def main():
    """Test Swarm Forecaster with example questions"""

    forecaster = PolymarketSwarmForecaster()

    # Example 1: Simple political question
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 1: Political Question", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    result1 = forecaster.forecast(
        question="Will Donald Trump win the 2024 US Presidential Election?",
        description="Market resolves YES if Trump wins the electoral college majority in November 2024.",
        context={
            'current_odds': {'yes': 0.55, 'no': 0.45},
            'volume_24h': 125000,
            'liquidity': 450000,
            'days_to_resolution': 45
        }
    )

    # Example 2: Technology question
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 2: Technology Question", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    result2 = forecaster.forecast(
        question="Will Tesla stock close above $300 on December 31, 2024?",
        description="Market resolves YES if TSLA closing price on Dec 31, 2024 is above $300.00.",
        context={
            'current_odds': {'yes': 0.42, 'no': 0.58},
            'volume_24h': 85000,
            'liquidity': 220000,
            'days_to_resolution': 90
        }
    )

    cprint("\n[SYM] Swarm Forecaster Tests Complete!\n", "green", attrs=['bold'])


if __name__ == "__main__":
    main()
