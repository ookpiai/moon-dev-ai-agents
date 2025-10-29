"""
[EMOJI] Moon Dev's Polymarket Event Catalyst Agent
Monitor breaking news and events that drive prediction market movements
Built with love by Moon Dev [EMOJI]
"""

import pandas as pd
import numpy as np
import time
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from termcolor import cprint

# Sentiment Analysis
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    cprint("[SYM] transformers not available - install with: pip install transformers torch", "yellow")
    FINBERT_AVAILABLE = False

# Twitter Integration
try:
    from twikit import Client as TwitterClient
    TWITTER_AVAILABLE = True
except ImportError:
    cprint("[WARNING] twikit not available - install with: pip install twikit", "yellow")
    TWITTER_AVAILABLE = False

# Add project root to path
import sys
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from src.polymarket_utils import PolymarketUtils, format_probability
from src.config import (
    POLYMARKET_EVENT_TWITTER_KEYWORDS,
    POLYMARKET_EVENT_NEWS_SOURCES,
    POLYMARKET_EVENT_SENTIMENT_THRESHOLD,
    POLYMARKET_EVENT_VOLUME_SPIKE_MULT,
    POLYMARKET_EVENT_CHECK_INTERVAL_SEC,
    POLYMARKET_DATA_DIR,
    POLYMARKET_VERBOSE_LOGGING
)


class PolymarketEventCatalystAgent:
    """
    Polymarket Event Catalyst Agent - News & Event Monitoring

    Monitors breaking news, events, and social media for catalysts that could
    drive prediction market movements.

    Key Features:
    1. Twitter monitoring for breaking news keywords
    2. Verified news source tracking (Reuters, AP, Bloomberg, etc.)
    3. FinBERT sentiment analysis for event impact assessment
    4. Volume spike detection (≥3x normal)
    5. Event-to-market matching (which markets are affected)
    6. Real-time signal generation

    Signal Strength = sentiment_magnitude × relevance_score × recency_decay
    """

    def __init__(self):
        """Initialize Event Catalyst Agent"""
        self.data_dir = Path(POLYMARKET_DATA_DIR) / 'event_catalyst'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.utils = PolymarketUtils()

        # Data files
        self.events_file = self.data_dir / 'events.csv'
        self.event_signals_file = self.data_dir / 'event_signals.csv'
        self.volume_spikes_file = self.data_dir / 'volume_spikes.csv'

        # Initialize data storage
        self.events = self._load_or_create_events()
        self.event_signals = self._load_or_create_event_signals()
        self.volume_spikes = self._load_or_create_volume_spikes()

        # Initialize FinBERT if available
        self.finbert_tokenizer = None
        self.finbert_model = None
        if FINBERT_AVAILABLE:
            self._initialize_finbert()

        # Initialize Twitter client (optional)
        self.twitter_client = None
        if TWITTER_AVAILABLE:
            cprint("[EMOJI] Twitter client available (requires auth)", "green")

        cprint(f"\n[EMOJI] Polymarket Event Catalyst Agent Initialized", "cyan", attrs=['bold'])
        cprint(f"[EMOJI] Historical Events: {len(self.events)}", "green")
        cprint(f"[EMOJI] Active Signals: {len(self.event_signals)}", "yellow")
        cprint(f"[EMOJI] Volume Spikes: {len(self.volume_spikes)}", "cyan")

    def _load_or_create_events(self) -> pd.DataFrame:
        """Load or create events tracking database"""
        if self.events_file.exists():
            df = pd.read_csv(self.events_file)
            # Filter to last 30 days
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = datetime.now() - timedelta(days=30)
            df = df[df['timestamp'] > cutoff].copy()
            cprint(f"[SYM] Loaded {len(df)} events from history", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'event_text',
                'source',  # twitter, reuters, bloomberg, etc.
                'sentiment_score',  # -1 to 1
                'sentiment_magnitude',  # 0 to 1
                'keywords_matched',
                'affected_markets'  # JSON list of market_ids
            ])
            cprint(f"[EMOJI] Created new events database", "yellow")
            return df

    def _load_or_create_event_signals(self) -> pd.DataFrame:
        """Load or create event signals database"""
        if self.event_signals_file.exists():
            df = pd.read_csv(self.event_signals_file)
            # Filter to last 24 hours
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = datetime.now() - timedelta(hours=24)
            df = df[df['timestamp'] > cutoff].copy()
            cprint(f"[SYM] Loaded {len(df)} event signals", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'market_id',
                'question',
                'event_text',
                'sentiment_score',
                'direction',  # BULLISH or BEARISH (for YES outcome)
                'signal_strength',
                'recency_hours'
            ])
            cprint(f"[EMOJI] Created new event signals database", "yellow")
            return df

    def _load_or_create_volume_spikes(self) -> pd.DataFrame:
        """Load or create volume spikes database"""
        if self.volume_spikes_file.exists():
            df = pd.read_csv(self.volume_spikes_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = datetime.now() - timedelta(days=7)
            df = df[df['timestamp'] > cutoff].copy()
            cprint(f"[SYM] Loaded {len(df)} volume spikes", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'market_id',
                'question',
                'normal_volume',
                'spike_volume',
                'spike_multiple',
                'spike_duration_minutes'
            ])
            cprint(f"[EMOJI] Created new volume spikes database", "yellow")
            return df

    def _initialize_finbert(self):
        """Initialize FinBERT model for sentiment analysis"""
        try:
            model_name = "ProsusAI/finbert"
            cprint(f"[EMOJI] Loading FinBERT model: {model_name}...", "cyan")

            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Set to eval mode
            self.finbert_model.eval()

            cprint(f"[SYM] FinBERT loaded successfully", "green")

        except Exception as e:
            cprint(f"[SYM] Failed to load FinBERT: {e}", "red")
            self.finbert_tokenizer = None
            self.finbert_model = None

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT

        Args:
            text: Text to analyze

        Returns:
            {
                'score': -1 to 1 (negative to positive),
                'magnitude': 0 to 1 (confidence),
                'label': 'positive', 'negative', or 'neutral'
            }
        """
        if not FINBERT_AVAILABLE or self.finbert_model is None:
            # Fallback to simple keyword-based sentiment
            return self._simple_sentiment(text)

        try:
            # Tokenize
            inputs = self.finbert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Get prediction
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)[0]

            # FinBERT labels: [positive, negative, neutral]
            positive_score = probs[0].item()
            negative_score = probs[1].item()
            neutral_score = probs[2].item()

            # Calculate sentiment score (-1 to 1)
            sentiment_score = positive_score - negative_score

            # Magnitude = max probability (confidence)
            magnitude = max(positive_score, negative_score, neutral_score)

            # Label
            if positive_score > negative_score and positive_score > neutral_score:
                label = 'positive'
            elif negative_score > positive_score and negative_score > neutral_score:
                label = 'negative'
            else:
                label = 'neutral'

            return {
                'score': sentiment_score,
                'magnitude': magnitude,
                'label': label
            }

        except Exception as e:
            cprint(f"[SYM] FinBERT analysis failed: {e}", "yellow")
            return self._simple_sentiment(text)

    def _simple_sentiment(self, text: str) -> Dict[str, float]:
        """
        Simple keyword-based sentiment analysis (fallback)

        Returns:
            {score, magnitude, label}
        """
        text_lower = text.lower()

        positive_words = ['win', 'success', 'surge', 'rally', 'bull', 'up', 'gain', 'positive', 'victory', 'confirmed']
        negative_words = ['loss', 'fail', 'crash', 'bear', 'down', 'drop', 'negative', 'defeat', 'rejected', 'denied']

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count + neg_count == 0:
            return {'score': 0, 'magnitude': 0, 'label': 'neutral'}

        score = (pos_count - neg_count) / (pos_count + neg_count)
        magnitude = min(1.0, (pos_count + neg_count) / 5)  # Cap at 5 words

        label = 'positive' if score > 0.2 else ('negative' if score < -0.2 else 'neutral')

        return {'score': score, 'magnitude': magnitude, 'label': label}

    def monitor_twitter(self, keywords: List[str], max_tweets: int = 50) -> List[Dict]:
        """
        Monitor Twitter for breaking news

        Args:
            keywords: Keywords to search for
            max_tweets: Maximum tweets to fetch

        Returns:
            List of tweet dicts with sentiment
        """
        if not TWITTER_AVAILABLE or self.twitter_client is None:
            cprint("[SYM] Twitter monitoring not available (requires auth)", "yellow")
            return []

        # This would require Twitter authentication
        # For now, return empty (implement with user's Twitter credentials)
        cprint("[EMOJI] Twitter monitoring requires authentication setup", "yellow")
        return []

    def detect_breaking_news(self, text: str) -> bool:
        """
        Detect if text contains breaking news keywords

        Args:
            text: Text to analyze

        Returns:
            True if breaking news detected
        """
        text_lower = text.lower()

        for keyword in POLYMARKET_EVENT_TWITTER_KEYWORDS:
            if keyword.lower() in text_lower:
                return True

        # Check for news source mentions
        for source in POLYMARKET_EVENT_NEWS_SOURCES:
            if source.lower() in text_lower:
                return True

        return False

    def process_event(
        self,
        event_text: str,
        source: str = 'manual',
        affected_markets: Optional[List[str]] = None
    ) -> Dict:
        """
        Process new event and generate signals

        Args:
            event_text: Event description
            source: Event source (twitter, reuters, bloomberg, manual, etc.)
            affected_markets: List of market IDs affected by this event

        Returns:
            Event processing result
        """
        cprint(f"\n[EMOJI] PROCESSING EVENT", "cyan", attrs=['bold'])
        cprint(f"[EMOJI] Text: {event_text[:100]}...", "white")
        cprint(f"[EMOJI] Source: {source}", "cyan")

        # Analyze sentiment
        sentiment = self.analyze_sentiment(event_text)

        cprint(f"[EMOJI] Sentiment: {sentiment['label'].upper()} (score: {sentiment['score']:.2f}, magnitude: {sentiment['magnitude']:.2f})", "cyan")

        # Check if magnitude meets threshold
        if sentiment['magnitude'] < POLYMARKET_EVENT_SENTIMENT_THRESHOLD:
            cprint(f"⏭  Event magnitude too low ({sentiment['magnitude']:.2f} < {POLYMARKET_EVENT_SENTIMENT_THRESHOLD})", "yellow")
            return None

        # Check for breaking news keywords
        is_breaking = self.detect_breaking_news(event_text)
        keywords_matched = [kw for kw in POLYMARKET_EVENT_TWITTER_KEYWORDS if kw.lower() in event_text.lower()]

        # Store event
        event_data = {
            'timestamp': datetime.now(),
            'event_text': event_text,
            'source': source,
            'sentiment_score': sentiment['score'],
            'sentiment_magnitude': sentiment['magnitude'],
            'keywords_matched': ','.join(keywords_matched) if keywords_matched else '',
            'affected_markets': ','.join(affected_markets) if affected_markets else ''
        }

        self.events = pd.concat([
            self.events,
            pd.DataFrame([event_data])
        ], ignore_index=True)

        # Generate signals for affected markets
        if affected_markets:
            for market_id in affected_markets:
                signal = self._generate_event_signal(
                    market_id=market_id,
                    event_text=event_text,
                    sentiment=sentiment
                )

                if signal:
                    self.event_signals = pd.concat([
                        self.event_signals,
                        pd.DataFrame([signal])
                    ], ignore_index=True)

        # Save data
        self._save_data()

        result = {
            'event_stored': True,
            'sentiment': sentiment,
            'is_breaking': is_breaking,
            'signals_generated': len(affected_markets) if affected_markets else 0
        }

        cprint(f"[SYM] Event processed: {result['signals_generated']} signals generated", "green")

        return result

    def _generate_event_signal(
        self,
        market_id: str,
        event_text: str,
        sentiment: Dict[str, float]
    ) -> Optional[Dict]:
        """
        Generate trading signal from event

        Args:
            market_id: Market ID
            event_text: Event description
            sentiment: Sentiment analysis result

        Returns:
            Signal dict or None
        """
        # Get market data
        market = self.utils.get_market(market_id)

        if not market:
            cprint(f"[SYM] Market {market_id} not found", "yellow")
            return None

        # Determine direction (for YES outcome)
        # Positive sentiment = BULLISH for YES
        # Negative sentiment = BEARISH for YES
        if sentiment['score'] > 0.2:
            direction = 'BULLISH'
        elif sentiment['score'] < -0.2:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'

        # Calculate signal strength
        # strength = magnitude × |score|
        signal_strength = sentiment['magnitude'] * abs(sentiment['score'])

        # Recency = 0 (fresh event)
        recency_hours = 0

        signal = {
            'timestamp': datetime.now(),
            'market_id': market_id,
            'question': market.get('question', 'Unknown'),
            'event_text': event_text[:200],  # Truncate
            'sentiment_score': sentiment['score'],
            'direction': direction,
            'signal_strength': signal_strength,
            'recency_hours': recency_hours
        }

        return signal

    def detect_volume_spike(
        self,
        market_id: str,
        current_volume: float,
        normal_volume: float
    ) -> Optional[Dict]:
        """
        Detect unusual volume spikes

        Args:
            market_id: Market ID
            current_volume: Current volume
            normal_volume: Typical/average volume

        Returns:
            Spike dict or None if no spike detected
        """
        if normal_volume == 0:
            return None

        spike_multiple = current_volume / normal_volume

        if spike_multiple >= POLYMARKET_EVENT_VOLUME_SPIKE_MULT:
            cprint(f"\n[EMOJI] VOLUME SPIKE DETECTED", "red", attrs=['bold'])
            cprint(f"[EMOJI] Market: {market_id}", "cyan")
            cprint(f"[EMOJI] Normal: ${normal_volume:,.0f} → Current: ${current_volume:,.0f}", "yellow")
            cprint(f"[EMOJI] Spike: {spike_multiple:.1f}x", "red")

            market = self.utils.get_market(market_id)

            spike_data = {
                'timestamp': datetime.now(),
                'market_id': market_id,
                'question': market.get('question', 'Unknown') if market else 'Unknown',
                'normal_volume': normal_volume,
                'spike_volume': current_volume,
                'spike_multiple': spike_multiple,
                'spike_duration_minutes': 0  # Updated with monitoring
            }

            self.volume_spikes = pd.concat([
                self.volume_spikes,
                pd.DataFrame([spike_data])
            ], ignore_index=True)

            self._save_data()

            return spike_data

        return None

    def get_active_signals(self, market_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get active event signals with recency decay

        Args:
            market_id: Optional market ID to filter by

        Returns:
            DataFrame of active signals
        """
        if self.event_signals.empty:
            return pd.DataFrame()

        signals = self.event_signals.copy()
        signals['timestamp'] = pd.to_datetime(signals['timestamp'])

        # Calculate recency (hours since event)
        now = datetime.now()
        signals['recency_hours'] = (now - signals['timestamp']).dt.total_seconds() / 3600

        # Apply recency decay (exponential)
        # strength decays by 50% every 12 hours
        half_life_hours = 12
        signals['recency_decay'] = np.exp(-np.log(2) * signals['recency_hours'] / half_life_hours)

        # Update signal strength with decay
        signals['signal_strength'] = signals['signal_strength'] * signals['recency_decay']

        # Filter out very old signals (>24h)
        signals = signals[signals['recency_hours'] <= 24].copy()

        # Filter by market if specified
        if market_id:
            signals = signals[signals['market_id'] == market_id].copy()

        # Sort by signal strength
        signals = signals.sort_values('signal_strength', ascending=False)

        return signals

    def _save_data(self):
        """Save all data to CSV files"""
        self.events.to_csv(self.events_file, index=False)
        self.event_signals.to_csv(self.event_signals_file, index=False)
        self.volume_spikes.to_csv(self.volume_spikes_file, index=False)

        if POLYMARKET_VERBOSE_LOGGING:
            cprint(f"[EMOJI] Data saved to {self.data_dir.name}/", "blue")

    def print_summary(self):
        """Print agent summary"""
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[EMOJI] EVENT CATALYST AGENT SUMMARY", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        # Events
        cprint(f"\n[EMOJI] EVENTS:", "yellow")
        cprint(f"   Total Events (30d): {len(self.events)}", "cyan")

        if not self.events.empty:
            recent = self.events[self.events['timestamp'] > (datetime.now() - timedelta(hours=24))]
            cprint(f"   Recent Events (24h): {len(recent)}", "green")

            positive = self.events[self.events['sentiment_score'] > 0.2]
            negative = self.events[self.events['sentiment_score'] < -0.2]
            cprint(f"   Positive: {len(positive)} | Negative: {len(negative)}", "cyan")

        # Signals
        active = self.get_active_signals()
        cprint(f"\n[EMOJI] ACTIVE SIGNALS:", "yellow")
        cprint(f"   Active Signals: {len(active)}", "green" if len(active) > 0 else "yellow")

        if not active.empty:
            bullish = active[active['direction'] == 'BULLISH']
            bearish = active[active['direction'] == 'BEARISH']
            cprint(f"   Bullish: {len(bullish)} | Bearish: {len(bearish)}", "cyan")
            cprint(f"   Avg Strength: {active['signal_strength'].mean():.2f}", "cyan")

        # Volume spikes
        cprint(f"\n[EMOJI] VOLUME SPIKES:", "yellow")
        cprint(f"   Spikes (7d): {len(self.volume_spikes)}", "cyan")

        if not self.volume_spikes.empty:
            recent_spikes = self.volume_spikes[
                self.volume_spikes['timestamp'] > (datetime.now() - timedelta(hours=24))
            ]
            cprint(f"   Recent Spikes (24h): {len(recent_spikes)}", "green" if len(recent_spikes) > 0 else "cyan")

        cprint(f"\n{'='*80}\n", "cyan")


def main():
    """Test Event Catalyst Agent"""

    agent = PolymarketEventCatalystAgent()

    # Example 1: Process breaking news event
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 1: Process Breaking News", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    result1 = agent.process_event(
        event_text="BREAKING: Federal Reserve announces interest rate cut of 0.5%, signaling dovish monetary policy shift. Markets rally on news.",
        source="reuters",
        affected_markets=['market_123', 'market_456']
    )

    # Example 2: Process negative event
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 2: Process Negative Event", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    result2 = agent.process_event(
        event_text="Major tech company reports disappointing earnings, misses revenue targets by 15%. Stock down in after-hours trading.",
        source="bloomberg",
        affected_markets=['market_789']
    )

    # Example 3: Detect volume spike
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 3: Detect Volume Spike", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    spike = agent.detect_volume_spike(
        market_id='market_123',
        current_volume=150000,
        normal_volume=40000
    )

    # Example 4: Get active signals
    cprint("\n" + "="*80, "magenta")
    cprint("TEST 4: Active Signals", "magenta", attrs=['bold'])
    cprint("="*80 + "\n", "magenta")

    active_signals = agent.get_active_signals()
    if not active_signals.empty:
        cprint(f"Found {len(active_signals)} active signals:", "green")
        for _, sig in active_signals.head(5).iterrows():
            cprint(
                f"  {sig['direction']:8s} | "
                f"Strength: {sig['signal_strength']:.2f} | "
                f"Age: {sig['recency_hours']:.1f}h | "
                f"{sig['question'][:50]}...",
                "cyan"
            )

    # Print summary
    agent.print_summary()

    cprint("\n[SYM] Event Catalyst Agent Tests Complete!\n", "green", attrs=['bold'])


if __name__ == "__main__":
    main()
