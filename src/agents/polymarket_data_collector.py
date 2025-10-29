"""
Moon Dev's Polymarket Data Collector
Real-time data collection from Polymarket API, Twitter, and RSS feeds
Built with love by Moon Dev
"""

import requests
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from termcolor import cprint
import feedparser
from py_clob_client.client import ClobClient

# Add project root to path
import sys
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from src.polymarket_utils import PolymarketUtils, format_currency, format_probability
from src.config import POLYMARKET_DATA_DIR


class PolymarketDataCollector:
    """
    Polymarket Data Collector - Live Market Data + Training Dataset Builder

    Collects:
    1. Market snapshots (prices, volume, liquidity, spread)
    2. Order book data (whale bets, large orders)
    3. News/Twitter events (RSS + Twitter API)
    4. Historical resolutions (ground truth labels)

    Output: Training dataset for meta-learning
    """

    def __init__(self):
        """Initialize Data Collector"""
        self.data_dir = Path(POLYMARKET_DATA_DIR) / 'training_data'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Data files
        self.snapshots_file = self.data_dir / 'market_snapshots.csv'
        self.orderbook_file = self.data_dir / 'orderbook_snapshots.csv'
        self.events_file = self.data_dir / 'event_snapshots.csv'
        self.resolutions_file = self.data_dir / 'market_resolutions.csv'

        # Load existing data
        self.snapshots = self._load_or_create_snapshots()
        self.orderbook_snapshots = self._load_or_create_orderbook()
        self.events = self._load_or_create_events()
        self.resolutions = self._load_or_create_resolutions()

        # API endpoints
        self.polymarket_api_base = "https://gamma-api.polymarket.com"
        self.clob_api_base = "https://clob.polymarket.com"

        # Initialize CLOB client for order book data
        self.clob_client = ClobClient(self.clob_api_base)

        # RSS feeds (for news monitoring)
        self.rss_feeds = [
            "http://feeds.reuters.com/reuters/topNews",
            "http://feeds.bbci.co.uk/news/world/rss.xml",
            "https://www.politico.com/rss/politicopicks.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
        ]

        cprint(f"\nPolymarket Data Collector Initialized", "cyan", attrs=['bold'])
        cprint(f"Historical Snapshots: {len(self.snapshots)}", "green")
        cprint(f"Order Book Snapshots: {len(self.orderbook_snapshots)}", "green")
        cprint(f"Event Records: {len(self.events)}", "green")
        cprint(f"[OK] Resolutions: {len(self.resolutions)}", "green")

    def _load_or_create_snapshots(self) -> pd.DataFrame:
        """Load or create market snapshots database"""
        if self.snapshots_file.exists():
            df = pd.read_csv(self.snapshots_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cprint(f"[OK] Loaded {len(df)} market snapshots", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'market_id',
                'question',
                'mid_yes',
                'mid_no',
                'spread',
                'liquidity',
                'volume_1m',
                'volume_24h',
                'volatility_lookback',
                'time_to_resolution_days',
                'market_type',  # politics, entertainment, crypto, etc.
                'regime'  # information, emotion, illiquid
            ])
            cprint(f"[NEW] Created new snapshots database", "yellow")
            return df

    def _load_or_create_orderbook(self) -> pd.DataFrame:
        """Load or create order book snapshots"""
        if self.orderbook_file.exists():
            df = pd.read_csv(self.orderbook_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cprint(f"[OK] Loaded {len(df)} order book snapshots", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'market_id',
                'whale_strength',
                'book_imbalance',
                'odds_velocity',
                'large_orders_yes',
                'large_orders_no',
                'top_wallet_exposure'
            ])
            cprint(f"[NEW] Created new order book database", "yellow")
            return df

    def _load_or_create_events(self) -> pd.DataFrame:
        """Load or create events database"""
        if self.events_file.exists():
            df = pd.read_csv(self.events_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cprint(f"[OK] Loaded {len(df)} event records", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'timestamp',
                'market_id',
                'event_text',
                'source',  # twitter, rss, manual
                'sentiment_score',
                'catalyst_impact',
                'match_score'  # relevance to market
            ])
            cprint(f"[NEW] Created new events database", "yellow")
            return df

    def _load_or_create_resolutions(self) -> pd.DataFrame:
        """Load or create resolutions database (ground truth)"""
        if self.resolutions_file.exists():
            df = pd.read_csv(self.resolutions_file)
            df['resolution_time'] = pd.to_datetime(df['resolution_time'])
            cprint(f"[OK] Loaded {len(df)} resolution records", "green")
            return df
        else:
            df = pd.DataFrame(columns=[
                'market_id',
                'question',
                'resolution_time',
                'resolved_outcome',  # 0 or 1 (YES=1, NO=0)
                'final_yes_price',
                'final_no_price'
            ])
            cprint(f"[NEW] Created new resolutions database", "yellow")
            return df

    # ===== POLYMARKET API METHODS =====

    def fetch_active_markets(self, limit: int = 100) -> List[Dict]:
        """
        Fetch active markets from Polymarket API

        Returns:
            List of market dicts
        """
        try:
            cprint(f"\n[FETCH] Fetching active markets from Polymarket API...", "cyan")

            url = f"{self.polymarket_api_base}/markets"
            params = {
                'limit': limit,
                'active': True,
                'closed': False
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            markets = response.json()

            cprint(f"[OK] Fetched {len(markets)} active markets", "green")

            return markets

        except Exception as e:
            cprint(f"[ERROR] Error fetching markets: {e}", "red")
            return []

    def fetch_market_data(self, market_id: str) -> Optional[Dict]:
        """
        Fetch detailed market data

        Returns:
            Market data dict
        """
        try:
            url = f"{self.polymarket_api_base}/markets/{market_id}"

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            market_data = response.json()

            return market_data

        except Exception as e:
            cprint(f"[ERROR] Error fetching market {market_id}: {e}", "red")
            return None

    def fetch_order_book(self, market: Dict) -> Optional[Dict]:
        """
        Fetch order book for market using CLOB client

        Args:
            market: Market dict from gamma API (must contain clobTokenIds)

        Returns:
            Order book dict with bids/asks
        """
        try:
            # Get token IDs from market (YES and NO outcomes)
            clob_token_ids = market.get('clobTokenIds')
            if not clob_token_ids:
                return None

            # Parse token IDs (they come as a JSON string)
            if isinstance(clob_token_ids, str):
                import json
                token_ids = json.loads(clob_token_ids)
            else:
                token_ids = clob_token_ids

            if not token_ids or len(token_ids) == 0:
                return None

            # Fetch order book for YES token (first token ID)
            token_id = token_ids[0]
            book = self.clob_client.get_order_book(token_id)

            return book

        except Exception as e:
            market_id = market.get('id', 'unknown')
            # Silently fail for markets without order books (expected)
            return None

    def fetch_trades(self, market_id: str, limit: int = 100) -> List[Dict]:
        """
        Fetch recent trades for market

        Returns:
            List of trade dicts
        """
        try:
            url = f"{self.clob_api_base}/trades"
            params = {
                'market': market_id,
                'limit': limit
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            trades = response.json()

            return trades

        except Exception as e:
            cprint(f"[ERROR] Error fetching trades for {market_id}: {e}", "red")
            return []

    # ===== SNAPSHOT COLLECTION =====

    def collect_market_snapshot(self, market_id: str) -> Optional[Dict]:
        """
        Collect complete market snapshot

        Returns:
            Snapshot dict ready for training dataset
        """
        cprint(f"\n[SNAPSHOT] Collecting snapshot for {market_id}...", "cyan")

        # Fetch market data
        market = self.fetch_market_data(market_id)
        if not market:
            return None

        # QUALITY FILTERS: Skip closed/inactive/low-liquidity markets
        if market.get('closed', False):
            cprint(f"[SKIP] Market {market_id} is closed", "yellow")
            return None

        if not market.get('active', True):
            cprint(f"[SKIP] Market {market_id} is inactive", "yellow")
            return None

        liquidity = float(market.get('liquidityNum', 0))
        if liquidity < 1000:
            cprint(f"[SKIP] Market {market_id} has low liquidity (${liquidity:.0f})", "yellow")
            return None

        volume_24h = float(market.get('volume24hr', market.get('volume', 0)))
        if volume_24h < 100:
            cprint(f"[SKIP] Market {market_id} has low volume (${volume_24h:.0f})", "yellow")
            return None

        # Fetch order book
        book = self.fetch_order_book(market)

        # Calculate metrics
        snapshot = {
            'timestamp': datetime.now(),
            'market_id': market_id,
            'question': market.get('question', ''),
            'mid_yes': self._calculate_mid_price(book, 'YES') if book else 0.5,
            'mid_no': self._calculate_mid_price(book, 'NO') if book else 0.5,
            'spread': self._calculate_spread(book, market),  # Now passes market for fallback
            'liquidity': float(market.get('liquidity', 0)),
            'volume_1m': self._calculate_volume_1m(market_id),
            'volume_24h': float(market.get('volume', 0)),
            'volatility_lookback': self._calculate_volatility(market_id),
            'time_to_resolution_days': self._calculate_time_to_resolution(market),
            'market_type': self._classify_market_type(market.get('question', '')),
            'regime': self._classify_regime(book, market)
        }

        # Store snapshot
        self.snapshots = pd.concat([
            self.snapshots,
            pd.DataFrame([snapshot])
        ], ignore_index=True)

        self._save_snapshots()

        cprint(f"[OK] Snapshot collected", "green")

        return snapshot

    def collect_orderbook_snapshot(self, market: Dict) -> Optional[Dict]:
        """
        Collect order book metrics for whale detection

        Args:
            market: Market dict from gamma API

        Returns:
            Order book snapshot
        """
        market_id = market.get('id')
        book = self.fetch_order_book(market)
        trades = self.fetch_trades(market_id, limit=100)

        if not book:
            return None

        orderbook_snapshot = {
            'timestamp': datetime.now(),
            'market_id': market_id,
            'whale_strength': self._detect_whale_activity(trades),
            'book_imbalance': self._calculate_book_imbalance(book),
            'odds_velocity': self._calculate_odds_velocity(market_id),
            'large_orders_yes': self._count_large_orders(trades, 'YES'),
            'large_orders_no': self._count_large_orders(trades, 'NO'),
            'top_wallet_exposure': self._calculate_top_wallet_exposure(trades)
        }

        # Store
        self.orderbook_snapshots = pd.concat([
            self.orderbook_snapshots,
            pd.DataFrame([orderbook_snapshot])
        ], ignore_index=True)

        self._save_orderbook()

        return orderbook_snapshot

    # ===== NEWS/EVENT COLLECTION =====

    def fetch_rss_feeds(self) -> List[Dict]:
        """
        Fetch news from RSS feeds

        Returns:
            List of news items
        """
        news_items = []

        for feed_url in self.rss_feeds:
            try:
                cprint(f"[NEWS] Fetching {feed_url}...", "cyan")

                feed = feedparser.parse(feed_url)

                for entry in feed.entries[:10]:  # Top 10 per feed
                    news_items.append({
                        'timestamp': datetime.now(),
                        'source': 'rss',
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', '')
                    })

                time.sleep(1)  # Rate limiting

            except Exception as e:
                cprint(f"[WARNING] Error fetching RSS feed {feed_url}: {e}", "yellow")

        cprint(f"[OK] Fetched {len(news_items)} news items", "green")

        return news_items

    def match_events_to_markets(
        self,
        event_text: str,
        markets: List[Dict]
    ) -> List[tuple]:
        """
        Match event to relevant markets using keyword matching

        Returns:
            List of (market_id, match_score) tuples
        """
        matches = []

        event_words = set(event_text.lower().split())

        for market in markets:
            question = market.get('question', '').lower()
            question_words = set(question.split())

            # Calculate overlap
            overlap = len(event_words & question_words)
            total = len(event_words | question_words)

            match_score = overlap / total if total > 0 else 0

            if match_score > 0.1:  # Threshold
                matches.append((market.get('id'), match_score))

        # Sort by match score
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    # ===== HELPER METHODS =====

    def _calculate_mid_price(self, book, side: str) -> float:
        """Calculate mid price from order book"""
        if not book:
            return 0.5

        try:
            # Handle OrderBookSummary object from CLOB client
            if hasattr(book, 'bids') and hasattr(book, 'asks'):
                bids = book.bids
                asks = book.asks
            # Handle dict from raw API
            elif isinstance(book, dict):
                bids = book.get('bids', [])
                asks = book.get('asks', [])
            else:
                return 0.5

            if not bids or not asks:
                return 0.5

            # Handle list of objects vs dicts
            if hasattr(bids[0], 'price'):
                best_bid = float(bids[0].price)
            else:
                best_bid = float(bids[0].get('price', 0))

            if hasattr(asks[0], 'price'):
                best_ask = float(asks[0].price)
            else:
                best_ask = float(asks[0].get('price', 1))

            mid = (best_bid + best_ask) / 2
            return mid
        except:
            return 0.5

    def _calculate_spread(self, book, market=None) -> float:
        """Calculate bid-ask spread with fallback to market prices"""
        # Try order book first
        if book:
            try:
                # Handle OrderBookSummary object from CLOB client
                if hasattr(book, 'bids') and hasattr(book, 'asks'):
                    bids = book.bids
                    asks = book.asks
                # Handle dict from raw API
                elif isinstance(book, dict):
                    bids = book.get('bids', [])
                    asks = book.get('asks', [])
                else:
                    bids, asks = None, None

                if bids and asks:
                    # Handle list of objects vs dicts
                    if hasattr(bids[0], 'price'):
                        best_bid = float(bids[0].price)
                    else:
                        best_bid = float(bids[0].get('price', 0))

                    if hasattr(asks[0], 'price'):
                        best_ask = float(asks[0].price)
                    else:
                        best_ask = float(asks[0].get('price', 1))

                    spread = best_ask - best_bid

                    # Sanity check: reject 50%+ spreads (probably stale/closed market)
                    if spread < 0.5:
                        return max(0.0, spread)
            except:
                pass

        # FALLBACK: Use market outcome prices
        if market:
            try:
                # outcomePrices is a JSON array like ["0.52", "0.48"]
                outcome_prices_str = market.get('outcomePrices', '[]')
                if isinstance(outcome_prices_str, str):
                    outcome_prices = json.loads(outcome_prices_str)
                else:
                    outcome_prices = outcome_prices_str

                if len(outcome_prices) >= 2:
                    yes_price = float(outcome_prices[0])
                    no_price = float(outcome_prices[1])

                    # Spread = inefficiency (for efficient markets, yes + no ≈ 1.0)
                    spread = abs(1.0 - (yes_price + no_price))

                    # Cap at 20% (sanity check)
                    return min(spread, 0.20)
            except:
                pass

        # Default fallback
        return 0.05

    def _calculate_volume_1m(self, market_id: str) -> float:
        """Calculate 1-minute volume"""
        # Get recent snapshots
        recent = self.snapshots[
            (self.snapshots['market_id'] == market_id) &
            (self.snapshots['timestamp'] > datetime.now() - timedelta(minutes=1))
        ]

        if len(recent) < 2:
            return 0.0

        # Simple approximation: delta in volume_24h
        volume_change = recent['volume_24h'].iloc[-1] - recent['volume_24h'].iloc[0]

        return max(0.0, volume_change)

    def _calculate_volatility(self, market_id: str) -> float:
        """Calculate price volatility over lookback window"""
        # Get recent snapshots (last hour)
        recent = self.snapshots[
            (self.snapshots['market_id'] == market_id) &
            (self.snapshots['timestamp'] > datetime.now() - timedelta(hours=1))
        ]

        if len(recent) < 2:
            return 0.0

        prices = recent['mid_yes'].values
        returns = np.diff(prices) / prices[:-1]

        volatility = np.std(returns) if len(returns) > 0 else 0.0

        return volatility

    def _calculate_time_to_resolution(self, market: Dict) -> float:
        """Calculate days until market resolution"""
        # Try multiple date fields (API inconsistencies)
        for field in ['endDate', 'endDateIso', 'end_date', 'endDateISO']:
            end_date_str = market.get(field)
            if end_date_str:
                try:
                    # Parse date (handles ISO8601, timezone-aware, etc.)
                    end_date = pd.to_datetime(end_date_str)

                    # Handle timezone-aware dates
                    if end_date.tzinfo:
                        now = datetime.now(end_date.tzinfo)
                    else:
                        now = datetime.now()

                    # Calculate delta in days
                    delta = (end_date - now).total_seconds() / 86400
                    return max(0.0, delta)
                except Exception as e:
                    # Try next field if parsing fails
                    continue

        # Only fallback to 999 if ALL date fields fail
        return 999.0

    def _classify_market_type(self, question: str) -> str:
        """Classify market by type"""
        question_lower = question.lower()

        if any(kw in question_lower for kw in ['election', 'president', 'congress', 'senate', 'vote']):
            return 'politics'
        elif any(kw in question_lower for kw in ['bitcoin', 'crypto', 'eth', 'price']):
            return 'crypto'
        elif any(kw in question_lower for kw in ['movie', 'oscars', 'award', 'celebrity']):
            return 'entertainment'
        elif any(kw in question_lower for kw in ['fed', 'rate', 'inflation', 'gdp', 'economy']):
            return 'economics'
        else:
            return 'other'

    def _classify_regime(self, book: Dict, market: Dict) -> str:
        """Classify market regime"""
        spread = self._calculate_spread(book)
        volume = float(market.get('volume', 0))
        liquidity = float(market.get('liquidity', 0))

        # Information regime
        if spread <= 0.04 and volume >= 5000 and liquidity >= 15000:
            return 'information'

        # Illiquid regime
        if spread > 0.06 or volume < 2000 or liquidity < 8000:
            return 'illiquid'

        # Emotion regime (high volume, wide spread)
        if spread > 0.05 and volume >= 10000:
            return 'emotion'

        return 'information'

    def _detect_whale_activity(self, trades: List[Dict]) -> float:
        """Detect whale activity from recent trades"""
        if not trades:
            return 0.0

        large_trades = [
            float(t.get('size', 0))
            for t in trades
            if float(t.get('size', 0)) >= 10000
        ]

        if not large_trades:
            return 0.0

        # Whale strength = sum of large trades / total volume
        total_volume = sum(float(t.get('size', 0)) for t in trades)

        whale_strength = sum(large_trades) / total_volume if total_volume > 0 else 0.0

        return min(1.0, whale_strength)

    def _calculate_book_imbalance(self, book) -> float:
        """Calculate order book imbalance"""
        if not book:
            return 0.0

        try:
            # Handle OrderBookSummary object from CLOB client
            if hasattr(book, 'bids') and hasattr(book, 'asks'):
                bids = book.bids
                asks = book.asks
            # Handle dict from raw API
            elif isinstance(book, dict):
                bids = book.get('bids', [])
                asks = book.get('asks', [])
            else:
                return 0.0

            # Calculate volumes
            if hasattr(bids[0] if bids else None, 'size'):
                bid_volume = sum(float(b.size) for b in bids)
                ask_volume = sum(float(a.size) for a in asks)
            else:
                bid_volume = sum(float(b.get('size', 0)) for b in bids)
                ask_volume = sum(float(a.get('size', 0)) for a in asks)

            total = bid_volume + ask_volume

            if total == 0:
                return 0.0

            imbalance = (bid_volume - ask_volume) / total
            return imbalance
        except:
            return 0.0

    def _calculate_odds_velocity(self, market_id: str) -> float:
        """Calculate rate of price change"""
        recent = self.snapshots[
            (self.snapshots['market_id'] == market_id) &
            (self.snapshots['timestamp'] > datetime.now() - timedelta(minutes=5))
        ]

        if len(recent) < 2:
            return 0.0

        price_change = recent['mid_yes'].iloc[-1] - recent['mid_yes'].iloc[0]
        time_delta = (recent['timestamp'].iloc[-1] - recent['timestamp'].iloc[0]).total_seconds() / 60

        velocity = price_change / time_delta if time_delta > 0 else 0.0

        return velocity

    def _count_large_orders(self, trades: List[Dict], side: str) -> int:
        """Count large orders on one side"""
        count = sum(
            1 for t in trades
            if t.get('side', '').upper() == side and float(t.get('size', 0)) >= 5000
        )

        return count

    def _calculate_top_wallet_exposure(self, trades: List[Dict]) -> float:
        """Calculate top wallet's market exposure"""
        if not trades:
            return 0.0

        # Group by wallet
        wallet_volumes = {}
        for trade in trades:
            wallet = trade.get('maker', 'unknown')
            size = float(trade.get('size', 0))
            wallet_volumes[wallet] = wallet_volumes.get(wallet, 0) + size

        if not wallet_volumes:
            return 0.0

        # Top wallet
        max_exposure = max(wallet_volumes.values())
        total_volume = sum(wallet_volumes.values())

        exposure_pct = max_exposure / total_volume if total_volume > 0 else 0.0

        return exposure_pct

    # ===== SAVE METHODS =====

    def _save_snapshots(self):
        """Save snapshots to CSV"""
        self.snapshots.to_csv(self.snapshots_file, index=False)

    def _save_orderbook(self):
        """Save order book snapshots to CSV"""
        self.orderbook_snapshots.to_csv(self.orderbook_file, index=False)

    def _save_events(self):
        """Save events to CSV"""
        self.events.to_csv(self.events_file, index=False)

    def _save_resolutions(self):
        """Save resolutions to CSV"""
        self.resolutions.to_csv(self.resolutions_file, index=False)

    # ===== MAIN COLLECTION LOOP =====

    def run_collection_loop(self, interval_seconds: int = 60, max_markets: int = 20):
        """
        Run continuous data collection loop

        Args:
            interval_seconds: Seconds between collections
            max_markets: Max markets to monitor per cycle
        """
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[CYCLE] STARTING DATA COLLECTION LOOP", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")
        cprint(f"[TIME]  Interval: {interval_seconds}s", "yellow")
        cprint(f"[DATA] Max Markets: {max_markets}", "yellow")

        cycle = 0

        try:
            while True:
                cycle += 1
                cprint(f"\n{'-'*80}", "cyan")
                cprint(f"[CYCLE] CYCLE {cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "cyan", attrs=['bold'])
                cprint(f"{'-'*80}", "cyan")

                # Fetch active markets
                markets = self.fetch_active_markets(limit=max_markets)

                # Collect snapshots
                for i, market in enumerate(markets[:max_markets], 1):
                    market_id = market.get('id')
                    cprint(f"\n[DATA] [{i}/{len(markets[:max_markets])}] {market.get('question', '')[:60]}...", "white")

                    # Market snapshot
                    self.collect_market_snapshot(market_id)

                    # Order book snapshot
                    self.collect_orderbook_snapshot(market)

                    time.sleep(1)  # Rate limiting

                # Fetch news
                news_items = self.fetch_rss_feeds()

                # Match events to markets
                for news in news_items:
                    event_text = f"{news['title']} {news['summary']}"
                    matches = self.match_events_to_markets(event_text, markets)

                    if matches:
                        cprint(f"[NEWS] Matched event to {len(matches)} markets", "green")

                # Summary
                cprint(f"\n[OK] Cycle {cycle} complete", "green", attrs=['bold'])
                cprint(f"[DATA] Total snapshots: {len(self.snapshots)}", "cyan")
                cprint(f"[DATA] Total orderbook snapshots: {len(self.orderbook_snapshots)}", "cyan")

                # Sleep
                cprint(f"\n[SLEEP] Sleeping {interval_seconds}s...\n", "yellow")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            cprint(f"\n[STOPPED] Collection stopped by user", "yellow")
        except Exception as e:
            cprint(f"\n[ERROR] Error in collection loop: {e}", "red")
        finally:
            # Final save
            self._save_snapshots()
            self._save_orderbook()
            self._save_events()
            cprint(f"\n[SAVED] Data saved. Collection complete.", "green")


def main():
    """Run data collector"""

    collector = PolymarketDataCollector()

    # Run collection loop (60 second intervals, monitor top 20 markets)
    collector.run_collection_loop(interval_seconds=60, max_markets=20)


if __name__ == "__main__":
    main()
