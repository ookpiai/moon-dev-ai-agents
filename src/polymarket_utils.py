"""
Moon Dev's Polymarket Utilities
Shared utilities for all Polymarket agents
Built with love by Moon Dev
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from termcolor import cprint

class PolymarketUtils:
    """Shared utilities for Polymarket trading system"""

    def __init__(self, markets_csv_path: str = 'data/polymarket_markets.csv'):
        """
        Initialize Polymarket utilities

        Args:
            markets_csv_path: Path to Polymarket markets CSV database
        """
        self.markets_csv_path = Path(markets_csv_path)
        self.markets_db = None
        self.load_markets_database()

    def load_markets_database(self):
        """Load Polymarket markets database from CSV"""
        try:
            if self.markets_csv_path.exists():
                self.markets_db = pd.read_csv(self.markets_csv_path)
                cprint(f"[OK] Loaded {len(self.markets_db)} markets from database", "green")
            else:
                cprint(f"[ERROR] Markets database not found at {self.markets_csv_path}", "red")
                cprint("[IDEA] Please run fetch_all_markets.py first", "yellow")
                self.markets_db = pd.DataFrame()
        except Exception as e:
            cprint(f"[ERROR] Error loading markets database: {str(e)}", "red")
            self.markets_db = pd.DataFrame()

    def refresh_markets_database(self):
        """Reload markets database from disk"""
        self.load_markets_database()

    def get_market(self, market_id: str) -> Optional[Dict]:
        """
        Get market data by ID

        Args:
            market_id: Polymarket market ID

        Returns:
            Market data as dict or None if not found
        """
        if self.markets_db.empty:
            return None

        market = self.markets_db[self.markets_db['market_id'] == market_id]

        if market.empty:
            return None

        return market.iloc[0].to_dict()

    def get_active_markets(self, min_volume_24h: float = 1000) -> pd.DataFrame:
        """
        Get markets with minimum 24h volume

        Args:
            min_volume_24h: Minimum 24h volume in USD

        Returns:
            DataFrame of active markets
        """
        if self.markets_db.empty:
            return pd.DataFrame()

        return self.markets_db[
            self.markets_db['volume_24h'] >= min_volume_24h
        ].copy()

    def search_markets(self, keywords: List[str], categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Search markets by keywords and optionally filter by categories

        Args:
            keywords: List of keywords to search for in question/description
            categories: Optional list of categories to filter by

        Returns:
            DataFrame of matching markets
        """
        if self.markets_db.empty:
            return pd.DataFrame()

        # Build search pattern (OR condition for keywords)
        pattern = '|'.join(keywords)

        # Search in question field
        mask = self.markets_db['question'].str.contains(
            pattern,
            case=False,
            na=False,
            regex=True
        )

        # Also search in description if available
        if 'description' in self.markets_db.columns:
            desc_mask = self.markets_db['description'].str.contains(
                pattern,
                case=False,
                na=False,
                regex=True
            )
            mask = mask | desc_mask

        results = self.markets_db[mask].copy()

        # Filter by categories if specified
        if categories and not results.empty and 'category' in results.columns:
            results = results[results['category'].isin(categories)]

        return results

    def parse_outcome_prices(self, outcome_prices_str: str) -> Dict[str, float]:
        """
        Parse outcome prices string into dict

        Args:
            outcome_prices_str: String like "0.65|0.35"

        Returns:
            Dict with 'YES' and 'NO' prices
        """
        try:
            if pd.isna(outcome_prices_str):
                return {'YES': 0.5, 'NO': 0.5}

            prices = outcome_prices_str.split('|')

            if len(prices) == 2:
                yes_price = float(prices[0])
                no_price = float(prices[1])
                return {'YES': yes_price, 'NO': no_price}
            else:
                return {'YES': 0.5, 'NO': 0.5}

        except Exception as e:
            cprint(f"[WARNING] Error parsing outcome prices '{outcome_prices_str}': {e}", "yellow")
            return {'YES': 0.5, 'NO': 0.5}

    def calculate_days_to_resolution(self, end_date_str: str) -> float:
        """
        Calculate days until market resolution

        Args:
            end_date_str: End date string from market data

        Returns:
            Days to resolution (float)
        """
        try:
            if pd.isna(end_date_str):
                return 999.0  # Unknown expiry

            end_date = pd.to_datetime(end_date_str)
            now = datetime.now()

            delta = end_date - now
            return delta.total_seconds() / 86400  # Convert to days

        except Exception as e:
            cprint(f"[WARNING] Error calculating days to resolution: {e}", "yellow")
            return 999.0

    def enrich_market_data(self, market: Dict) -> Dict:
        """
        Enrich market data with parsed fields

        Args:
            market: Raw market data dict

        Returns:
            Enriched market data
        """
        enriched = market.copy()

        # Parse outcome prices
        if 'outcome_prices' in market:
            prices = self.parse_outcome_prices(market['outcome_prices'])
            enriched['yes_price'] = prices['YES']
            enriched['no_price'] = prices['NO']
            enriched['yes_odds'] = prices['YES']  # Alias
            enriched['no_odds'] = prices['NO']    # Alias

        # Calculate days to resolution
        if 'end_date' in market:
            enriched['days_to_resolution'] = self.calculate_days_to_resolution(market['end_date'])

        # Calculate spread
        if 'yes_price' in enriched and 'no_price' in enriched:
            # Spread = sum of prices - 1 (should be close to 0 in efficient market)
            enriched['spread'] = abs((enriched['yes_price'] + enriched['no_price']) - 1.0)

        return enriched

    def filter_markets_by_criteria(
        self,
        min_volume_24h: float = 1000,
        min_liquidity: float = 10000,
        max_spread: float = 0.06,
        max_days_to_resolution: Optional[float] = None,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Filter markets by multiple criteria

        Args:
            min_volume_24h: Minimum 24h volume
            min_liquidity: Minimum liquidity
            max_spread: Maximum spread
            max_days_to_resolution: Maximum days to resolution (None = no limit)
            categories: List of categories to include

        Returns:
            Filtered DataFrame
        """
        if self.markets_db.empty:
            return pd.DataFrame()

        df = self.markets_db.copy()

        # Volume filter
        if 'volume_24h' in df.columns:
            df = df[df['volume_24h'] >= min_volume_24h]

        # Liquidity filter
        if 'liquidity' in df.columns:
            df = df[df['liquidity'] >= min_liquidity]

        # Category filter
        if categories and 'category' in df.columns:
            df = df[df['category'].isin(categories)]

        # Enrich and filter by spread
        enriched_markets = []
        for _, market in df.iterrows():
            enriched = self.enrich_market_data(market.to_dict())

            # Spread filter
            if enriched.get('spread', 0) <= max_spread:
                # Days to resolution filter
                if max_days_to_resolution is None or enriched.get('days_to_resolution', 0) <= max_days_to_resolution:
                    enriched_markets.append(enriched)

        return pd.DataFrame(enriched_markets) if enriched_markets else pd.DataFrame()

    def get_market_snapshot(self, market_id: str) -> Optional[Dict]:
        """
        Get enriched market snapshot for trading decisions

        Args:
            market_id: Market ID

        Returns:
            Enriched market snapshot or None
        """
        market = self.get_market(market_id)

        if market is None:
            return None

        return self.enrich_market_data(market)


def format_currency(amount: float) -> str:
    """Format currency with commas and 2 decimals"""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format percentage with 1 decimal"""
    return f"{value:.1f}%"


def format_probability(prob: float) -> str:
    """Format probability (0-1) as percentage with 1 decimal"""
    return f"{prob * 100:.1f}%"
