"""
Lightweight Polymarket Scanner
Runs periodic quick scans for trading opportunities without full orchestrator overhead.
Only triggers full analysis when strong signals detected.
"""

import time
import pandas as pd
import requests
from datetime import datetime
from termcolor import cprint
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import (
    POLYMARKET_EV_MIN,
    POLYMARKET_Z_MIN,
    POLYMARKET_MAX_SPREAD,
    POLYMARKET_MIN_LIQUIDITY,
    POLYMARKET_SCAN_INTERVAL_MINUTES
)

class PolymarketScanner:
    """Lightweight scanner for quick market opportunity checks"""

    def __init__(self):
        self.api_base = "https://gamma-api.polymarket.com"
        self.last_scan_time = None
        self.scan_count = 0

    def fetch_active_markets(self, limit=50):
        """Fetch currently active markets from Polymarket API"""
        try:
            url = f"{self.api_base}/markets"
            params = {
                "closed": "false",
                "limit": limit,
                "order": "volume24hr",
                "ascending": "false"
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                cprint(f"[WARN] API returned {response.status_code}", "yellow")
                return []
        except Exception as e:
            cprint(f"[ERROR] Failed to fetch markets: {e}", "red")
            return []

    def quick_filter_markets(self, markets):
        """Apply quick filters to identify promising markets"""
        promising = []

        for market in markets:
            try:
                # Extract key metrics
                question = market.get("question", "")
                market_id = market.get("condition_id", "")

                # Get outcome tokens
                outcomes = market.get("outcomes", [])
                if len(outcomes) < 2:
                    continue

                # Calculate current odds and spread
                outcome_prices = market.get("outcomePrices", [])
                if len(outcome_prices) < 2:
                    continue

                yes_price = float(outcome_prices[0]) if outcome_prices[0] else 0.5
                no_price = float(outcome_prices[1]) if outcome_prices[1] else 0.5

                spread = abs(yes_price - no_price)

                # Get liquidity
                liquidity = float(market.get("liquidity", 0))
                volume_24h = float(market.get("volume24hr", 0))

                # Quick filters
                passes_spread = spread <= POLYMARKET_MAX_SPREAD
                passes_liquidity = liquidity >= POLYMARKET_MIN_LIQUIDITY
                has_volume = volume_24h > 1000  # At least $1k daily volume

                # Check for extreme pricing (potential value)
                extreme_price = yes_price < 0.15 or yes_price > 0.85

                if passes_spread and passes_liquidity and (has_volume or extreme_price):
                    promising.append({
                        "market_id": market_id,
                        "question": question,
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "spread": spread,
                        "liquidity": liquidity,
                        "volume_24h": volume_24h,
                        "extreme_price": extreme_price
                    })

            except Exception as e:
                continue

        return promising

    def check_historical_data(self, market_id):
        """Check if we have historical data for volatility/anomaly detection"""
        snapshots_path = "src/data/polymarket/training_data/market_snapshots.csv"

        try:
            if os.path.exists(snapshots_path):
                df = pd.read_csv(snapshots_path)
                market_data = df[df['market_id'] == market_id]

                if len(market_data) >= 10:  # At least 10 snapshots (10 minutes)
                    # Calculate price volatility
                    recent = market_data.tail(20)
                    if 'yes_price' in recent.columns:
                        volatility = recent['yes_price'].std()
                        return {
                            "has_history": True,
                            "volatility": volatility,
                            "data_points": len(market_data)
                        }

            return {"has_history": False}

        except Exception as e:
            return {"has_history": False}

    def score_opportunity(self, market, history_check):
        """Score opportunity from 0-100 based on quick signals"""
        score = 0
        reasons = []

        # Spread bonus (tighter = better)
        if market['spread'] <= 0.03:
            score += 30
            reasons.append(f"Tight spread ({market['spread']:.2%})")
        elif market['spread'] <= 0.06:
            score += 15

        # Liquidity bonus
        if market['liquidity'] >= 50000:
            score += 20
            reasons.append(f"High liquidity (${market['liquidity']:,.0f})")
        elif market['liquidity'] >= 10000:
            score += 10

        # Volume bonus
        if market['volume_24h'] >= 100000:
            score += 20
            reasons.append(f"Strong volume (${market['volume_24h']:,.0f})")
        elif market['volume_24h'] >= 10000:
            score += 10

        # Extreme price bonus (potential value)
        if market['extreme_price']:
            score += 15
            reasons.append(f"Extreme pricing ({market['yes_price']:.2%})")

        # Historical data bonus
        if history_check.get('has_history'):
            score += 15
            volatility = history_check.get('volatility', 0)
            if volatility > 0.05:  # High volatility = opportunity
                score += 10
                reasons.append(f"High volatility ({volatility:.2%})")

        return score, reasons

    def scan_markets(self):
        """Run a single scan cycle"""
        self.scan_count += 1
        self.last_scan_time = datetime.now()

        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[SCAN #{self.scan_count}] {self.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}", "cyan")
        cprint(f"{'='*80}", "cyan")

        # Fetch markets
        cprint("[1/4] Fetching active markets...", "white")
        markets = self.fetch_active_markets(limit=50)
        cprint(f"[OK] Retrieved {len(markets)} markets", "green")

        # Quick filter
        cprint("[2/4] Applying quick filters...", "white")
        promising = self.quick_filter_markets(markets)
        cprint(f"[OK] {len(promising)} markets passed filters", "green")

        if len(promising) == 0:
            cprint("[RESULT] No opportunities found in this scan", "yellow")
            return []

        # Check historical data and score
        cprint("[3/4] Scoring opportunities...", "white")
        scored_opportunities = []

        for market in promising:
            history = self.check_historical_data(market['market_id'])
            score, reasons = self.score_opportunity(market, history)

            if score >= 40:  # Threshold for "strong signal"
                scored_opportunities.append({
                    **market,
                    "score": score,
                    "reasons": reasons,
                    "history": history
                })

        scored_opportunities.sort(key=lambda x: x['score'], reverse=True)
        cprint(f"[OK] {len(scored_opportunities)} opportunities scored >= 40", "green")

        # Display results
        cprint("[4/4] Results:", "white")
        if len(scored_opportunities) > 0:
            cprint(f"\n[FOUND] {len(scored_opportunities)} STRONG OPPORTUNITIES:", "green", attrs=["bold"])

            for i, opp in enumerate(scored_opportunities[:5], 1):  # Top 5
                cprint(f"\n#{i} [Score: {opp['score']}/100]", "cyan", attrs=["bold"])
                cprint(f"  Market: {opp['question'][:80]}", "white")
                cprint(f"  Yes Price: {opp['yes_price']:.2%} | No Price: {opp['no_price']:.2%}", "white")
                cprint(f"  Spread: {opp['spread']:.2%} | Liquidity: ${opp['liquidity']:,.0f}", "white")
                cprint(f"  Volume 24h: ${opp['volume_24h']:,.0f}", "white")
                cprint(f"  Reasons: {', '.join(opp['reasons'])}", "yellow")

                if opp['history'].get('has_history'):
                    cprint(f"  Historical Data: {opp['history']['data_points']} snapshots", "magenta")
        else:
            cprint("[RESULT] No strong opportunities (score >= 40) found", "yellow")

        return scored_opportunities

    def should_trigger_full_analysis(self, opportunities):
        """Decide if we should trigger full orchestrator"""
        if len(opportunities) == 0:
            return False

        # Trigger if top opportunity scores >= 70
        top_score = opportunities[0]['score']
        if top_score >= 70:
            cprint(f"\n[TRIGGER] Top opportunity score {top_score} >= 70", "green", attrs=["bold"])
            cprint("[ACTION] Starting full orchestrator analysis...", "green")
            return True

        # Or if we have 3+ opportunities with score >= 50
        high_score_count = sum(1 for opp in opportunities if opp['score'] >= 50)
        if high_score_count >= 3:
            cprint(f"\n[TRIGGER] {high_score_count} opportunities with score >= 50", "green", attrs=["bold"])
            cprint("[ACTION] Starting full orchestrator analysis...", "green")
            return True

        return False

    def run_continuous(self, scan_interval_minutes=None):
        """Run continuous scanning loop"""
        interval = scan_interval_minutes or POLYMARKET_SCAN_INTERVAL_MINUTES

        cprint("\n" + "="*80, "cyan", attrs=["bold"])
        cprint("POLYMARKET LIGHTWEIGHT SCANNER", "cyan", attrs=["bold"])
        cprint("="*80, "cyan", attrs=["bold"])
        cprint(f"\nScan interval: {interval} minutes", "white")
        cprint(f"Min spread: {POLYMARKET_MAX_SPREAD:.2%}", "white")
        cprint(f"Min liquidity: ${POLYMARKET_MIN_LIQUIDITY:,.0f}", "white")
        cprint(f"Strong signal threshold: 40/100", "white")
        cprint(f"Full analysis trigger: 70/100 or 3+ signals >= 50/100", "white")
        cprint("\nPress Ctrl+C to stop\n", "yellow")

        try:
            while True:
                # Run scan
                opportunities = self.scan_markets()

                # Check if should trigger full analysis
                if self.should_trigger_full_analysis(opportunities):
                    cprint("\n[EXEC] Running full orchestrator...", "cyan")
                    os.system("python src/agents/polymarket_orchestrator.py --single-pass")

                # Wait for next scan
                next_scan = datetime.now().timestamp() + (interval * 60)
                cprint(f"\n[SLEEP] Next scan in {interval} minutes ({datetime.fromtimestamp(next_scan).strftime('%H:%M:%S')})", "cyan")
                time.sleep(interval * 60)

        except KeyboardInterrupt:
            cprint("\n\n[STOP] Scanner stopped by user", "yellow")
            cprint(f"Total scans completed: {self.scan_count}", "white")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket Lightweight Scanner")
    parser.add_argument("--interval", type=int, default=10, help="Scan interval in minutes (default: 10)")
    parser.add_argument("--once", action="store_true", help="Run single scan and exit")

    args = parser.parse_args()

    scanner = PolymarketScanner()

    if args.once:
        # Single scan mode
        opportunities = scanner.scan_markets()
        if scanner.should_trigger_full_analysis(opportunities):
            cprint("\n[EXEC] Would trigger full orchestrator (use --once with caution)", "yellow")
    else:
        # Continuous mode
        scanner.run_continuous(scan_interval_minutes=args.interval)


if __name__ == "__main__":
    main()
