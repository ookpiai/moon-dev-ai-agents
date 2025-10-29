"""
🌙 Moon Dev's Polymarket Quick Start Script
Get your AI trading system up and running in 5 minutes
Built with love by Moon Dev 🚀
"""

import os
import sys
from pathlib import Path
from termcolor import cprint
import time

# Add to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def print_banner():
    """Print welcome banner"""
    cprint("\n" + "="*80, "cyan")
    cprint("🎲 POLYMARKET AI TRADING SYSTEM - QUICK START", "cyan", attrs=['bold'])
    cprint("="*80 + "\n", "cyan")


def check_dependencies():
    """Check if required packages are installed"""
    cprint("📦 Checking dependencies...", "yellow")

    required = [
        'pandas', 'numpy', 'sklearn', 'requests',
        'feedparser', 'termcolor', 'joblib'
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
            cprint(f"   ✅ {package}", "green")
        except ImportError:
            cprint(f"   ❌ {package} - MISSING", "red")
            missing.append(package)

    if missing:
        cprint(f"\n⚠️  Missing packages. Install with:", "yellow")
        cprint(f"   pip install {' '.join(missing)}\n", "white")
        return False

    cprint("✅ All dependencies installed!\n", "green")
    return True


def test_polymarket_api():
    """Test Polymarket API connection"""
    cprint("🔌 Testing Polymarket API connection...", "yellow")

    try:
        import requests
        response = requests.get(
            "https://gamma-api.polymarket.com/markets?limit=3&active=true",
            timeout=10
        )
        response.raise_for_status()

        markets = response.json()
        cprint(f"✅ Connected! Fetched {len(markets)} markets:", "green")

        for market in markets[:3]:
            question = market.get('question', 'N/A')[:60]
            cprint(f"   - {question}...", "cyan")

        return True

    except Exception as e:
        cprint(f"❌ Connection failed: {e}", "red")
        cprint("   Check internet connection or API status", "yellow")
        return False


def create_directory_structure():
    """Create required directories"""
    cprint("\n📁 Creating directory structure...", "yellow")

    dirs = [
        'src/data/polymarket/training_data',
        'src/data/polymarket/meta_learning',
        'src/data/polymarket/whale_flow',
        'src/data/polymarket/event_catalyst',
        'src/data/polymarket/anomaly',
        'src/data/polymarket/positions',
        'src/data/polymarket/swarm_forecasts',
        'src/data/polymarket/llm_forecasts',
        'src/data/polymarket/quant_layer'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        cprint(f"   ✅ {dir_path}", "green")

    cprint("✅ Directories created!\n", "green")


def show_next_steps():
    """Show next steps to user"""
    cprint("\n" + "="*80, "cyan")
    cprint("🚀 QUICK START COMPLETE!", "cyan", attrs=['bold'])
    cprint("="*80 + "\n", "cyan")

    cprint("📋 NEXT STEPS:\n", "yellow", attrs=['bold'])

    steps = [
        ("1️⃣  COLLECT DATA (24-48 hours minimum)", [
            "python src/agents/polymarket_data_collector.py",
            "   ↳ Collects market snapshots every 60 seconds",
            "   ↳ Let this run for at least 1-2 days"
        ]),
        ("\n2️⃣  TRAIN META-LEARNER (after data collection)", [
            "python src/agents/polymarket_meta_learner.py",
            "   ↳ Trains models on collected data",
            "   ↳ Generates calibration.json with agent weights"
        ]),
        ("\n3️⃣  TEST IN PAPER TRADING MODE", [
            "python src/agents/polymarket_orchestrator.py",
            "   ↳ Simulates trades without real money",
            "   ↳ Tests full trading pipeline"
        ]),
        ("\n4️⃣  MONITOR PERFORMANCE", [
            "Check src/data/polymarket/positions/closed_positions.csv",
            "   ↳ Review win rate, P&L, and exit rule distribution"
        ]),
        ("\n5️⃣  GO LIVE (when confident)", [
            "Update src/config.py:",
            "   POLYMARKET_PAPER_TRADING = False",
            "   ↳ ⚠️  WARNING: Real money at risk!"
        ])
    ]

    for step, commands in steps:
        cprint(step, "yellow", attrs=['bold'])
        for cmd in commands:
            if cmd.startswith("   ↳"):
                cprint(cmd, "cyan")
            else:
                cprint(cmd, "white", attrs=['bold'])

    cprint("\n" + "─"*80, "cyan")
    cprint("📚 DOCUMENTATION:", "yellow", attrs=['bold'])
    cprint("   Full Setup Guide: POLYMARKET_SETUP_GUIDE.md", "white")
    cprint("   Configuration: src/config.py (lines 126-292)", "white")
    cprint("   Agent Docs: docs/polymarket_agents.md", "white")

    cprint("\n" + "─"*80, "cyan")
    cprint("🎯 KEY CONFIGURATION OPTIONS:", "yellow", attrs=['bold'])

    configs = [
        ("Entry Thresholds", [
            "POLYMARKET_EV_MIN = 0.03  # 3% edge minimum",
            "POLYMARKET_Z_MIN = 1.5    # 1.5 sigma significance"
        ]),
        ("Position Sizing", [
            "POLYMARKET_KELLY_FRACTION = 0.25  # 25% Kelly",
            "POLYMARKET_MAX_POSITION_SIZE = 1000  # Max $1k per trade"
        ]),
        ("Exit Rules (ANY triggers exit)", [
            "POLYMARKET_EXIT_EV_DECAY = 0.01  # Exit if EV < 1%",
            "POLYMARKET_EXIT_PROFIT_TARGET = 0.08  # Exit at +8%",
            "POLYMARKET_EXIT_STOP_LOSS = -0.03  # Exit at -3%"
        ])
    ]

    for section, settings in configs:
        cprint(f"\n{section}:", "green")
        for setting in settings:
            cprint(f"   {setting}", "cyan")

    cprint("\n" + "="*80, "cyan")
    cprint("✅ You're ready to start collecting data!", "green", attrs=['bold'])
    cprint("="*80 + "\n", "cyan")


def run_demo():
    """Run a quick demo of the system"""
    cprint("\n🎬 RUNNING QUICK DEMO...\n", "magenta", attrs=['bold'])

    try:
        from src.agents.polymarket_swarm_forecaster import PolymarketSwarmForecaster

        cprint("Testing Swarm Forecaster with demo question...\n", "yellow")

        forecaster = PolymarketSwarmForecaster()

        result = forecaster.forecast(
            question="Will Bitcoin hit $100k by end of 2024?",
            description="Market resolves YES if Bitcoin closes above $100,000 on December 31, 2024.",
            context={
                'current_odds': {'yes': 0.42, 'no': 0.58},
                'volume_24h': 125000,
                'liquidity': 350000,
                'days_to_resolution': 60
            }
        )

        cprint("\n✅ Demo complete! Swarm forecaster is working.", "green", attrs=['bold'])

    except Exception as e:
        cprint(f"❌ Demo failed: {e}", "red")
        cprint("This is OK - you can still collect data and train models", "yellow")


def main():
    """Main quick start function"""
    print_banner()

    # Step 1: Check dependencies
    if not check_dependencies():
        cprint("❌ Please install missing packages first.\n", "red")
        return

    # Step 2: Test API
    if not test_polymarket_api():
        cprint("⚠️  API connection failed, but you can still proceed.\n", "yellow")

    # Step 3: Create directories
    create_directory_structure()

    # Step 4: Offer to run demo
    cprint("Would you like to run a quick demo? (y/n): ", "yellow", end="")
    try:
        response = input().strip().lower()
        if response == 'y':
            run_demo()
    except KeyboardInterrupt:
        cprint("\n\nSkipping demo...", "yellow")

    # Step 5: Show next steps
    show_next_steps()


if __name__ == "__main__":
    main()
