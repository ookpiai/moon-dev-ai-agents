# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an experimental AI trading system that orchestrates 50+ specialized AI agents to analyze markets, execute strategies, and manage risk across:
- **Cryptocurrency markets** (primarily Solana via BirdEye, Helius RPC)
- **Polymarket prediction markets** (probability arbitrage system)

The project uses a modular agent architecture with unified LLM provider abstraction supporting Claude, GPT-4, DeepSeek, Groq, Gemini, and local Ollama models.

## Key Development Commands

### Environment Setup
```bash
# Use existing conda environment (DO NOT create new virtual environments)
conda activate tflow

# Install/update dependencies
pip install -r requirements.txt

# IMPORTANT: Update requirements.txt every time you add a new package
pip freeze > requirements.txt
```

### Running the System

#### Crypto Trading System
```bash
# Run main orchestrator (controls multiple agents)
python src/main.py

# Run individual agents standalone
python src/agents/trading_agent.py
python src/agents/risk_agent.py
python src/agents/rbi_agent.py
python src/agents/chat_agent.py
# ... any agent in src/agents/ can run independently
```

#### Polymarket System
```bash
# Step 1: Collect data (run for 24-48 hours minimum)
python src/agents/polymarket_data_collector.py

# Step 2: Train meta-learner (after data collection)
python src/agents/polymarket_meta_learner.py

# Step 3: Run orchestrator (paper trading mode by default)
python src/agents/polymarket_orchestrator.py

# Monitor progress anytime
python check_progress.py
```

### Backtesting & Pinescript Conversion
```bash
# Use backtesting.py library with pandas_ta or talib for indicators
# Sample OHLCV data available at:
# src/data/rbi/BTC-USD-15m.csv (2023 data, ~31k bars)
# src/data/rbi/BTC-USD-1h-2020-2025.csv (2-year hourly, ~17k bars)

# Run converted Pinescript strategies
python src/data/pinescript_conversions/backtests/ADX_Squeeze_R_Based_BT_v2.py

# Extract backtest results (daily NAV + trades list)
python extract_backtest_data.py
# Output: backtest_daily_nav.csv, backtest_trades.csv

# Download fresh BTC data
python download_btc_15m_2025.py  # Last 60 days, 15-minute bars
python download_btc_1h_2020_2025.py  # 2 years, hourly bars
```

## Architecture Overview

### Core Structure
```
src/
├── agents/              # 50+ specialized AI agents (each <800 lines)
│   ├── trading_agent.py, risk_agent.py, etc.        # Crypto trading
│   └── polymarket_*.py                               # Polymarket system (10 agents)
├── models/              # LLM provider abstraction (ModelFactory pattern)
├── strategies/          # User-defined trading strategies
├── scripts/             # Standalone utility scripts
├── data/                # Agent outputs, memory, analysis results
│   ├── polymarket/      # Polymarket-specific data
│   │   ├── training_data/      # Market snapshots, events, resolutions
│   │   ├── meta_learning/      # calibration.json (learned weights)
│   │   ├── positions/          # Open/closed positions
│   │   └── [agent_outputs]/    # Per-agent analysis results
│   ├── pinescript_conversions/ # TradingView Pinescript → Python conversions
│   │   ├── backtests/          # Converted backtest strategies
│   │   └── analysis/           # Strategy analysis documents
│   └── rbi/             # OHLCV data, backtest results, RBI agent outputs
├── config.py            # Global configuration (crypto + Polymarket)
├── polymarket_utils.py  # Shared Polymarket utilities
├── main.py              # Main orchestrator for crypto trading
├── nice_funcs.py        # ~1,200 lines of shared trading utilities
├── nice_funcs_hl.py     # Hyperliquid-specific utilities
├── extract_backtest_data.py  # Extract daily NAV + trades from backtests
├── download_btc_*.py    # BTC data downloaders (yfinance)
└── ezbot.py             # Legacy trading controller
```

### Agent Ecosystem

#### Crypto Trading Agents
**Trading**: `trading_agent`, `strategy_agent`, `risk_agent`, `copybot_agent`
**Market Analysis**: `sentiment_agent`, `whale_agent`, `funding_agent`, `liquidation_agent`, `chartanalysis_agent`
**Content Creation**: `chat_agent`, `clips_agent`, `tweet_agent`, `video_agent`, `phone_agent`
**Strategy Development**: `rbi_agent` (Research-Based Inference - codes backtests from videos/PDFs), `research_agent`
**Specialized**: `sniper_agent`, `solana_agent`, `tx_agent`, `million_agent`, `tiktok_agent`, `compliance_agent`

#### Polymarket Prediction Market Agents (10 Agents)

**Core Pipeline** (5-phase: SENSE → THINK → DECIDE → TRADE → EXIT):

1. **Data Collection & Meta-Learning**:
   - `polymarket_data_collector.py` - Real-time market snapshots, order book data (via `py-clob-client`), RSS feeds, whale tracking
   - `polymarket_meta_learner.py` - Trains Ridge/Logistic models to learn optimal agent weights

2. **Sensing Layer** (Phase 1: SENSE):
   - `polymarket_whale_flow_agent.py` - Tracks top wallets (≥60% win rate, ≥$10k bets)
   - `polymarket_event_catalyst_agent.py` - RSS monitoring + FinBERT sentiment analysis
   - `polymarket_anomaly_agent.py` - Z-score anomaly detection (price/volume/liquidity)

3. **Forecasting Layer** (Phase 2: THINK):
   - `polymarket_swarm_forecaster.py` - 6-model parallel consensus (Claude, GPT-4, DeepSeek, Grok, Gemini, Ollama)
   - `polymarket_llm_forecaster.py` - Bounded adjustment (±15% max) using learned weights

4. **Execution Layer** (Phases 3-5: DECIDE → TRADE → EXIT):
   - `polymarket_quant_layer.py` - Entry gates (EV, z-score, Kelly sizing, regime classification)
   - `polymarket_exit_manager.py` - 6-rule exit system (ANY trigger exits position)
   - `polymarket_orchestrator.py` - Master control system (coordinates all agents)

5. **Utilities**:
   - `polymarket_utils.py` - Shared utilities for all Polymarket agents

Each agent can run independently or as part of orchestration loops.

### Pinescript Conversion System

The system now includes **precise TradingView Pinescript → Python conversion** capabilities for backtesting complex strategies with zero loss in translation.

**Conversion Methodology**:
1. **Deep Analysis**: Thoroughly understand Pinescript logic before converting
2. **Exact Parameter Matching**: Preserve all default settings from original strategy
3. **Indicator Precision**: Handle custom libraries (loxx, etc.) and specialized functions
4. **Stop Management**: Implement complex multi-stop systems (trailing, ATR-based, band-based)
5. **Position Sizing**: Convert R-based, Kelly, or custom sizing methods accurately
6. **Validation**: Compare results on identical time periods to verify correctness

**Current Conversions**:
- **ADX + Squeeze [R-BASED]** (`src/data/pinescript_conversions/backtests/ADX_Squeeze_R_Based_BT_v2.py`)
  - Complex squeeze indicator using EMA crossovers with ATR threshold
  - ADX confluence detection (ADX > 20, +DI vs -DI, EMA 12 vs 50)
  - R-based position sizing with 0.5% risk per trade
  - 4 stop types: Initial ATR, Dynamic R Trailing, D-Bands, ATR Regime
  - Stop selection logic: tightest valid stop that doesn't loosen
  - Signal expiration system (13 bars for ADX, instant for squeeze)
  - Results: +6.83% on 2023 data, -1.55% on 2023-2025 hourly (strategy requires trends)

**Data Extraction Tools**:
- `extract_backtest_data.py` - Extracts daily NAV and complete trades list from backtests
  - Output: `backtest_daily_nav.csv` (date, NAV, daily return %)
  - Output: `backtest_trades.csv` (entry/exit times, prices, PnL, all indicators)
  - Includes all strategy indicators at entry/exit for analysis

**Data Downloaders**:
- `download_btc_15m_2025.py` - Last 60 days of BTC 15-minute bars (yfinance limit)
- `download_btc_1h_2020_2025.py` - 2 years of BTC hourly bars (17k+ bars)

**Key Files**:
- Converted strategies: `src/data/pinescript_conversions/backtests/`
- Analysis documents: `src/data/pinescript_conversions/analysis/` (e.g., `ADX_SQUEEZE_ANALYSIS.md`)
- OHLCV data: `src/data/rbi/BTC-USD-*.csv`

**Conversion Rules**:
- **"It is imperative that there is nothing lost in translation"** - user requirement
- Thoroughly understand Pinescript logic before converting
- Match exact default settings (not modified versions in other files)
- Test on comparable time periods to validate correctness
- Document all conversions with analysis files

### LLM Integration (Model Factory)

Located at `src/models/model_factory.py` and `src/models/README.md`

**Unified Interface**: All agents use `ModelFactory.create_model()` for consistent LLM access
**Supported Providers**: Anthropic Claude (default), OpenAI, DeepSeek, Groq, Google Gemini, Ollama (local)
**Key Pattern**:
```python
from src.models.model_factory import ModelFactory

model = ModelFactory.create_model('anthropic')  # or 'openai', 'deepseek', 'groq', etc.
response = model.generate_response(system_prompt, user_content, temperature, max_tokens)
```

### Configuration Management

**Primary Config**: `src/config.py`

**Crypto Trading Settings**:
- Trading settings: `MONITORED_TOKENS`, `EXCLUDED_TOKENS`, position sizing (`usd_size`, `max_usd_order_size`)
- Risk management: `CASH_PERCENTAGE`, `MAX_POSITION_PERCENTAGE`, `MAX_LOSS_USD`, `MAX_GAIN_USD`, `MINIMUM_BALANCE_USD`
- Agent behavior: `SLEEP_BETWEEN_RUNS_MINUTES`, `ACTIVE_AGENTS` dict in `main.py`
- AI settings: `AI_MODEL`, `AI_MAX_TOKENS`, `AI_TEMPERATURE`

**Polymarket Settings** (lines 126-292, 100+ parameters):
- Entry gates: `POLYMARKET_EV_MIN`, `POLYMARKET_Z_MIN`, `POLYMARKET_MAX_SPREAD`
- Position sizing: `POLYMARKET_KELLY_FRACTION`, `POLYMARKET_MAX_POSITION_SIZE`
- Exit rules: `POLYMARKET_EXIT_EV_DECAY`, `POLYMARKET_EXIT_PROFIT_TARGET`, `POLYMARKET_EXIT_STOP_LOSS`
- Whale tracking: `POLYMARKET_WHALE_MIN_BET_SIZE`, `POLYMARKET_WHALE_MIN_WIN_RATE`
- Regime multipliers: `POLYMARKET_REGIME_INFORMATION_MULT`, `POLYMARKET_REGIME_EMOTION_MULT`
- Feature flags: `POLYMARKET_ENABLED`, `POLYMARKET_PAPER_TRADING`, `POLYMARKET_USE_SWARM`

**Environment Variables**: `.env` (see `.env_example`)
- Trading APIs: `BIRDEYE_API_KEY`, `MOONDEV_API_KEY`, `COINGECKO_API_KEY`
- AI Services: `ANTHROPIC_KEY`, `OPENAI_KEY`, `DEEPSEEK_KEY`, `GROQ_API_KEY`, `GEMINI_KEY`
- Blockchain: `SOLANA_PRIVATE_KEY`, `HYPER_LIQUID_ETH_PRIVATE_KEY`, `RPC_ENDPOINT`

### Shared Utilities

**Crypto Trading**:
- `src/nice_funcs.py` (~1,200 lines): Core trading functions
  - Data: `token_overview()`, `token_price()`, `get_position()`, `get_ohlcv_data()`
  - Trading: `market_buy()`, `market_sell()`, `chunk_kill()`, `open_position()`
  - Analysis: Technical indicators, PnL calculations, rug pull detection
- `src/agents/api.py`: `MoonDevAPI` class for custom Moon Dev API endpoints
  - `get_liquidation_data()`, `get_funding_data()`, `get_oi_data()`, `get_copybot_follow_list()`

**Polymarket**:
- `src/polymarket_utils.py`: Shared utilities for all Polymarket agents
  - Market database management, querying, enrichment
  - Helper functions: `format_currency()`, `format_probability()`

### Data Flow Patterns

**Crypto Trading**:
```
Config/Input → Agent Init → API Data Fetch → Data Parsing →
LLM Analysis (via ModelFactory) → Decision Output →
Result Storage (CSV/JSON in src/data/) → Optional Trade Execution
```

**Polymarket Probability Arbitrage**:
```
Data Collection (24-48h) → Meta-Learning Training → calibration.json →

Live Trading Loop:
  SENSE (Whale + Event + Anomaly) →
  THINK (Swarm + LLM with learned weights) →
  DECIDE (Quant: 6 entry gates ALL must pass) →
  TRADE (Limit-at-fair execution) →
  EXIT (6 exit rules ANY triggers) →
  Update Meta-Learner (weekly retraining)
```

## Development Rules

### File Management
- **Keep files under 800 lines** - if longer, split into new files and update README
- **DO NOT move files without asking** - you can create new files but no moving
- **NEVER create new virtual environments** - use existing `conda activate tflow`
- **Update requirements.txt** after adding any new package

### Backtesting
- Use `backtesting.py` library (NOT their built-in indicators)
- Use `pandas_ta` or `talib` for technical indicators instead
- Sample data available at `/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv`

### Code Style
- **No fake/synthetic data** - always use real data or fail the script
- **Minimal error handling** - user wants to see errors, not over-engineered try/except blocks
- **No API key exposure** - never show keys from `.env` in output
- **Windows emoji handling**: Use UTF-8 encoding wrapper for Windows console compatibility

### Agent Development Pattern

When creating new agents:
1. Inherit from base patterns in existing agents
2. Use `ModelFactory` for LLM access
3. Store outputs in `src/data/[agent_name]/`
4. Make agent independently executable (standalone script)
5. Add configuration to `config.py` if needed
6. Follow naming: `[purpose]_agent.py`

### Testing Strategies

Place strategy definitions in `src/strategies/` folder:
```python
class YourStrategy(BaseStrategy):
    name = "strategy_name"
    description = "what it does"

    def generate_signals(self, token_address, market_data):
        return {
            "action": "BUY"|"SELL"|"NOTHING",
            "confidence": 0-100,
            "reasoning": "explanation"
        }
```

## Important Context

### Risk-First Philosophy
- Risk Agent runs first in main loop before any trading decisions
- Configurable circuit breakers (`MAX_LOSS_USD`, `MINIMUM_BALANCE_USD`)
- AI confirmation for position-closing decisions (configurable via `USE_AI_CONFIRMATION`)

### Data Sources

**Crypto Trading**:
1. **BirdEye API** - Solana token data (price, volume, liquidity, OHLCV)
2. **Moon Dev API** - Custom signals (liquidations, funding rates, OI, copybot data)
3. **CoinGecko API** - 15,000+ token metadata, market caps, sentiment
4. **Helius RPC** - Solana blockchain interaction

**Polymarket**:
1. **Polymarket API** - Public REST API (no authentication required!)
   - Markets: `https://gamma-api.polymarket.com/markets`
   - Order book: `https://clob.polymarket.com/book` (optional, requires auth)
2. **RSS Feeds** - News monitoring (Reuters, BBC, Politico, NYT)
3. **Twitter API** - Optional (can use free Twikit or skip entirely)

### Autonomous Execution

**Crypto Trading**:
- Main loop runs every 15 minutes by default (`SLEEP_BETWEEN_RUNS_MINUTES`)
- Agents handle errors gracefully and continue execution
- Keyboard interrupt for graceful shutdown
- All agents log to console with color-coded output (termcolor)

**Polymarket**:
- Data collector runs continuously (60-second intervals)
- Meta-learner trains weekly (or on-demand)
- Orchestrator monitors positions and checks exit rules every cycle
- Paper trading mode enabled by default (`POLYMARKET_PAPER_TRADING = True`)

### AI-Driven Strategy Generation (RBI Agent)
1. User provides: YouTube video URL / PDF / trading idea text
2. DeepSeek-R1 analyzes and extracts strategy logic
3. Generates backtesting.py compatible code
4. Executes backtest and returns performance metrics
5. Cost: ~$0.027 per backtest execution (~6 minutes)

## Common Patterns

### Adding New Agent
1. Create `src/agents/your_agent.py`
2. Implement standalone execution logic
3. Add to `ACTIVE_AGENTS` in `main.py` if needed for orchestration
4. Use `ModelFactory` for LLM calls
5. Store results in `src/data/your_agent/`

### Switching AI Models
Edit `config.py`:
```python
AI_MODEL = "claude-3-haiku-20240307"  # Fast, cheap
# AI_MODEL = "claude-3-sonnet-20240229"  # Balanced
# AI_MODEL = "claude-3-opus-20240229"  # Most powerful
```

Or use different models per agent via ModelFactory:
```python
model = ModelFactory.create_model('deepseek')  # Reasoning tasks
model = ModelFactory.create_model('groq')      # Fast inference
```

### Reading Market Data

**Crypto**:
```python
from src.nice_funcs import token_overview, get_ohlcv_data, token_price

# Get comprehensive token data
overview = token_overview(token_address)

# Get price history
ohlcv = get_ohlcv_data(token_address, timeframe='1H', days_back=3)

# Get current price
price = token_price(token_address)
```

**Polymarket**:
```python
from src.polymarket_utils import PolymarketUtils

utils = PolymarketUtils()

# Get market data
market = utils.get_market(market_id)

# Search markets
results = utils.search_markets(['election', 'president'])

# Enrich with calculations
enriched = utils.enrich_market_data(market)
```

## Polymarket System Quick Reference

**See `docs/POLYMARKET_PHILOSOPHY.md` for detailed explanation of core concepts.**

### Key Concepts
- **Probability Arbitrage**: Profit from market convergence, not binary outcomes
- **Meta-Learning**: Continuously train to learn which agents produce best results
- **6+6 Rule System**: 6 entry gates (ALL must pass) + 6 exit rules (ANY triggers)
- **Whale Tracking**: Monitor wallets with ≥60% win rate, ≥20 bets, ≥$10k positions
- **Regime Classification**: Information (1.0x), Illiquid (0.5x), Emotion (1.5x) sizing

### Three-Phase Setup
1. **Collect Data**: Run `polymarket_data_collector.py` for 24-48 hours
2. **Train Meta-Learner**: Run `polymarket_meta_learner.py` to generate `calibration.json`
3. **Live Trading**: Run `polymarket_orchestrator.py` (paper trading mode default)

### Critical Files
- `src/config.py` (lines 126-292): All Polymarket parameters
- `src/data/polymarket/meta_learning/calibration.json`: Learned agent weights
- `src/data/polymarket/positions/`: Open and closed positions
- `POLYMARKET_SYSTEM_COMPLETE.md`: Comprehensive system documentation
- `POLYMARKET_SETUP_GUIDE.md`: Step-by-step setup instructions
- `docs/POLYMARKET_PHILOSOPHY.md`: Why the system works this way

### Expected Behaviors
- Order book API errors (400/401) are **expected and acceptable** - optional feature
- Trades API errors (401) are **expected** - whale detection works without them
- Data collector auto-restarts after WiFi outage (no data loss from completed snapshots)
- Meta-learner requires minimum 1,440 snapshots (24 hours), optimal 10,080 (1 week)

## Project Philosophy

This is an **experimental, educational project** demonstrating AI agent patterns through algorithmic trading:
- No guarantees of profitability (substantial risk of loss)
- Open source and free for learning
- YouTube-driven development with weekly updates
- Community-supported via Discord
- No token associated with project (avoid scams)

The goal is to democratize AI agent development and show practical multi-agent orchestration patterns that can be applied beyond trading.

### Crypto Trading Focus
- Autonomous 24/7 trading across multiple DEXs
- Risk-first approach with circuit breakers
- Multi-agent collaboration for consensus decisions

### Polymarket Focus
- Probability arbitrage (not binary outcome betting)
- Continuous learning via meta-learning framework
- Profit from market convergence and mispricing detection
- Insider trading detection via whale wallet monitoring
