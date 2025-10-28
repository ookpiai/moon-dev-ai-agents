"""
🌙 Moon Dev's Configuration File
Built with love by Moon Dev 🚀
"""

# 🔄 Exchange Selection
EXCHANGE = 'solana'  # Options: 'solana', 'hyperliquid'

# 💰 Trading Configuration
USDC_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # Never trade or close
SOL_ADDRESS = "So11111111111111111111111111111111111111111"   # Never trade or close

# Create a list of addresses to exclude from trading/closing
EXCLUDED_TOKENS = [USDC_ADDRESS, SOL_ADDRESS]

# Token List for Trading 📋
# NOTE: Trading Agent now has its own token list - see src/agents/trading_agent.py lines 101-104
MONITORED_TOKENS = [
    # '9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump',    # 🌬️ FART
    # 'DitHyRMQiSDhn5cnKMJV2CDDt6sVct96YrECiM49pump'     # housecoin
]

# Moon Dev's Token Trading List 🚀
# Each token is carefully selected by Moon Dev for maximum moon potential! 🌙
tokens_to_trade = MONITORED_TOKENS  # Using the same list for trading

# ⚡ HyperLiquid Configuration
HYPERLIQUID_SYMBOLS = ['BTC', 'ETH', 'SOL']  # Symbols to trade on HyperLiquid perps
HYPERLIQUID_LEVERAGE = 5  # Default leverage for HyperLiquid trades (1-50)

# 🔄 Exchange-Specific Token Lists
# Use this to determine which tokens/symbols to trade based on active exchange
def get_active_tokens():
    """Returns the appropriate token/symbol list based on active exchange"""
    if EXCHANGE == 'hyperliquid':
        return HYPERLIQUID_SYMBOLS
    else:
        return MONITORED_TOKENS

# Token to Exchange Mapping (for future hybrid trading)
TOKEN_EXCHANGE_MAP = {
    'BTC': 'hyperliquid',
    'ETH': 'hyperliquid',
    'SOL': 'hyperliquid',
    # All other tokens default to Solana
}

# Token and wallet settings
symbol = '9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump'
address = '4wgfCBf2WwLSRKLef9iW7JXZ2AfkxUxGM4XcKpHm3Sin' # YOUR WALLET ADDRESS HERE

# Position sizing 🎯
usd_size = 25  # Size of position to hold
max_usd_order_size = 3  # Max order size
tx_sleep = 30  # Sleep between transactions
slippage = 199  # Slippage settings

# Risk Management Settings 🛡️
CASH_PERCENTAGE = 20  # Minimum % to keep in USDC as safety buffer (0-100)
MAX_POSITION_PERCENTAGE = 30  # Maximum % allocation per position (0-100)
STOPLOSS_PRICE = 1 # NOT USED YET 1/5/25    
BREAKOUT_PRICE = .0001 # NOT USED YET 1/5/25
SLEEP_AFTER_CLOSE = 600  # Prevent overtrading

MAX_LOSS_GAIN_CHECK_HOURS = 12  # How far back to check for max loss/gain limits (in hours)
SLEEP_BETWEEN_RUNS_MINUTES = 15  # How long to sleep between agent runs 🕒


# Max Loss/Gain Settings FOR RISK AGENT 1/5/25
USE_PERCENTAGE = False  # If True, use percentage-based limits. If False, use USD-based limits

# USD-based limits (used if USE_PERCENTAGE is False)
MAX_LOSS_USD = 25  # Maximum loss in USD before stopping trading
MAX_GAIN_USD = 25 # Maximum gain in USD before stopping trading

# USD MINIMUM BALANCE RISK CONTROL
MINIMUM_BALANCE_USD = 50  # If balance falls below this, risk agent will consider closing all positions
USE_AI_CONFIRMATION = True  # If True, consult AI before closing positions. If False, close immediately on breach

# Percentage-based limits (used if USE_PERCENTAGE is True)
MAX_LOSS_PERCENT = 5  # Maximum loss as percentage (e.g., 20 = 20% loss)
MAX_GAIN_PERCENT = 5  # Maximum gain as percentage (e.g., 50 = 50% gain)

# Transaction settings ⚡
slippage = 199  # 500 = 5% and 50 = .5% slippage
PRIORITY_FEE = 100000  # ~0.02 USD at current SOL prices
orders_per_open = 3  # Multiple orders for better fill rates

# Market maker settings 📊
buy_under = .0946
sell_over = 1

# Data collection settings 📈
DAYSBACK_4_DATA = 3
DATA_TIMEFRAME = '1H'  # 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 8H, 12H, 1D, 3D, 1W, 1M
SAVE_OHLCV_DATA = False  # 🌙 Set to True to save data permanently, False will only use temp data during run

# AI Model Settings 🤖
AI_MODEL = "claude-3-haiku-20240307"  # Model Options:
                                     # - claude-3-haiku-20240307 (Fast, efficient Claude model)
                                     # - claude-3-sonnet-20240229 (Balanced Claude model)
                                     # - claude-3-opus-20240229 (Most powerful Claude model)
AI_MAX_TOKENS = 1024  # Max tokens for response
AI_TEMPERATURE = 0.7  # Creativity vs precision (0-1)

# Trading Strategy Agent Settings - MAY NOT BE USED YET 1/5/25
ENABLE_STRATEGIES = True  # Set this to True to use strategies
STRATEGY_MIN_CONFIDENCE = 0.7  # Minimum confidence to act on strategy signals

# Sleep time between main agent runs
SLEEP_BETWEEN_RUNS_MINUTES = 15  # How long to sleep between agent runs 🕒

# in our nice_funcs in token over view we look for minimum trades last hour
MIN_TRADES_LAST_HOUR = 2


# Real-Time Clips Agent Settings 🎬
REALTIME_CLIPS_ENABLED = True
REALTIME_CLIPS_OBS_FOLDER = '/Volumes/Moon 26/OBS'  # Your OBS recording folder
REALTIME_CLIPS_AUTO_INTERVAL = 120  # Check every N seconds (120 = 2 minutes)
REALTIME_CLIPS_LENGTH = 2  # Minutes to analyze per check
REALTIME_CLIPS_AI_MODEL = 'groq'  # Model type: groq, openai, claude, deepseek, xai, ollama
REALTIME_CLIPS_AI_MODEL_NAME = None  # None = use default for model type
REALTIME_CLIPS_TWITTER = True  # Auto-open Twitter compose after clip

# ═══════════════════════════════════════════════════════════════════════════════
# 🎲 POLYMARKET CONFIGURATION - Probability Arbitrage System
# ═══════════════════════════════════════════════════════════════════════════════

# 📊 Market Data Settings
POLYMARKET_MARKETS_CSV = 'data/polymarket_markets.csv'  # Path to markets database
POLYMARKET_DATA_DIR = 'src/data/polymarket'  # Base directory for agent data
POLYMARKET_SCAN_INTERVAL_MINUTES = 10  # Lightweight scanner interval (default: 10 minutes)

# 🧮 Quant Layer Thresholds - Entry Gates
POLYMARKET_EV_MIN = 0.03  # Minimum expected value (3% edge after costs)
POLYMARKET_Z_MIN = 1.5  # Minimum z-score for statistical significance (1.5 sigma)
POLYMARKET_MAX_SPREAD = 0.06  # Maximum spread to accept (6% = 0.06)
POLYMARKET_MIN_LIQUIDITY = 10000  # Minimum market liquidity in USD ($10,000)
POLYMARKET_MIN_VOLUME_24H = 1000  # Minimum 24h volume in USD ($1,000)
POLYMARKET_MAX_DAYS_TO_RESOLUTION = 90  # Only trade markets resolving within 90 days

# Entry requires ALL of: EV_net ≥ 0.03, z ≥ 1.5, spread ≤ 0.06, liquidity ≥ $10k

# 💰 Position Sizing - Fractional Kelly
POLYMARKET_KELLY_FRACTION = 0.25  # Use 25% of full Kelly (aggressive but safe)
POLYMARKET_MAX_PER_MARKET = 0.15  # Max 15% of capital per market
POLYMARKET_MAX_PER_THEME = 0.30  # Max 30% of capital per theme (e.g., "Election 2024")
POLYMARKET_MIN_POSITION_SIZE = 50  # Minimum position size in USD
POLYMARKET_MAX_POSITION_SIZE = 1000  # Maximum position size in USD

# Position size formula:
# f = KELLY_FRACTION × |edge| × confidence × regime_multiplier
# Constrained by: MIN_POSITION_SIZE ≤ f ≤ min(MAX_PER_MARKET, MAX_PER_THEME, MAX_POSITION_SIZE)

# 📈 Regime Classification - Adaptive Sizing Multipliers
POLYMARKET_REGIME_INFORMATION_MULT = 1.0  # Normal size in Information regime (tight spreads, high volume)
POLYMARKET_REGIME_ILLIQUID_MULT = 0.5  # Half size in Illiquid regime (wide spreads, low volume)
POLYMARKET_REGIME_EMOTION_MULT = 1.5  # 1.5x size in Emotion regime (panic/euphoria, high volume, wide spreads)

# Regime Detection Rules:
# Information: spread ≤ 0.04 AND volume_24h ≥ $5k AND liquidity ≥ $15k
# Illiquid: spread > 0.06 OR volume_24h < $2k OR liquidity < $8k
# Emotion: spread > 0.05 AND volume_24h ≥ $10k AND price_change_1h > 10%

# 🚪 Exit Rules - Multi-Trigger System (ANY trigger exits position)
POLYMARKET_EXIT_EV_DECAY = 0.01  # Exit if EV_net falls below 1% (originally entered at ≥3%)
POLYMARKET_EXIT_Z_REVERSION = 0.8  # Exit if z-score reverts to ≤ 0.8 (originally entered at ≥1.5)
POLYMARKET_EXIT_TRAILING_EV_ALPHA = 0.7  # Trailing EV: exit if current_EV < alpha × peak_EV (70% of peak)
POLYMARKET_EXIT_TIME_GATE_DAYS = 7  # Exit if no 30%+ EV improvement in 7 days
POLYMARKET_EXIT_SIGNAL_REVERSAL = True  # Exit if 3+ sensing agents flip bearish
POLYMARKET_EXIT_PROFIT_TARGET = 0.08  # Exit at 8% profit target
POLYMARKET_EXIT_STOP_LOSS = -0.03  # Exit at -3% stop loss

# Exit Trigger Details:
# 1. EV Decay: EV_net < 0.01
# 2. Z-Score Reversion: z < 0.8
# 3. Trailing EV: current_EV < 0.7 × peak_EV
# 4. Time Gate: No 30%+ EV improvement in 7 days
# 5. Signal Reversal: ≥3 agents flip from bullish to bearish
# 6. Profit Target/Stop Loss: ±8% / -3%

# 🎯 Execution Settings - Limit-at-Fair Strategy
POLYMARKET_LIMIT_AT_FAIR = True  # Always place limit orders at fair value
POLYMARKET_NEVER_CHASE = True  # Never use market orders or chase price
POLYMARKET_MAX_WAIT_SECONDS = 300  # Cancel unfilled limit orders after 5 minutes
POLYMARKET_REORDER_ON_PRICE_MOVE = 0.02  # Re-place order if fair value moves ±2%
POLYMARKET_MAX_SLIPPAGE = 0.01  # Maximum acceptable slippage (1%) if limit fails

# Execution Strategy:
# 1. Calculate fair_value = TRUE_PROB
# 2. Place limit order at fair_value
# 3. Wait up to MAX_WAIT_SECONDS
# 4. If unfilled and price moved ≥2%, cancel and re-place at new fair_value
# 5. NEVER use market orders (always limit-at-fair)

# 🐋 Whale Flow Settings
POLYMARKET_WHALE_MIN_BET_SIZE = 10000  # Track bets ≥ $10,000
POLYMARKET_WHALE_MIN_WIN_RATE = 0.60  # Only follow wallets with ≥60% historical win rate
POLYMARKET_WHALE_MIN_SAMPLE_SIZE = 20  # Require ≥20 historical bets for win rate calculation
POLYMARKET_WHALE_COPY_THRESHOLD = 0.75  # Auto-copy if whale + 3 other signals agree (75% consensus)
POLYMARKET_WHALE_DECAY_HOURS = 24  # Whale signal decays over 24 hours

# Whale Detection:
# - Monitor top 100 wallets by total volume
# - Track all bets ≥ $10k across all markets
# - Calculate historical win rate (require ≥20 bets)
# - Only signal on wallets with ≥60% win rate

# 📡 Event Catalyst Settings
POLYMARKET_EVENT_TWITTER_KEYWORDS = ['breaking', 'just in', 'confirmed', 'official', 'announced']
POLYMARKET_EVENT_NEWS_SOURCES = ['Reuters', 'AP', 'Bloomberg', 'WSJ', 'CNN', 'BBC']
POLYMARKET_EVENT_SENTIMENT_THRESHOLD = 0.3  # FinBERT sentiment magnitude threshold
POLYMARKET_EVENT_VOLUME_SPIKE_MULT = 3.0  # Flag if volume > 3x normal
POLYMARKET_EVENT_CHECK_INTERVAL_SEC = 60  # Check for new events every 60 seconds

# Event Catalyst Agent:
# - Monitors Twitter for breaking news (via keywords)
# - Tracks verified news sources
# - FinBERT sentiment analysis on event impact
# - Flags unusual volume spikes (≥3x normal)

# 🔬 Anomaly Detection Settings
POLYMARKET_ANOMALY_Z_THRESHOLD = 2.0  # Flag price movements ≥2 standard deviations
POLYMARKET_ANOMALY_WINDOW_HOURS = 24  # Calculate z-scores over 24-hour rolling window
POLYMARKET_ANOMALY_MIN_DATA_POINTS = 20  # Require ≥20 data points for z-score calculation
POLYMARKET_ANOMALY_VOLUME_Z_THRESHOLD = 2.5  # Flag volume spikes ≥2.5 sigma
POLYMARKET_ANOMALY_CHECK_INTERVAL_SEC = 120  # Check for anomalies every 2 minutes

# Anomaly Detection:
# - Track price, volume, liquidity changes
# - Calculate rolling z-scores (24h window)
# - Flag: price_z ≥ 2.0 OR volume_z ≥ 2.5
# - Require ≥20 data points for statistical validity

# 🧠 Swarm Forecaster Settings (6-Model Consensus)
POLYMARKET_SWARM_MODELS = {
    'claude': True,      # Claude Sonnet 4.5 - Balanced performance
    'openai': False,     # GPT-5 (disabled - quota exceeded)
    'deepseek': True,    # DeepSeek Chat - Cheapest
    'groq': False,       # Llama 3.1 8B (disabled - network/firewall issue)
    'gemini': False,     # Gemini 2.5 Flash (disabled - empty responses)
    'xai': False,        # Grok-4 (disabled - no API key)
    'ollama': False,     # DeepSeek-R1 Local (disabled - requires local setup)
}
POLYMARKET_SWARM_TIMEOUT_SEC = 45  # Max 45 seconds per model
POLYMARKET_SWARM_CONSENSUS_MIN = 2  # Require ≥2 models to agree for consensus (reduced for 3-model setup)
POLYMARKET_SWARM_DISAGREEMENT_THRESHOLD = 0.15  # Flag if model spread ≥ 15%

# Swarm Forecaster:
# - 6 models run in parallel (ThreadPoolExecutor)
# - Each outputs probability + reasoning
# - Consensus = median of all models
# - Flag high disagreement (spread ≥15%) as low confidence

# 🤖 LLM Forecaster Settings (Bounded Adjustment)
POLYMARKET_LLM_BASE_MODEL = 'claude'  # Base model for LLM forecasting
POLYMARKET_LLM_MAX_ADJUSTMENT = 0.15  # Max ±15% adjustment from swarm prior
POLYMARKET_LLM_MIN_CONFIDENCE = 0.6  # Minimum confidence to use LLM forecast (0-1)
POLYMARKET_LLM_REASONING_REQUIRED = True  # Require explicit reasoning in output

# LLM Bounded Forecasting:
# - Starts with swarm_consensus as prior
# - LLM adjusts ±10-15% maximum
# - Formula: final_prob = swarm_prior + bounded_adjustment
# - Prevents overconfidence and anchors to multi-model consensus

# 📊 Risk Management - Portfolio Level
POLYMARKET_MAX_OPEN_POSITIONS = 10  # Maximum concurrent positions
POLYMARKET_MAX_DAILY_TRADES = 20  # Maximum trades per day (prevent overtrading)
POLYMARKET_MAX_DAILY_LOSS_USD = 200  # Stop trading if daily loss ≥ $200
POLYMARKET_MIN_ACCOUNT_BALANCE_USD = 500  # Minimum account balance to continue trading

# Portfolio Risk:
# - Max 10 open positions (diversification)
# - Max 20 trades/day (overtrading protection)
# - Circuit breaker at -$200 daily loss
# - Shut down if account < $500

# 🎛️ Feature Flags
POLYMARKET_ENABLED = True  # Master switch for Polymarket trading
POLYMARKET_PAPER_TRADING = True  # Start with paper trading (no real orders)
POLYMARKET_USE_SWARM = True  # Enable 6-model swarm forecasting
POLYMARKET_USE_LLM_ADJUSTMENT = False  # Enable bounded LLM adjustment (DISABLED - fixes Claude model 404 error)
POLYMARKET_USE_WHALE_FLOW = True  # Enable whale wallet monitoring
POLYMARKET_USE_EVENT_CATALYST = True  # Enable news/event monitoring
POLYMARKET_USE_ANOMALY_DETECTION = True  # Enable statistical anomaly detection
POLYMARKET_VERBOSE_LOGGING = True  # Detailed logs for debugging

# Feature Toggles:
# - Set PAPER_TRADING=True for testing (no real money)
# - Disable individual agents as needed
# - VERBOSE_LOGGING includes EV calculations, z-scores, and full reasoning

# Future variables (not active yet) 🔮
sell_at_multiple = 3
USDC_SIZE = 1
limit = 49
timeframe = '15m'
stop_loss_perctentage = -.24
EXIT_ALL_POSITIONS = False
DO_NOT_TRADE_LIST = ['777']
CLOSED_POSITIONS_TXT = '777'
minimum_trades_in_last_hour = 777
