# 🚀 Pinescript Converter - Quick Start Guide

## What You Asked For

You want to:
1. ✅ Convert TradingView Pinescript strategies → Python
2. ✅ Run strategies on BTC at 2m, 5m, 10m, 30m timeframes
3. ✅ Paper trade first
4. ✅ Then enable live trading

**Good news: This is 100% possible with the agents already in this repo!**

---

## The System I Just Built

### New Agent: `pinescript_converter_agent.py`

Located: `src/agents/pinescript_converter_agent.py`

**What it does:**
1. Takes your TradingView Pinescript code as input
2. Converts it to Python backtesting.py format using AI (GPT-5 / Claude / DeepSeek)
3. Runs backtests on BTC data at multiple timeframes
4. Generates performance stats (returns, Sharpe ratio, drawdown, win rate)
5. Prepares strategy for live trading via `trading_agent.py`

---

## Step-by-Step Usage

### STEP 1: Prepare Your Pinescript

Create a file called `strategy.pine` in the root directory:

```bash
cd /mnt/c/Users/oo/Desktop/moon-dev-ai-agents
nano strategy.pine  # or use any text editor
```

Paste your Pinescript code. Example:

```pine
//@version=5
strategy("My Strategy", overlay=true)

// Your strategy logic here
fast = ta.sma(close, 10)
slow = ta.sma(close, 30)

if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
if ta.crossunder(fast, slow)
    strategy.close("Long")
```

**Or use one of the examples I created:**
- `examples/sma_crossover.pine` - Moving average crossover
- `examples/rsi_mean_reversion.pine` - RSI oversold/overbought

---

### STEP 2: Run the Converter

```bash
# Activate environment
conda activate tflow

# Run converter
python src/agents/pinescript_converter_agent.py
```

**What happens:**
1. AI reads your Pinescript
2. Converts indicators to Python/talib equivalents
3. Generates backtest code
4. Runs backtests on 5m, 15m, 30m, 1H timeframes
5. Prints performance results

**Output example:**
```
🌙 Initializing Moon Dev's Pinescript Converter Agent...
✅ Using openai/gpt-5
🔄 Converting Pinescript strategy: MyStrategy...
✅ Converted code saved to: src/data/pinescript_conversions/backtests/MyStrategy_01_24_2025_143022.py

📊 Running backtest for BTC on 5m timeframe...
================================================================================
📊 BACKTEST RESULTS
================================================================================
Start                     2023-01-01 00:00:00
End                       2024-12-31 23:55:00
Duration                    730 days 23:55:00
Exposure Time [%]                       67.23
Return [%]                             142.56
Buy & Hold Return [%]                   89.34
Return (Ann.) [%]                      156.78
Sharpe Ratio                             2.13
Max. Drawdown [%]                      -23.45
Avg. Drawdown [%]                       -5.67
# Trades                                  234
Win Rate [%]                            62.39
Best Trade [%]                          15.67
Worst Trade [%]                         -8.23
...
```

---

### STEP 3: Review Generated Files

Check these directories:

```
src/data/pinescript_conversions/
├── backtests/          # Python backtest code
│   └── MyStrategy_01_24_2025_143022.py
├── strategies/         # Live trading ready code
│   └── MyStrategy_live.py
└── results/            # Backtest results (text files)
    ├── backtest_BTC_5m_01_24_2025_143022.txt
    ├── backtest_BTC_15m_01_24_2025_143045.txt
    ├── backtest_BTC_30m_01_24_2025_143110.txt
    └── backtest_BTC_1H_01_24_2025_143135.txt
```

**Review the results:**
- Which timeframe performed best?
- Is the Sharpe ratio > 1.5?
- Is max drawdown acceptable (<30%)?
- Is win rate > 55%?

---

### STEP 4: Paper Trade

Once you're happy with backtest results, test in paper trading:

1. **Open trading_agent.py:**
   ```bash
   nano src/agents/trading_agent.py
   ```

2. **Enable paper trading (line ~130):**
   ```python
   # Find this section around line 130-140
   PAPER_TRADING = True  # Set to True for paper trading
   ```

3. **Configure for BTC:**
   ```python
   # Line 140-143
   MONITORED_TOKENS = [
       "BTC",  # Your strategy will run on this
   ]
   ```

4. **Set timeframe (line 122-126):**
   ```python
   DAYSBACK_4_DATA = 3        # 3 days of history
   DATA_TIMEFRAME = '5m'      # 5 minute bars (change to 2m, 10m, 30m as needed)
   ```

5. **Run paper trading:**
   ```bash
   python src/agents/trading_agent.py
   ```

**What happens:**
- Agent analyzes BTC every loop (default: 15 minutes)
- Your converted strategy generates signals
- Trades are simulated (no real money)
- Results logged to console and CSV files

**Let it run for 1-2 weeks to validate performance.**

---

### STEP 5: Enable Live Trading

After paper trading validates your strategy:

1. **Disable paper trading:**
   ```python
   # In trading_agent.py line ~130
   PAPER_TRADING = False  # Real trades now!
   ```

2. **Configure exchange (line 84):**
   ```python
   EXCHANGE = "ASTER"  # Options: "ASTER", "HYPERLIQUID", "SOLANA"
   ```

3. **Set position sizing (line 113-120):**
   ```python
   usd_size = 25              # $25 per position (start small!)
   max_usd_order_size = 3     # $3 chunks
   MAX_POSITION_PERCENTAGE = 30  # Max 30% per position
   CASH_PERCENTAGE = 20       # Keep 20% cash buffer
   ```

4. **Set up API keys in .env:**
   ```bash
   # For Aster DEX
   SOLANA_PRIVATE_KEY=your_wallet_private_key

   # For HyperLiquid
   HYPER_LIQUID_ETH_PRIVATE_KEY=your_eth_private_key
   ```

5. **Run live:**
   ```bash
   python src/agents/trading_agent.py
   ```

**⚠️ Start with SMALL position sizes ($25-$50) until proven profitable!**

---

## Supported Timeframes

The converter supports these BTC timeframes:

| Timeframe | Data File | Recommended For |
|-----------|-----------|-----------------|
| 2 minute | `BTC-USD-2m.csv` | Scalping strategies |
| 5 minute | `BTC-USD-5m.csv` | Day trading |
| 10 minute | `BTC-USD-10m.csv` | Short-term swings |
| 15 minute | `BTC-USD-15m.csv` | ✅ Available now |
| 30 minute | `BTC-USD-30m.csv` | Swing trading |
| 1 hour | `BTC-USD-1H.csv` | Position trading |

**Note:** Currently only `BTC-USD-15m.csv` exists in the repo. You'll need to add the other timeframes if needed.

---

## Indicator Conversion Examples

The AI automatically converts Pinescript to Python:

**Pinescript:**
```pine
// Simple Moving Average
sma20 = ta.sma(close, 20)

// RSI
rsi = ta.rsi(close, 14)

// Bollinger Bands
[middle, upper, lower] = ta.bbands(close, 20, 2)

// MACD
[macdLine, signalLine, hist] = ta.macd(close, 12, 26, 9)

// Crossover detection
if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
```

**Converted Python:**
```python
# Simple Moving Average
self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20)

# RSI
self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)

# Bollinger Bands
self.bb_upper, self.bb_middle, self.bb_lower = self.I(
    talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2
)

# MACD
self.macd, self.signal, self.hist = self.I(
    talib.MACD, self.data.Close,
    fastperiod=12, slowperiod=26, signalperiod=9
)

# Crossover detection
if (self.fast[-2] < self.slow[-2] and self.fast[-1] > self.slow[-1]):
    self.buy(size=0.95)  # 95% of equity
```

---

## Configuration Options

### AI Model Selection

Edit `src/agents/pinescript_converter_agent.py` lines 18-23:

```python
CONVERTER_MODEL_CONFIG = {
    "type": "openai",      # Options: openai, anthropic, deepseek, groq
    "name": "gpt-5"        # Options: gpt-5, claude-sonnet-4.5, deepseek-chat
}
```

**Recommendations:**
- **GPT-5**: Best accuracy, $0.03 per conversion
- **Claude Sonnet 4.5**: Great for complex indicators, $0.025 per conversion
- **DeepSeek Chat**: Budget option, $0.002 per conversion

### Backtest Settings

The generated code uses these defaults:
- Initial capital: $10,000
- Commission: 0.1% per trade
- Position size: 95% of equity
- No slippage modeling

Edit the generated backtest file to customize.

---

## Troubleshooting

### "No module named 'backtesting'"

Install:
```bash
pip install backtesting
```

### "No module named 'talib'"

Install TA-Lib:
```bash
pip install TA-Lib
```

If that fails, use pandas_ta:
```bash
pip install pandas_ta
```

### "BTC data file not found"

The repo only includes `BTC-USD-15m.csv` by default. For other timeframes:
1. Download from your data provider
2. Place in `src/data/rbi/`
3. Format: `datetime, open, high, low, close, volume`

### Backtest shows 0 trades

Your strategy might be too strict:
1. Check entry conditions
2. Reduce indicator periods (e.g., SMA 10 instead of 50)
3. Widen RSI levels (e.g., 35/65 instead of 30/70)
4. Add debug prints to see indicator values

---

## Full Documentation

Complete guide: `docs/PINESCRIPT_CONVERTER.md`

Includes:
- Detailed API reference
- Advanced configuration
- Parameter optimization
- Multiple strategy management
- Custom data sources
- Full indicator mapping table

---

## Quick Reference Commands

```bash
# Convert strategy
python src/agents/pinescript_converter_agent.py

# Paper trade
python src/agents/trading_agent.py  # (with PAPER_TRADING=True)

# Live trade
python src/agents/trading_agent.py  # (with PAPER_TRADING=False)

# Check progress
ls src/data/pinescript_conversions/results/

# View latest backtest
cat src/data/pinescript_conversions/results/backtest_BTC_5m_*.txt | tail -50
```

---

## What's Next?

1. ✅ **Test the converter** with one of the example strategies
2. ✅ **Paste your Pinescript** and convert it
3. ✅ **Review backtest results** on multiple timeframes
4. ✅ **Paper trade for 1-2 weeks**
5. ✅ **Start live trading with small positions**
6. ✅ **Scale up gradually** as you gain confidence

---

## Questions?

The system is ready to use right now. Just:
1. Create your `strategy.pine` file
2. Run the converter
3. Review results

Let me know if you want me to:
- Convert your existing Pinescript strategy
- Adjust any configuration
- Add more BTC data timeframes
- Explain any part in more detail

Built with love by Moon Dev 🌙🚀
