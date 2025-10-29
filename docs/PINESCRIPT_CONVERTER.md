# 🌙 TradingView Pinescript Converter Agent

## Overview

The **Pinescript Converter Agent** automatically converts your TradingView Pinescript strategies into Python backtesting code and prepares them for live trading via the Trading Agent.

**Full Workflow:**
1. Input TradingView Pinescript →
2. Convert to Python (backtesting.py) →
3. Backtest on BTC historical data →
4. Paper trade via Trading Agent →
5. Enable live trading

---

## Quick Start

### Step 1: Create Your Pinescript File

Create a file called `strategy.pine` in the root directory:

```pine
//@version=5
strategy("My Strategy", overlay=true)

// Inputs
fastLength = input.int(10, "Fast SMA")
slowLength = input.int(30, "Slow SMA")

// Indicators
fastSMA = ta.sma(close, fastLength)
slowSMA = ta.sma(close, slowLength)

// Entry conditions
longCondition = ta.crossover(fastSMA, slowSMA)
exitCondition = ta.crossunder(fastSMA, slowSMA)

// Execute trades
if (longCondition)
    strategy.entry("Long", strategy.long)
if (exitCondition)
    strategy.close("Long")

// Plot
plot(fastSMA, color=color.blue, title="Fast SMA")
plot(slowSMA, color=color.red, title="Slow SMA")
```

### Step 2: Run the Converter

```bash
conda activate tflow
python src/agents/pinescript_converter_agent.py
```

### Step 3: Review Generated Files

The agent creates:

```
src/data/pinescript_conversions/
├── backtests/          # Python backtest code
│   └── MyStrategy_12_25_2025_143022.py
├── strategies/         # Live trading strategies
│   └── MyStrategy_live.py
└── results/            # Backtest results
    └── backtest_BTC_5m_12_25_2025_143022.txt
```

### Step 4: Review Backtest Results

The agent automatically backtests your strategy on multiple timeframes:
- 5 minute bars
- 15 minute bars
- 30 minute bars
- 1 hour bars

Results show:
- Total Return
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Number of Trades
- Average Trade Duration

### Step 5: Paper Trade (Optional)

Before going live, test in paper trading mode:

1. Open `src/agents/trading_agent.py`
2. Set `PAPER_TRADING = True` (around line 130)
3. Add your strategy to monitored tokens

### Step 6: Enable Live Trading

Once validated in paper trading:

1. Set `PAPER_TRADING = False`
2. Configure position sizing
3. Set risk management parameters
4. Run `python src/agents/trading_agent.py`

---

## Supported Indicators

The converter automatically maps Pinescript indicators to Python:

| Pinescript | Python (talib) |
|-----------|---------------|
| `ta.sma(close, 20)` | `self.I(talib.SMA, self.data.Close, timeperiod=20)` |
| `ta.ema(close, 20)` | `self.I(talib.EMA, self.data.Close, timeperiod=20)` |
| `ta.rsi(close, 14)` | `self.I(talib.RSI, self.data.Close, timeperiod=14)` |
| `ta.macd(close, 12, 26, 9)` | `self.I(talib.MACD, self.data.Close, 12, 26, 9)` |
| `ta.bbands(close, 20, 2)` | `self.I(talib.BBANDS, self.data.Close, 20, 2, 2)` |
| `ta.atr(14)` | `self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)` |
| `ta.stoch(14, 3, 3)` | `self.I(talib.STOCH, self.data.High, self.data.Low, self.data.Close, 14, 3, 3)` |
| `ta.crossover(a, b)` | `(a[-2] < b[-2] and a[-1] > b[-1])` |
| `ta.crossunder(a, b)` | `(a[-2] > b[-2] and a[-1] < b[-1])` |

---

## Configuration

Edit `src/agents/pinescript_converter_agent.py` lines 18-23:

```python
# Model configuration for conversion
CONVERTER_MODEL_CONFIG = {
    "type": "openai",  # Options: "openai", "anthropic", "deepseek", "groq"
    "name": "gpt-5"    # Options: "gpt-5", "claude-sonnet-4.5", "deepseek-chat"
}
```

**Recommended Models:**
- **GPT-5**: Best accuracy for complex strategies
- **Claude Sonnet 4.5**: Best for indicator-heavy strategies
- **DeepSeek Chat**: Cost-effective alternative

---

## Available BTC Data

The system includes BTC historical data at multiple timeframes:

- `BTC-USD-2m.csv` - 2 minute bars
- `BTC-USD-5m.csv` - 5 minute bars
- `BTC-USD-10m.csv` - 10 minute bars
- `BTC-USD-15m.csv` - 15 minute bars
- `BTC-USD-30m.csv` - 30 minute bars
- `BTC-USD-1H.csv` - 1 hour bars

All located in: `src/data/rbi/`

**Data Format:**
```csv
datetime, open, high, low, close, volume
2023-01-01 00:00:00, 16531.83, 16532.69, 16509.11, 16510.82, 231.05338022
2023-01-01 00:05:00, 16509.78, 16534.66, 16509.11, 16533.43, 308.12276951
```

---

## Common Pinescript Patterns

### 1. Simple Moving Average Crossover

```pine
//@version=5
strategy("SMA Cross", overlay=true)

fast = ta.sma(close, 10)
slow = ta.sma(close, 30)

if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
if ta.crossunder(fast, slow)
    strategy.close("Long")
```

### 2. RSI Oversold/Overbought

```pine
//@version=5
strategy("RSI Strategy", overlay=false)

rsi = ta.rsi(close, 14)

if rsi < 30  // Oversold
    strategy.entry("Long", strategy.long)
if rsi > 70  // Overbought
    strategy.close("Long")
```

### 3. Bollinger Band Breakout

```pine
//@version=5
strategy("BB Breakout", overlay=true)

[middle, upper, lower] = ta.bbands(close, 20, 2)

if close > upper
    strategy.entry("Long", strategy.long)
if close < middle
    strategy.close("Long")
```

### 4. MACD Crossover

```pine
//@version=5
strategy("MACD Strategy", overlay=false)

[macdLine, signalLine, histLine] = ta.macd(close, 12, 26, 9)

if ta.crossover(macdLine, signalLine)
    strategy.entry("Long", strategy.long)
if ta.crossunder(macdLine, signalLine)
    strategy.close("Long")
```

---

## Integration with Trading Agent

### Automatic Integration

Once your strategy is converted and validated:

1. **Copy generated strategy** from `src/data/pinescript_conversions/strategies/`
2. **Paste to** `src/strategies/custom/`
3. **Edit trading_agent.py** to import your strategy
4. **Configure in config.py**:
   ```python
   MONITORED_TOKENS = [
       "BTC",  # Your strategy will run on this
   ]

   # Position sizing
   usd_size = 25  # $25 per position
   max_usd_order_size = 3  # $3 order chunks
   ```

### Trading Agent Modes

**Paper Trading Mode:**
```python
# In trading_agent.py line ~130
PAPER_TRADING = True  # Simulate trades
```

**Live Trading Mode:**
```python
PAPER_TRADING = False  # Execute real trades
EXCHANGE = "ASTER"  # or "HYPERLIQUID", "SOLANA"
```

### Timeframe Configuration

Edit `trading_agent.py` lines 122-126:

```python
DAYSBACK_4_DATA = 3  # Days of historical data
DATA_TIMEFRAME = '5m'  # Options: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H
```

---

## Advanced Usage

### Custom Data Path

If you want to use different BTC data:

1. Place CSV file in `src/data/rbi/`
2. Format: `datetime, open, high, low, close, volume`
3. Update `pinescript_converter_agent.py` line 195:
   ```python
   data_map = {
       "2m": "BTC-USD-2m.csv",
       "5m": "YOUR-CUSTOM-DATA.csv",  # Change here
       # ...
   }
   ```

### Multiple Strategies

Convert and run multiple strategies:

```python
# Create multiple .pine files
# strategy1.pine
# strategy2.pine
# strategy3.pine

# Modify pinescript_converter_agent.py main() function:
strategies = ["strategy1.pine", "strategy2.pine", "strategy3.pine"]
for strategy_file in strategies:
    # Convert each...
```

### Optimization

The backtesting.py library supports parameter optimization:

```python
# In generated backtest code, add:
stats = bt.optimize(
    fast=range(5, 20, 5),
    slow=range(20, 50, 10),
    maximize='Sharpe Ratio'
)
```

---

## Troubleshooting

### Error: "Module 'talib' not found"

Install TA-Lib:
```bash
pip install TA-Lib
```

Or use pandas_ta instead:
```bash
pip install pandas_ta
```

### Error: "BTC data file not found"

Check available data files:
```bash
ls src/data/rbi/BTC*.csv
```

If missing, download from your data provider or use the included sample data.

### Error: "Position size must be integer"

The converter automatically fixes this, but if you see this error:

Change:
```python
self.buy(size=3546.0993)  # ❌ Wrong
```

To:
```python
self.buy(size=int(round(3546.0993)))  # ✅ Correct
```

Or use fractional sizing:
```python
self.buy(size=0.95)  # 95% of equity ✅
```

### Backtest Shows Zero Trades

Your strategy conditions might be too strict:
1. Review entry/exit conditions
2. Check indicator calculations
3. Verify data timeframe matches strategy design
4. Add debug prints to see indicator values

---

## Examples

See `examples/` directory for complete examples:
- `examples/sma_crossover.pine` - Simple moving average crossover
- `examples/rsi_mean_reversion.pine` - RSI-based mean reversion
- `examples/bollinger_breakout.pine` - Bollinger Band breakout
- `examples/macd_trend.pine` - MACD trend following

---

## Next Steps

1. ✅ **Convert your Pinescript strategy**
2. ✅ **Run backtests on multiple timeframes**
3. ✅ **Review performance metrics**
4. ✅ **Paper trade for 1-2 weeks**
5. ✅ **Enable live trading with small position sizes**
6. ✅ **Monitor and adjust parameters**
7. ✅ **Scale up gradually**

---

## Support

- **Issues**: Report at https://github.com/moon-dev-ai/moon-dev-ai-agents/issues
- **Discord**: Join Moon Dev community
- **YouTube**: Watch tutorials at Moon Dev channel

Built with love by Moon Dev 🌙🚀
