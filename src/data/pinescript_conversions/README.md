# TradingView Pinescript → Python Conversion System

This directory contains **precise, zero-loss conversions** of TradingView Pinescript strategies to Python backtesting.py implementations.

## 🎯 Mission Statement

**"It is imperative that there is nothing lost in translation"**

Every conversion must preserve:
- Exact parameter defaults from the original Pinescript
- All indicator calculations (including custom libraries)
- Position sizing logic (R-based, Kelly, fixed, etc.)
- Stop management systems (trailing, ATR-based, band-based, etc.)
- Entry/exit signal generation
- Signal expiration rules

## 📁 Directory Structure

```
pinescript_conversions/
├── README.md                           # This file
├── ADX_SQUEEZE_ANALYSIS.md             # Deep-dive analysis of ADX + Squeeze strategy
├── backtests/                          # Converted Python strategies
│   └── ADX_Squeeze_R_Based_BT_v2.py   # ADX + Squeeze [R-BASED] strategy
├── strategies/                         # Original Pinescript files (for reference)
└── results/                            # Backtest output data (CSVs, charts)
```

## 🔄 Conversion Methodology

### 1. Deep Analysis Phase
- **Read the entire Pinescript code** - understand every line before converting
- **Document indicator logic** - especially custom libraries (loxx, etc.)
- **Map all parameters** - note default values vs modified versions
- **Understand stop systems** - often the most complex part
- **Trace signal flow** - entry triggers, filters, confirmations, exits

Example: See `ADX_SQUEEZE_ANALYSIS.md` for 11-section deep-dive

### 2. Parameter Matching Phase
- **Use EXACT defaults** from original Pinescript (not modified versions)
- **Verify line numbers** - reference specific lines (e.g., "Line 24: tradingDirection = 'Long Only'")
- **Cross-check settings** - if multiple versions exist, use the one user specifies

### 3. Indicator Implementation Phase
- **Use pandas_ta or talib** - NOT backtesting.py built-in indicators
- **Handle custom sources** - loxx library transforms (HAB, Heikin Ashi, etc.)
- **Preserve calculation order** - indicators may depend on each other
- **Validate calculations** - compare intermediate values to Pinescript if possible

### 4. Strategy Logic Phase
- **Entry signals** - implement exact conditions (AND/OR logic, thresholds)
- **Signal expiration** - if enabled, track bars since signal generated
- **Position sizing** - R-based, Kelly, fixed, or custom methods
- **Stop orders** - Pinescript uses "stop entry" orders, Python approximates with conditionals

### 5. Stop Management Phase
- **Initial stops** - ATR-based, fixed %, or swing-based
- **Trailing stops** - linear, exponential, or custom logic
- **Multiple stop types** - implement all stop systems (e.g., 4 stops: Initial, Dynamic R, D-Bands, ATR Regime)
- **Stop selection logic** - tightest valid stop that doesn't loosen

### 6. Validation Phase
- **Compare results** - run on identical time periods if possible
- **Check trade counts** - should be similar (not exact due to execution differences)
- **Verify metrics** - win rate, max drawdown, profit factor should align
- **Document differences** - explain any discrepancies (e.g., data availability)

## 📊 Current Conversions

### ADX + Squeeze [R-BASED]
**File**: `backtests/ADX_Squeeze_R_Based_BT_v2.py`

**Strategy Overview**:
- **Squeeze Indicator**: EMA(5) vs EMA(7) with ATR(50)*0.4 threshold
- **ADX Confluence**: ADX > 20, +DI vs -DI direction, EMA(12) vs EMA(50) trend
- **R-Based Sizing**: `qty = (equity * 0.5%) / stopDistance`
- **4 Stop Types**:
  1. Initial ATR-based (with regime multipliers)
  2. Dynamic R Trailing (exponential tightening)
  3. D-Bands (double WMA with asymmetric bands)
  4. ATR Regime (LOW/NORMAL/HIGH volatility states)
- **Stop Selection**: Tightest valid stop that doesn't loosen
- **Signal Expiration**: 13 bars for ADX signals, instant for squeeze

**Performance**:
- 2023 15m data: **+6.83%** return, -6.29% max DD, 87 trades, 40.23% win rate
- 2023-2025 1h data: **-1.55%** return, -5.95% max DD, 45 trades, 35.56% win rate
- Strategy requires strong trends to perform (TradingView showed +172% in trending 2025 market)

**Key Implementation Details**:
- Loxx library with "Close" source simplifies to standard EMA
- Stop entry orders approximated with conditional market orders
- Regime detection uses hysteresis to avoid flickering
- All 4 stops tracked and compared every bar

## 🛠 Tools & Utilities

### Data Downloaders

**`download_btc_15m_2025.py`** (root directory)
- Downloads last 60 days of BTC-USD 15-minute bars (yfinance limit)
- Output: `src/data/rbi/BTC-USD-15m-2025.csv`
- ~5,700 bars covering Aug-Oct 2025

**`download_btc_1h_2020_2025.py`** (root directory)
- Downloads 2 years of BTC-USD hourly bars
- Output: `src/data/rbi/BTC-USD-1h-2020-2025.csv`
- ~17,500 bars covering Oct 2023 - Oct 2025

### Backtest Data Extraction

**`extract_backtest_data.py`** (root directory)
- Extracts daily NAV and complete trades list from backtests
- **Outputs**:
  - `backtest_daily_nav.csv` - Date, NAV, daily return %
  - `backtest_trades.csv` - Entry/exit times, prices, PnL, all indicators
- Includes all strategy indicators at entry/exit for post-analysis

**Usage**:
```bash
python extract_backtest_data.py
```

**Sample NAV Output**:
```csv
Date,NAV,Daily_Return_Pct
2023-01-01,100000.0,
2023-01-02,100000.0,0.0
2023-01-06,100217.67,0.218
2023-11-20,106833.42,0.0
```

**Sample Trades Output**:
```csv
EntryTime,ExitTime,Size,EntryPrice,ExitPrice,ReturnPct,PnL
2023-01-06 16:00:00,2023-01-06 19:45:00,4,16832.67,16920.84,0.323,217.67
2023-01-08 23:15:00,2023-01-09 14:00:00,5,17032.68,17224.35,0.924,787.06
...
```

## 📈 Running Converted Strategies

### Quick Start
```bash
# Run backtest with interactive chart
python src/data/pinescript_conversions/backtests/ADX_Squeeze_R_Based_BT_v2.py

# Extract results for analysis
python extract_backtest_data.py
```

### Custom Data Period
Edit the strategy file:
```python
# Change data source
data_path = "src/data/rbi/BTC-USD-15m-2025.csv"  # Recent 60 days
# or
data_path = "src/data/rbi/BTC-USD-1h-2020-2025.csv"  # 2-year hourly
```

### Modify Strategy Parameters
All parameters are class attributes - easy to modify:
```python
class ADX_Squeeze_R_Based(Strategy):
    # ADX Settings
    adx_threshold = 20  # Change threshold

    # Entry Settings
    tradingDirection = "Long Only"  # "Both", "Short Only"
    signalExpirationBars = 13

    # Position Sizing
    risk_per_trade = 0.005  # 0.5% risk per trade
```

## 🔍 Common Pitfalls

### 1. Using Wrong Parameter Defaults
**Problem**: Multiple Pinescript files exist with different settings
**Solution**: Ask user which file is the source of truth, reference specific line numbers

### 2. Ignoring Custom Libraries
**Problem**: Assuming standard indicators when custom libraries are used
**Solution**: Research the library (e.g., loxx source transformations), understand what it does

### 3. Oversimplifying Stop Logic
**Problem**: Pinescript may have 3-4 different stop systems running simultaneously
**Solution**: Implement ALL stop types, compare them, select tightest valid

### 4. Stop Entry Orders
**Problem**: Python backtesting.py doesn't support stop entry orders
**Solution**: Approximate with conditional market orders (check if price crossed threshold)

### 5. Signal Expiration
**Problem**: Forgetting to implement signal expiration (signals remain active forever)
**Solution**: Track bars since signal, clear after N bars if not executed

### 6. Data Period Mismatches
**Problem**: Comparing results from different time periods
**Solution**: Document data periods, explain why results differ if periods don't align

## 📝 Adding New Conversions

1. **Create analysis document**: `[STRATEGY_NAME]_ANALYSIS.md`
   - 10+ section deep-dive of strategy logic
   - Parameter table with defaults
   - Indicator calculations
   - Entry/exit conditions
   - Stop management
   - Potential conversion challenges

2. **Implement Python backtest**: `backtests/[STRATEGY_NAME]_BT_v2.py`
   - Use backtesting.py framework
   - Inherit from `Strategy` class
   - Implement `init()` and `next()` methods
   - Add detailed docstring with conversion notes

3. **Test and validate**:
   - Run on multiple time periods
   - Compare to Pinescript results (if data available)
   - Document performance differences
   - Extract trades and NAV for analysis

4. **Update this README**:
   - Add to Current Conversions section
   - Note any unique challenges
   - Document performance on standard datasets

## 🎓 Learning Resources

- **backtesting.py docs**: https://kernc.github.io/backtesting.py/
- **pandas_ta docs**: https://github.com/twopirllc/pandas-ta
- **TA-Lib docs**: https://mrjbq7.github.io/ta-lib/
- **Pinescript reference**: https://www.tradingview.com/pine-script-reference/

## ⚠️ Important Notes

1. **Execution Differences**: Pinescript uses stop orders, Python uses market orders - trade execution will differ slightly
2. **Commission Models**: Adjust commission settings to match your broker (default: 0.1%)
3. **Slippage**: Not modeled by default - add if needed for live trading simulation
4. **Fractional Trading**: Bitcoin prices exceed $100k - use fractional backtest class or increase capital
5. **Data Limitations**: yfinance has limited historical data for short timeframes (15m = 60 days max)

## 🤝 Contributing

When adding conversions:
- Follow the 6-phase methodology above
- Create comprehensive analysis documents
- Verify results before committing
- Document all assumptions and limitations
- Include performance metrics on standard datasets

---

Built by Moon Dev for the AI Agents for Trading project
