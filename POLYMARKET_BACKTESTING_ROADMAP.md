# Polymarket Backtesting Roadmap

## 🎯 Current Status
- ✅ **Live Trading System**: 95% functional
- ✅ **5-Phase Pipeline**: SENSE → THINK → DECIDE → TRADE → EXIT working
- ✅ **Entry Gates**: 6 gates (ALL must pass) - validated
- ✅ **Swarm Forecaster**: Claude + DeepSeek generating forecasts
- ✅ **Quant Layer**: EV, z-score, Kelly sizing working
- ❌ **Backtesting**: 0% - no historical data yet

## 📋 Critical Path to Backtesting

### **Phase 1: Fix Remaining Bugs** ⏱ 10 minutes
**Status**: In Progress

**Issue**: LLM Forecaster using wrong Claude model
- Current: `claude-3-haiku` (doesn't exist - 404 error)
- Should use: `claude-sonnet-4-5` or disable LLM adjustment

**Fix Options**:
1. **Option A**: Disable LLM forecaster (fastest)
   ```python
   # In src/config.py
   POLYMARKET_USE_LLM_ADJUSTMENT = False
   ```

2. **Option B**: Use DeepSeek for LLM forecaster (cheapest)
   ```python
   # In src/config.py
   POLYMARKET_LLM_BASE_MODEL = 'deepseek'  # Was 'claude'
   ```

3. **Option C**: Use correct Claude model
   - Swarm forecaster already requests `claude-sonnet-4-5`
   - Need to ensure LLM forecaster does the same

---

### **Phase 2: Start Data Collection** ⏱ 24-48 hours minimum
**Status**: Not Started

**What to collect**:
1. **Market snapshots** (every 60 seconds)
   - Price, volume, liquidity, spread
   - Current odds (YES/NO)
   - Days to resolution

2. **Order book data** (optional - API often fails)
   - Bid/ask prices
   - Depth

3. **RSS feed events** (news monitoring)
   - Reuters, BBC, Politico, NYT
   - FinBERT sentiment scores

4. **Whale wallet tracking**
   - Top trader bets (>$10k positions)
   - Win rates, bet sizes

**Run Commands**:
```bash
# Start data collector (runs indefinitely)
python src/agents/polymarket_data_collector.py

# Monitor progress
python check_progress.py
```

**Expected Data**:
- **Minimum**: 1,440 snapshots (24 hours @ 60s intervals)
- **Optimal**: 10,080 snapshots (1 week)
- **Storage**: ~50MB per day

**Output Files**:
```
src/data/polymarket/training_data/
├── market_snapshots_YYYYMMDD_HHMMSS.csv
├── whale_bets_YYYYMMDD.csv
├── events_YYYYMMDD.csv
└── resolutions_YYYYMMDD.csv
```

---

### **Phase 3: Train Meta-Learner** ⏱ 5-10 minutes
**Status**: Waiting on Phase 2

**What it does**:
- Learns which agents (Whale, Event, Anomaly) produce best forecasts
- Trains Ridge Regression for probability adjustment weights
- Trains Logistic Regression for trade acceptance
- Outputs calibrated weights to `calibration.json`

**Requirements**:
- Minimum 1,440 market snapshots (24 hours)
- At least 20 resolved markets for calibration
- Whale bets and event data for feature engineering

**Run Commands**:
```bash
# Train meta-learner (after data collection)
python src/agents/polymarket_meta_learner.py

# Check output
cat src/data/polymarket/meta_learning/calibration.json
```

**Expected Output** (`calibration.json`):
```json
{
  "probability_adjustment": {
    "model": "ridge",
    "whale_weight": 0.35,
    "event_weight": 0.28,
    "anomaly_weight": 0.18,
    "baseline_weight": 0.19
  },
  "trade_acceptance": {
    "model": "logistic",
    "coefficients": [1.2, -0.8, 0.5],
    "threshold": 0.65
  },
  "metadata": {
    "trained_on": 1440,
    "win_rate": 0.58,
    "avg_edge": 0.032
  }
}
```

---

### **Phase 4: Create Backtesting Framework** ⏱ 2-3 hours
**Status**: Not Started

**What to build**:
1. **Historical Market Replayer**
   - Loads market snapshots from CSV
   - Replays each snapshot through orchestrator
   - Simulates limit-at-fair execution (95% fill rate)

2. **Paper Trading Engine**
   - Tracks virtual positions
   - Calculates PnL based on market movements
   - Applies exit rules (6-rule system)

3. **Performance Analytics**
   - Win rate, average edge, Sharpe ratio
   - EV vs actual PnL correlation
   - Calibration curves (predicted vs actual)

**Implementation Plan**:
```python
# src/agents/polymarket_backtester.py (NEW FILE)

class PolymarketBacktester:
    def __init__(self, start_date, end_date, initial_capital=10000):
        self.orchestrator = PolymarketOrchestrator(portfolio_value=initial_capital)
        self.snapshots = self.load_historical_snapshots(start_date, end_date)
        self.positions = []
        self.trades = []

    def load_historical_snapshots(self, start, end):
        """Load all market snapshots in date range"""
        # Read from src/data/polymarket/training_data/market_snapshots_*.csv
        pass

    def simulate_execution(self, decision, current_price):
        """Simulate limit-at-fair order (95% fill rate)"""
        if decision['entry_decision'] != 'ENTER':
            return None

        # 95% chance of fill at fair price (true_prob)
        if random.random() < 0.95:
            return {
                'filled': True,
                'fill_price': decision['true_prob'],
                'size': decision['final_position_size']
            }
        return {'filled': False}

    def run_backtest(self):
        """Run full backtest"""
        for snapshot in self.snapshots:
            # Analyze market at this snapshot
            result = self.orchestrator.analyze_market(
                market_id=snapshot['market_id'],
                question=snapshot['question'],
                current_yes_price=snapshot['yes_price'],
                # ... other snapshot data
            )

            # Simulate execution
            if result['decision']['entry_decision'] == 'ENTER':
                execution = self.simulate_execution(
                    result['decision'],
                    snapshot['yes_price']
                )
                if execution['filled']:
                    self.positions.append({
                        'entry_time': snapshot['timestamp'],
                        'entry_price': execution['fill_price'],
                        'size': execution['size'],
                        'market_id': snapshot['market_id']
                    })

            # Check exit rules for open positions
            for position in self.positions:
                exit_check = self.orchestrator.exit_manager.check_position(
                    position_id=position['position_id'],
                    current_price=snapshot['yes_price'],
                    # ... exit rule checks
                )
                if exit_check['should_exit']:
                    self.close_position(position, snapshot)

        return self.calculate_performance()

    def calculate_performance(self):
        """Calculate backtest statistics"""
        return {
            'total_trades': len(self.trades),
            'win_rate': sum(1 for t in self.trades if t['pnl'] > 0) / len(self.trades),
            'total_pnl': sum(t['pnl'] for t in self.trades),
            'sharpe_ratio': self.calculate_sharpe(),
            'max_drawdown': self.calculate_max_drawdown(),
            'avg_hold_time': np.mean([t['hold_time'] for t in self.trades])
        }
```

**Run Commands**:
```bash
# Run backtest on collected data
python src/agents/polymarket_backtester.py --start 2025-01-01 --end 2025-01-30

# Generate backtest report
python src/agents/polymarket_backtester.py --report
```

---

### **Phase 5: Analyze & Optimize** ⏱ Ongoing
**Status**: Future

**What to analyze**:
1. **Calibration**
   - Do forecasts match reality? (calibration curves)
   - Are edge estimates accurate?

2. **Entry Gate Tuning**
   - Are thresholds too strict/loose?
   - Which gates filter most opportunities?

3. **Exit Rule Performance**
   - Which rules trigger most often?
   - Average PnL per exit rule

4. **Model Performance**
   - Which LLMs produce best forecasts?
   - Does meta-learning improve accuracy?

---

## 🚀 Quick Start (Right Now)

### **Step 1: Fix LLM Forecaster** (Recommended: Disable it)
```bash
# Edit src/config.py, change line:
POLYMARKET_USE_LLM_ADJUSTMENT = False  # Was True
```

### **Step 2: Test Orchestrator**
```bash
# Verify it runs without errors
python src/agents/polymarket_orchestrator.py
```

### **Step 3: Start Data Collection** (24-48 hours)
```bash
# Run data collector (keep it running!)
python src/agents/polymarket_data_collector.py

# In another terminal, monitor progress
python check_progress.py
```

### **Step 4: Wait 24-48 Hours** ⏳
- Let data collector run
- Need minimum 1,440 snapshots for training
- Check progress every few hours

### **Step 5: Train Meta-Learner**
```bash
# After 24-48 hours of data collection
python src/agents/polymarket_meta_learner.py
```

### **Step 6: Build Backtester** (2-3 hours of coding)
- Create `src/agents/polymarket_backtester.py`
- Implement historical replay logic
- Add performance analytics

---

## 📊 Expected Timeline

| Phase | Duration | Blocking | Can Start |
|-------|----------|----------|-----------|
| 1. Bug Fixes | 10 min | Yes | NOW |
| 2. Data Collection | 24-48 hours | Yes | After Phase 1 |
| 3. Meta-Learning | 5-10 min | Yes | After Phase 2 |
| 4. Build Backtester | 2-3 hours | No | After Phase 2 |
| 5. Run Backtest | 10-30 min | No | After Phases 3 & 4 |
| **Total Time** | **26-51 hours** | - | - |

**Realistic estimate**: **2-3 days** (mostly waiting for data collection)

---

## ⚠️ Important Notes

### **Why Can't We Backtest Now?**
- No historical market data (need snapshots over time)
- No trained meta-learner weights (need resolved markets)
- System designed for **probability arbitrage**, not outcome prediction
- Profit comes from **market convergence**, not binary outcomes

### **What Makes This Different from Crypto Backtesting?**
- **No OHLCV data**: Polymarket doesn't have bar data like stocks/crypto
- **Event-driven**: Markets react to news, not technical patterns
- **Short timeframes**: Most markets resolve in days/weeks, not months
- **Binary outcomes**: YES/NO, not continuous price movement
- **Edge decay**: Probability arbitrage opportunities close fast

### **Current Workarounds** (if you want to test NOW):
1. **Use current markets**: Run orchestrator on live markets (paper trading)
2. **Manual data**: Fetch 10-20 markets, analyze decisions
3. **Synthetic backtest**: Create fake historical data for testing code
4. **Forward test**: Run for 1 week, track all decisions

---

## 🎯 Success Criteria

**Minimum Viable Backtest**:
- ✅ 1,440+ market snapshots (24 hours)
- ✅ 10+ resolved markets for calibration
- ✅ Meta-learner trained (calibration.json exists)
- ✅ Backtester can replay historical snapshots
- ✅ Performance report generated

**Production-Ready Backtest**:
- ✅ 10,080+ market snapshots (1 week)
- ✅ 50+ resolved markets across categories
- ✅ Whale tracking data (top wallet bets)
- ✅ Event catalyst data (news + sentiment)
- ✅ Calibration curves match reality
- ✅ Sharpe ratio > 1.5, Win rate > 55%

---

## 📞 Next Steps (Your Decision)

**Option A: Start Data Collection Now** (Recommended)
```bash
# Disable LLM forecaster to remove bug
nano src/config.py  # Set POLYMARKET_USE_LLM_ADJUSTMENT = False

# Start data collector (let it run 24-48 hours)
python src/agents/polymarket_data_collector.py
```

**Option B: Build Backtester First** (Coding)
- Create backtesting framework (2-3 hours)
- Use synthetic/dummy data for testing
- Wait for real data to validate

**Option C: Forward Test** (Hybrid)
- Run orchestrator in paper trading mode
- Track all decisions for 7 days
- Analyze results manually

**My Recommendation**: **Option A** (Start data collection immediately, build backtester while waiting)
