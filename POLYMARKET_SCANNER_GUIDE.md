# POLYMARKET LIGHTWEIGHT SCANNER - QUICK START

## What It Does

The lightweight scanner is a **fast, efficient way to monitor Polymarket for trading opportunities** without the overhead of running the full orchestrator continuously.

**Key Features**:
- ⚡ **Fast scans** (<30 seconds vs 2-3 minutes for full orchestrator)
- 💰 **Low cost** (~$0.01-0.05 per scan vs $0.10-0.20)
- 🎯 **Smart filtering** (spread, liquidity, volume checks)
- 🤖 **Auto-trigger** (only runs full analysis when strong signals detected)
- 📊 **Uses your collected data** (checks historical volatility)

## How It Works

### 4-Phase Quick Scan:
1. **Fetch Markets** - Gets top 50 markets by 24h volume
2. **Quick Filter** - Checks spread, liquidity, volume thresholds
3. **Score Opportunities** - Rates each market 0-100 based on signals
4. **Trigger Decision** - Runs full orchestrator if score ≥ 70 or 3+ markets ≥ 50

### Scoring System (0-100):
- **Spread**: Tight spread (≤3%) = +30 points, Medium (≤6%) = +15
- **Liquidity**: High (≥$50k) = +20, Medium (≥$10k) = +10
- **Volume**: Strong (≥$100k/day) = +20, Medium (≥$10k) = +10
- **Pricing**: Extreme price (<15% or >85%) = +15
- **Historical Data**: Has history = +15, High volatility = +10

### Trigger Rules:
- **Single strong signal**: Any market scoring ≥ 70/100
- **Multiple signals**: 3+ markets scoring ≥ 50/100
- When triggered → Runs full orchestrator with all AI models

---

## Usage

### Option 1: Run Single Scan (Test)
```bash
python src/agents/polymarket_scanner.py --once
```
**Use case**: Test the scanner, see current opportunities

### Option 2: Continuous Scanning (Recommended)
```bash
# Scan every 10 minutes (default)
python src/agents/polymarket_scanner.py

# Or customize interval
python src/agents/polymarket_scanner.py --interval 15  # Every 15 minutes
python src/agents/polymarket_scanner.py --interval 5   # Every 5 minutes
```
**Use case**: Run 24/7 to catch opportunities as they appear

### Option 3: Windows Task Scheduler (Set and Forget)
1. Open Windows Task Scheduler
2. Create Basic Task: "Polymarket Scanner"
3. Trigger: Every 10 minutes
4. Action: Start a program
   - Program: `C:\path\to\python.exe`
   - Arguments: `src/agents/polymarket_scanner.py --once`
   - Start in: `C:\Users\oo\Desktop\moon-dev-ai-agents`

**Use case**: Completely automated, restarts after reboot

---

## Example Output

```
================================================================================
[SCAN #1] 2025-10-28 13:44:12
================================================================================
[1/4] Fetching active markets...
[OK] Retrieved 50 markets
[2/4] Applying quick filters...
[OK] 12 markets passed filters
[3/4] Scoring opportunities...
[OK] 3 opportunities scored >= 40
[4/4] Results:

[FOUND] 3 STRONG OPPORTUNITIES:

#1 [Score: 75/100]
  Market: Will Bitcoin hit $100k by end of 2025?
  Yes Price: 68.50% | No Price: 31.50%
  Spread: 2.30% | Liquidity: $87,450
  Volume 24h: $152,000
  Reasons: Tight spread (2.30%), High liquidity ($87,450), Strong volume ($152,000)
  Historical Data: 156 snapshots

#2 [Score: 62/100]
  Market: Will Trump announce 2024 campaign by March?
  Yes Price: 12.80% | No Price: 87.20%
  Spread: 4.50% | Liquidity: $45,200
  Volume 24h: $68,900
  Reasons: Extreme pricing (12.80%), High volatility (8.20%)
  Historical Data: 203 snapshots

[TRIGGER] Top opportunity score 75 >= 70
[ACTION] Starting full orchestrator analysis...
```

---

## Configuration

Edit `src/config.py` to customize behavior:

```python
# Scan interval (minutes)
POLYMARKET_SCAN_INTERVAL_MINUTES = 10  # Default: 10 minutes

# Filter thresholds
POLYMARKET_MAX_SPREAD = 0.06  # Max 6% spread
POLYMARKET_MIN_LIQUIDITY = 10000  # Min $10k liquidity
POLYMARKET_MIN_VOLUME_24H = 1000  # Min $1k daily volume
```

---

## Why This Approach?

### Problems with 24/7 Orchestrator:
- ❌ Takes 2-3 minutes to initialize (loads 4 AI models)
- ❌ Heavy memory usage
- ❌ Expensive API costs even when no opportunities exist
- ❌ Can crash and requires restart

### Benefits of Lightweight Scanner:
- ✅ Fast (<30 seconds per scan)
- ✅ Minimal API costs
- ✅ Runs frequently (every 5-10 minutes)
- ✅ Only pays for full analysis when needed
- ✅ Auto-restarts easily

---

## Current System Status

### ✅ Data Collection (Running)
```bash
# Check status
python polymarket_status.py
```
- Collecting market snapshots every 60 seconds
- Building training dataset for meta-learner
- Running in background (bash ID: 6d03e4)

### ✅ Meta-Learning (Trained)
- Calibration file: `src/data/polymarket/meta_learning/calibration.json`
- Trained on 22+ hours of data
- Will improve with weekly retraining

### ✅ Scanner (Ready to Run)
```bash
# Start scanning now
python src/agents/polymarket_scanner.py
```

---

## Next Steps

### Immediate (Today):
1. ✅ **Data collection running** (24/7 background)
2. ✅ **Meta-learner trained** (initial calibration)
3. ✅ **Scanner created** (ready to use)
4. 🔄 **Start scanner** → `python src/agents/polymarket_scanner.py`

### This Week:
- Monitor scanner output for opportunities
- Let data collector accumulate more data
- Review any paper trades triggered by scanner

### Weekly:
- Retrain meta-learner: `python src/agents/polymarket_meta_learner.py`
- Review performance metrics
- Adjust thresholds if needed

---

## Troubleshooting

**"No opportunities found in this scan"**
- Normal! Markets don't always have arbitrage opportunities
- Scanner runs frequently to catch them when they appear
- Consider lowering thresholds in `config.py` if too strict

**"API returned 500"**
- Polymarket API temporary issue
- Scanner will retry on next cycle
- No action needed

**Scanner exits immediately**
- Check Python path in command
- Verify working directory is correct
- Run with `--once` flag first to test

---

## Cost Estimates

### Lightweight Scanner:
- **Per scan**: ~$0.01-0.05 (just API calls, no AI)
- **Per hour** (6 scans): ~$0.06-0.30
- **Per day** (144 scans): ~$1.44-7.20

### Full Orchestrator (when triggered):
- **Per analysis**: ~$0.10-0.20 (4 AI models)
- **Triggers**: Only when score ≥ 70 or 3+ signals ≥ 50
- **Expected**: 1-3 times per day

### Total Expected Cost:
- **Light activity**: $2-10/day
- **High activity**: $10-20/day
- **Compare to**: 24/7 orchestrator = $50-100/day

---

**READY TO START?**

```bash
# Start scanning for opportunities every 10 minutes
python src/agents/polymarket_scanner.py
```

Press Ctrl+C to stop anytime.
