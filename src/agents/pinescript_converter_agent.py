"""
🌙 Moon Dev's TradingView Pinescript Converter Agent 🌙

Converts TradingView Pinescript strategies to Python backtesting code
and enables live trading via the Trading Agent.

WORKFLOW:
1. Input: TradingView Pinescript code
2. Convert: Pinescript → Python (backtesting.py compatible)
3. Backtest: Run on BTC historical data (2m, 5m, 10m, 30m)
4. Paper Trade: Test strategy in paper trading mode
5. Live Trade: Enable live trading once validated

Built with love by Moon Dev 🚀
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from termcolor import cprint
from src.models.model_factory import ModelFactory

# ============================================================================
# Initialize Model Factory
# ============================================================================
model_factory = ModelFactory()

# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

# Model configuration for conversion
CONVERTER_MODEL_CONFIG = {
    "type": "claude",  # Using Claude (Anthropic) for conversion
    "name": "claude-sonnet-4-5"  # Sonnet 4.5 (NOT Opus - cost savings)
}

# Output directories
BASE_DIR = Path("src/data/pinescript_conversions")
BACKTESTS_DIR = BASE_DIR / "backtests"
STRATEGIES_DIR = BASE_DIR / "strategies"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
for dir_path in [BASE_DIR, BACKTESTS_DIR, STRATEGIES_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 🎯 CONVERSION PROMPTS
# ============================================================================

PINESCRIPT_TO_PYTHON_PROMPT = """
You are Moon Dev's Advanced Pinescript Conversion AI 🌙

Your task: Convert TradingView Pinescript strategies to Python backtesting.py code.
Special focus: R-BASED POSITION SIZING and sophisticated stop loss management.

PINESCRIPT CODE TO CONVERT:
{pinescript_code}

CRITICAL CONVERSION RULES:

===== 1. R-BASED POSITION SIZING =====

IF you detect ANY of these patterns in the Pinescript:
- Variables named "risk_percent" or "risk_per_trade"
- Calculations like: equity * risk% / stopDistance
- Functions named "calculatePositionSize" or similar
- Comments mentioning "R-based" or "risk-based" sizing

THEN you MUST convert as follows:

```python
class ConvertedStrategy(Strategy):
    # R-based position sizing parameters
    risk_percent = 0.5  # Extract from Pinescript input

    def init(self):
        # Track state for R-based sizing
        self.entry_price = None
        self.initial_stop = None
        self.initial_risk = None

    def next(self):
        # Calculate stop distance BEFORE entry
        stop_distance = <calculate from ATR, bands, or other indicator>

        # R-BASED POSITION SIZE CALCULATION
        current_equity = self.equity
        risk_amount = current_equity * (self.risk_percent / 100.0)
        position_qty = risk_amount / stop_distance
        position_qty = int(round(position_qty))  # MUST be integer

        # Entry with calculated size
        if <entry_condition> and not self.position:
            self.entry_price = self.data.Close[-1]
            self.initial_stop = self.entry_price - stop_distance  # for long
            self.initial_risk = stop_distance

            self.buy(size=position_qty)
            print(f"🚀 LONG @ {{self.entry_price:.2f}} | Qty: {{position_qty}} | Risk: {{self.risk_percent}}% (${{risk_amount:.2f}}) | Stop: ${{self.initial_stop:.2f}}")
```

KEY POINTS:
- Position size = (equity * risk%) / stop_distance
- ALWAYS convert to int(round(...)) for backtesting.py
- Calculate stop distance BEFORE calculating position size
- Preserve separate long/short stop multipliers if present
- Include debug prints showing: entry price, quantity, risk $, stop price

===== 2. SOPHISTICATED STOP LOSS CONVERSION =====

IF you detect MULTIPLE stop types (initial, trailing, ATR-based, band-based):

```python
def next(self):
    # STOP TYPE 1: Initial Stop (set at entry)
    if <just_entered>:
        stop_dist = <ATR or fixed distance>
        self.initial_stop = entry_price - stop_dist
        self.current_stop = self.initial_stop

    # STOP TYPE 2: Dynamic R Trailing (parabolic curve)
    if self.position:
        profit_r = (close - entry) / initial_risk
        if profit_r > activation_threshold:
            desired_risk_factor = max_r - (reduction_formula)
            dynamic_stop = close - (desired_risk_factor * initial_risk)

    # STOP TYPE 3: Band-based (D-Bands, Bollinger, etc.)
    if self.position:
        band_stop = self.lower_band[-1]  # or upper for shorts

    # STOP TYPE 4: ATR Regime-based
    if self.position:
        atr_mult = self.volatility_regime_multiplier
        atr_stop = close - (atr_value * atr_mult)

    # SELECT BEST STOP (tightest valid stop that doesn't loosen)
    if self.position.is_long:
        candidate_stops = [dynamic_stop, band_stop, atr_stop]
        candidate_stops = [s for s in candidate_stops if s >= self.initial_stop]
        if candidate_stops:
            best_stop = max(candidate_stops)  # Tightest (highest for long)
            if best_stop > self.current_stop:
                self.current_stop = best_stop
                print(f"📈 Stop tightened to ${{self.current_stop:.2f}}")
```

KEY POINTS:
- Preserve ALL stop types from Pinescript
- Implement stop selection logic (choose tightest valid stop)
- Never loosen stops (only tighten)
- Track which stop type is active
- Separate initial stop from trailing stops

===== 3. ADVANCED ENTRY LOGIC =====

DETECT and convert these patterns:

A) STOP ENTRY ORDERS (Pinescript: strategy.entry with stop= parameter):
```python
# Convert to: wait for price to cross threshold, then market order
if <signal_triggered>:
    self.pending_long_price = high + offset_ticks
    self.pending_long_signal_bar = len(self.data)

if self.pending_long_price and close >= self.pending_long_price:
    self.buy(size=calculated_qty)
    self.pending_long_price = None
```

B) SIGNAL EXPIRATION:
```python
if self.pending_long_signal_bar:
    bars_elapsed = len(self.data) - self.pending_long_signal_bar
    if bars_elapsed >= expiration_bars:
        self.pending_long_price = None
        print("⏱️ Long signal expired")
```

C) ADD-ONS (Pyramiding):
```python
if <add_on_signal> and self.position.is_long:
    add_on_qty = int(round(risk_amount / stop_distance))
    self.buy(size=add_on_qty)
    print(f"➕ Added {{add_on_qty}} to long position")
```

D) POSITION REVERSALS:
```python
if <reverse_signal> and self.position.is_long:
    self.position.close()
    # Calculate opposite position
    short_qty = int(round(risk_amount / stop_distance))
    self.sell(size=short_qty)
    print(f"🔄 Reversed: Long → Short")
```

===== 4. INDICATOR MAPPINGS =====

Standard indicators (Pinescript → Python):
- ta.sma(close, 20) → self.I(talib.SMA, self.data.Close, timeperiod=20)
- ta.ema(close, 20) → self.I(talib.EMA, self.data.Close, timeperiod=20)
- ta.rsi(close, 14) → self.I(talib.RSI, self.data.Close, timeperiod=14)
- ta.atr(14) → self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
- ta.wma(close, 20) → self.I(talib.WMA, self.data.Close, timeperiod=20)
- ta.dmi(len, smooth) → use talib.PLUS_DI, talib.MINUS_DI, talib.ADX
- ta.bbands(close, len, mult) → self.I(talib.BBANDS, self.data.Close, timeperiod=len, nbdevup=mult, nbdevdn=mult)
- ta.crossover(a, b) → (a[-2] < b[-2] and a[-1] > b[-1])
- ta.crossunder(a, b) → (a[-2] > b[-2] and a[-1] < b[-1])

Advanced patterns:
- Double WMA → ta.wma(ta.wma(src, len), len)
- Request.security → Use same timeframe (backtesting.py limitation)
- Custom functions → Implement inline with numpy/pandas

===== 5. LIBRARY & IMPORTS =====

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np
```

===== 6. DATA LOADING & BACKTEST EXECUTION =====

```python
if __name__ == "__main__":
    # Load BTC data
    df = pd.read_csv("BTC_DATA_PATH")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Run backtest
    bt = Backtest(df, ConvertedStrategy,
                  cash=10000,      # Match Pinescript initial capital
                  commission=0.001,  # 0.1% commission
                  exclusive_orders=True)

    stats = bt.run()
    print(stats)
    print(stats._strategy)

    # CRITICAL: Enable interactive chart
    bt.plot()  # Opens browser with interactive Bokeh chart
```

===== 7. MOON DEV DEBUG STYLE =====

Add comprehensive debug prints:
```python
# Entry
print(f"🚀 LONG @ ${{entry_price:.2f}} | Qty: {{qty}} | Risk: {{risk_percent}}% (${{risk_amount:.2f}})")
print(f"   Stop: ${{initial_stop:.2f}} ({{stop_pct:.2f}}%) | Target: ${{target:.2f}}")

# Exit
print(f"💰 EXIT @ ${{exit_price:.2f}} | P/L: ${{pnl:.2f}} ({{pnl_pct:.1f}}%) | R: {{r_multiple:.2f}}R")

# Stop updates
print(f"📈 Stop tightened: ${{old_stop:.2f}} → ${{new_stop:.2f}} ({{stop_type}})")

# Errors
print(f"❌ Position sizing error: {{error_message}}")
```

===== 8. EXTERNAL LIBRARY HANDLING =====

IF Pinescript imports external libraries (loxx, etc.):
- For standard indicators: Use talib equivalent
- For custom MAs: Implement simplified version with numpy
- For unavailable functions: Add comment: "# TODO: Implement <function_name>"
- Document any missing functionality

===== 9. BACKTESTING.PY LIMITATIONS =====

CRITICAL conversions for backtesting.py compatibility:
- NO strategy.exit() with multiple conditions → Use if/else in next()
- NO stop entry orders → Convert to market orders when price crosses
- Position size MUST be int(round()) or fraction (0-1)
- Stop/Limit prices are ABSOLUTE prices, not distances
- Cannot access future data (no forward-looking logic)

===== FINAL CHECKLIST =====

Before outputting code, verify:
✅ R-based position sizing preserved (if present)
✅ All stop types converted and selection logic implemented
✅ Entry logic handles stop orders, add-ons, reversals
✅ All indicators wrapped in self.I()
✅ Position sizes are int(round(...))
✅ bt.plot() included for charts
✅ Debug prints use Moon Dev emoji style
✅ Comments explain complex conversions
✅ Code runs without external library dependencies (except talib)

CRITICAL: Output ONLY the complete Python code. No explanations, no markdown code blocks. Just raw Python code starting with imports.
"""

# ============================================================================
# 🚀 PINESCRIPT CONVERTER CLASS
# ============================================================================

class PinescriptConverter:
    def __init__(self):
        """Initialize the Pinescript Converter"""
        cprint("🌙 Initializing Moon Dev's Pinescript Converter Agent...", "cyan")

        # Initialize AI model
        self.model = model_factory.get_model(
            CONVERTER_MODEL_CONFIG["type"],
            CONVERTER_MODEL_CONFIG["name"]
        )
        cprint(f"✅ Using {CONVERTER_MODEL_CONFIG['name']} for conversion", "green")

    def convert_pinescript(self, pinescript_code, strategy_name="ConvertedStrategy"):
        """Convert Pinescript to Python backtesting code"""
        cprint(f"\n🔄 Converting Pinescript strategy: {strategy_name}...", "yellow")

        # Fill in the prompt
        prompt = PINESCRIPT_TO_PYTHON_PROMPT.format(
            pinescript_code=pinescript_code
        )

        # Generate Python code
        cprint("🤖 Asking AI to convert strategy (this may take 30-60 seconds for complex strategies)...", "cyan")
        python_code = self.model.generate_response(
            system_prompt="You are an expert at converting TradingView Pinescript to Python backtesting.py code, with special focus on R-based position sizing and sophisticated stop loss management.",
            user_content=prompt,
            temperature=0.2,  # Lower for more deterministic conversion
            max_tokens=8000  # Increased for complex strategies like ADX+Squeeze
        )

        # Clean up response (remove markdown if present)
        python_code = self._clean_code_response(python_code)

        # Save converted code
        timestamp = datetime.now().strftime("%m_%d_%Y_%H%M%S")
        output_file = BACKTESTS_DIR / f"{strategy_name}_{timestamp}.py"

        with open(output_file, 'w') as f:
            f.write(python_code)

        cprint(f"✅ Converted code saved to: {output_file}", "green")
        return str(output_file), python_code

    def _clean_code_response(self, response):
        """Remove markdown code blocks if present"""
        # Remove ```python and ``` markers
        cleaned = response.strip()
        if cleaned.startswith("```python"):
            cleaned = cleaned[9:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def backtest_strategy(self, code_file, timeframe="5m", symbol="BTC"):
        """Run backtest on the converted strategy"""
        cprint(f"\n📊 Running backtest for {symbol} on {timeframe} timeframe...", "yellow")

        # Determine data file path
        data_map = {
            "2m": "BTC-USD-2m.csv",
            "5m": "BTC-USD-5m.csv",
            "10m": "BTC-USD-10m.csv",
            "30m": "BTC-USD-30m.csv",
            "1H": "BTC-USD-1H.csv",
            "15m": "BTC-USD-15m.csv"
        }

        data_file = data_map.get(timeframe, "BTC-USD-15m.csv")
        data_path = Path(f"src/data/rbi/{data_file}")

        if not data_path.exists():
            cprint(f"❌ Data file not found: {data_path}", "red")
            cprint(f"   Available: {', '.join(data_map.values())}", "yellow")
            return None

        # Update code to use correct data path
        with open(code_file, 'r') as f:
            code = f.read()

        # Replace placeholder with actual path
        code = code.replace("BTC_DATA_PATH", str(data_path))

        # Save updated code
        with open(code_file, 'w') as f:
            f.write(code)

        # Execute the backtest
        cprint("🚀 Executing backtest...", "cyan")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, code_file],
                capture_output=True,
                text=True,
                timeout=120
            )

            cprint("\n" + "="*80, "cyan")
            cprint("📊 BACKTEST RESULTS", "cyan", attrs=['bold'])
            cprint("="*80, "cyan")
            print(result.stdout)

            if result.stderr:
                cprint("\n⚠️ Errors/Warnings:", "yellow")
                print(result.stderr)

            # Save results
            timestamp = datetime.now().strftime("%m_%d_%Y_%H%M%S")
            result_file = RESULTS_DIR / f"backtest_{symbol}_{timeframe}_{timestamp}.txt"
            with open(result_file, 'w') as f:
                f.write(f"Backtest Results - {symbol} {timeframe}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write("="*80 + "\n\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\nErrors/Warnings:\n")
                    f.write(result.stderr)

            cprint(f"\n✅ Results saved to: {result_file}", "green")
            return result.stdout

        except subprocess.TimeoutExpired:
            cprint("❌ Backtest timeout (>2 minutes)", "red")
            return None
        except Exception as e:
            cprint(f"❌ Error running backtest: {e}", "red")
            return None

    def prepare_for_live_trading(self, code_file, strategy_name):
        """Convert backtest code to live trading strategy format"""
        cprint(f"\n🔄 Preparing {strategy_name} for live trading...", "yellow")

        # Read backtest code
        with open(code_file, 'r') as f:
            backtest_code = f.read()

        # Extract strategy class
        # (This would need more sophisticated parsing in production)
        cprint("📝 Converting to strategy_agent compatible format...", "cyan")

        # Create strategy file for strategy_agent
        strategy_output = STRATEGIES_DIR / f"{strategy_name}_live.py"

        live_strategy_template = f"""
# Generated from Pinescript conversion
# Strategy: {strategy_name}

from src.config import *
import pandas as pd
import talib
import numpy as np

class {strategy_name}:
    name = "{strategy_name}"
    description = "Converted from TradingView Pinescript"

    def __init__(self):
        self.position = None

    def generate_signals(self, token_address, market_data):
        '''
        Analyze market data and generate trading signals.

        Args:
            token_address: Token to analyze
            market_data: OHLCV DataFrame

        Returns:
            dict with action, confidence, reasoning
        '''
        try:
            df = market_data.copy()
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            # Calculate indicators (from backtest)
            # TODO: Extract indicator calculations from backtest code

            # Generate signal
            action = "NOTHING"  # "BUY", "SELL", "NOTHING"
            confidence = 0
            reasoning = "Strategy logic from Pinescript"

            return {{
                "action": action,
                "confidence": confidence,
                "reasoning": reasoning
            }}

        except Exception as e:
            return {{
                "action": "NOTHING",
                "confidence": 0,
                "reasoning": f"Error: {{str(e)}}"
            }}

# Backtest code below (for reference)
'''
{backtest_code}
'''
"""

        with open(strategy_output, 'w') as f:
            f.write(live_strategy_template)

        cprint(f"✅ Live strategy saved to: {strategy_output}", "green")
        cprint("\n📌 Next steps:", "cyan")
        cprint("   1. Review the generated strategy code", "white")
        cprint("   2. Test in paper trading mode first", "white")
        cprint("   3. Add to trading_agent.py MONITORED_TOKENS", "white")
        cprint("   4. Enable live trading once validated", "white")

        return str(strategy_output)

# ============================================================================
# 🎯 MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow"""
    cprint("\n" + "="*80, "cyan", attrs=['bold'])
    cprint("🌙 MOON DEV'S PINESCRIPT CONVERTER AGENT 🌙", "cyan", attrs=['bold'])
    cprint("="*80 + "\n", "cyan", attrs=['bold'])

    # Example usage
    cprint("📋 USAGE:", "yellow", attrs=['bold'])
    cprint("   1. Create a .pine file with your TradingView strategy", "white")
    cprint("   2. Or paste Pinescript code directly in this script", "white")
    cprint("   3. Run conversion and backtest", "white")
    cprint("   4. Review results and prepare for live trading\n", "white")

    # Check if pinescript file exists
    pinescript_file = Path("strategy.pine")

    if pinescript_file.exists():
        cprint(f"✅ Found pinescript file: {pinescript_file}", "green")
        with open(pinescript_file, 'r', encoding='utf-8') as f:
            pinescript_code = f.read()

        # Initialize converter
        converter = PinescriptConverter()

        # Convert
        code_file, python_code = converter.convert_pinescript(
            pinescript_code,
            strategy_name="MyStrategy"
        )

        # Backtest on multiple timeframes
        for timeframe in ["5m", "15m", "30m", "1H"]:
            converter.backtest_strategy(code_file, timeframe=timeframe)

        # Prepare for live trading
        converter.prepare_for_live_trading(code_file, "MyStrategy")

    else:
        cprint(f"❌ No pinescript file found at: {pinescript_file}", "red")
        cprint("\n💡 To use this agent:", "yellow")
        cprint("   1. Create 'strategy.pine' in the root directory", "white")
        cprint("   2. Paste your TradingView Pinescript code", "white")
        cprint("   3. Run this script again\n", "white")

        # Show example
        cprint("📄 Example strategy.pine content:", "cyan")
        example = '''
//@version=5
strategy("Simple SMA Cross", overlay=true)

// Inputs
fastLength = input.int(10, "Fast SMA")
slowLength = input.int(30, "Slow SMA")

// Indicators
fastSMA = ta.sma(close, fastLength)
slowSMA = ta.sma(close, slowLength)

// Entry conditions
longCondition = ta.crossover(fastSMA, slowSMA)
shortCondition = ta.crossunder(fastSMA, slowSMA)

// Execute trades
if (longCondition)
    strategy.entry("Long", strategy.long)
if (shortCondition)
    strategy.close("Long")

// Plot
plot(fastSMA, color=color.blue, title="Fast SMA")
plot(slowSMA, color=color.red, title="Slow SMA")
'''
        cprint(example, "white")

if __name__ == "__main__":
    main()
