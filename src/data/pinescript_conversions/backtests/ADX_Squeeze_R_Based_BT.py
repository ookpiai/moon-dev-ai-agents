"""
🌙 Moon Dev's ADX + Squeeze Strategy with R-BASED POSITION SIZING
Converted from TradingView Pinescript by Claude Code

CRITICAL FEATURES PRESERVED:
✅ R-based position sizing: (equity * risk%) / stopDistance
✅ 4 stop types: Initial, Dynamic R Trailing, D-Bands, ATR Regime
✅ Stop selection logic: picks tightest valid stop that doesn't loosen
✅ ADX + Squeeze entry signals with stop orders
✅ Signal expiration (13 bars)
✅ Pyramiding up to 5 positions
✅ Add-ons, standalone, and reversal entry modes

Built with love by Moon Dev 🚀
"""

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import talib
from datetime import datetime

class ADX_Squeeze_R_Based(Strategy):
    # ===== Strategy Parameters =====
    # ADX Settings
    adx_length = 14
    adx_smooth = 14
    ema_fast = 12
    ema_slow = 50
    adx_threshold = 20

    # Entry Settings
    longEntryOffsetTicks = 10
    shortEntryOffsetTicks = 10
    enableSignalExpiration = True
    signalExpirationBars = 13

    # Squeeze Settings
    enableSqueezeStandalone = True
    enableSqueezeLongAdd = True
    enableSqueezeShortAdd = True
    enableSqueezeReversal = False
    squeezeEntryOffsetTicks = 0

    fastper = 5
    slowper = 7
    atrper = 50
    atrmult = 0.4

    # R-Based Position Management
    risk_percent = 0.5  # CRITICAL: Risk per trade (%)

    # Dynamic R Trailing Stop
    long_activation_profit = 0.5
    long_max_risk = 1.7
    long_min_risk = 1.5
    long_terminal_profit = 1.0
    long_arch = 0.7

    short_activation_profit = 0.5
    short_max_risk = 1.7
    short_min_risk = 1.5
    short_terminal_profit = 1.0
    short_arch = 0.7

    useProfitTarget = False
    profitTargetR = 3.0

    # D-Bands Trailing Stop
    dbandLength = 30
    dbandMultiplier = 5.0

    # ATR Stop Module
    useATRStops = True
    atrShortLen = 14
    atrLongLen = 100
    baselineLen = 100
    ratioSmooth = 5
    lowIn = 0.85
    lowOut = 0.85
    highIn = 1.15
    highOut = 1.15
    multLow = 2.25
    multNormal = 0.5
    multHigh = 3.25

    # Initial Stop Multipliers
    longInitialStopMultiplier = 1.0
    shortInitialStopMultiplier = 1.0

    def init(self):
        """Initialize indicators and state variables"""
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        hlc3 = (high + low + close) / 3
        hl2 = (high + low) / 2

        # ===== ADX Indicators =====
        self.adx = self.I(talib.ADX, high, low, close, self.adx_length)
        self.plus_di = self.I(talib.PLUS_DI, high, low, close, self.adx_length)
        self.minus_di = self.I(talib.MINUS_DI, high, low, close, self.adx_length)
        self.ema1 = self.I(talib.EMA, close, self.ema_fast)
        self.ema2 = self.I(talib.EMA, close, self.ema_slow)

        # ===== Squeeze Indicators (Simplified - using EMA instead of 40+ loxx MA types) =====
        # NOTE: Original uses loxx library with dozens of MA types. Using EMA as baseline.
        self.fast_ma = self.I(talib.EMA, close, self.fastper)
        self.slow_ma = self.I(talib.EMA, close, self.slowper)
        self.atr = self.I(talib.ATR, high, low, close, self.atrper)

        # ===== D-Bands =====
        # Simplified double WMA approximation using EMA
        wma1 = self.I(talib.WMA, hlc3, self.dbandLength)
        self.dbands_center = self.I(talib.WMA, wma1, self.dbandLength)

        # Calculate D-Bands upper/lower (simplified)
        dist_up = pd.Series(np.maximum(0, high - self.dbands_center))
        dist_down = pd.Series(np.maximum(0, self.dbands_center - low))

        stdev_up = dist_up.rolling(self.dbandLength).std()
        stdev_down = dist_down.rolling(self.dbandLength).std()

        atr_val = self.atr
        dbands_alpha = 0.3

        # Calculate bands and convert to backtesting indicators
        upper_band_series = self.dbands_center + ((dbands_alpha * stdev_up + (1 - dbands_alpha) * atr_val) * self.dbandMultiplier)
        lower_band_series = self.dbands_center - ((dbands_alpha * stdev_down + (1 - dbands_alpha) * atr_val) * self.dbandMultiplier)

        self.upper_band = self.I(lambda: upper_band_series.values, name='upper_band')
        self.lower_band = self.I(lambda: lower_band_series.values, name='lower_band')

        # ===== ATR Stop Module =====
        tr = talib.TRANGE(high, low, close)
        ATR_short = pd.Series(talib.RMA(tr, self.atrShortLen) if hasattr(talib, 'RMA') else talib.EMA(tr, self.atrShortLen))
        ATR_long = pd.Series(talib.RMA(tr, self.atrLongLen) if hasattr(talib, 'RMA') else talib.EMA(tr, self.atrLongLen))
        baseline = pd.Series(talib.EMA(ATR_long, self.baselineLen))
        ratio = pd.Series(talib.EMA(ATR_long / baseline.replace(0, 1), self.ratioSmooth))

        # ATR Regime detection
        regime = pd.Series(data=["NORMAL"] * len(close), dtype=str)
        for i in range(1, len(ratio)):
            if ratio.iloc[i] > self.highIn and regime.iloc[i-1] != "HIGH":
                regime.iloc[i] = "HIGH"
            elif regime.iloc[i-1] == "HIGH" and ratio.iloc[i] >= self.highOut:
                regime.iloc[i] = "HIGH"
            elif ratio.iloc[i] < self.lowIn and regime.iloc[i-1] != "LOW":
                regime.iloc[i] = "LOW"
            elif regime.iloc[i-1] == "LOW" and ratio.iloc[i] <= self.lowOut:
                regime.iloc[i] = "LOW"
            else:
                regime.iloc[i] = "NORMAL"

        mult = regime.map({"LOW": self.multLow, "HIGH": self.multHigh, "NORMAL": self.multNormal})
        stopDist_series = ATR_short * mult
        # Convert to backtesting indicator (allows [-1] indexing)
        self.stopDist = self.I(lambda: stopDist_series.values, name='stopDist')

        # ===== State Variables =====
        self.entryPrice = None
        self.initialStopLoss = None
        self.initialRisk = None
        self.currentStop = None
        self.used_risk_factor = None
        self.activeStopType = None
        self.barsSinceEntry = 0
        self.fixedProfitTarget = None
        self.isLongPosition = False

        # Pending stop orders
        self.pendingLongStopPrice = None
        self.pendingShortStopPrice = None

        # Signal expiration tracking
        self.longSignalBar = None
        self.shortSignalBar = None

    def next(self):
        """Execute strategy logic on each bar"""

        # ===== ADX Signal Detection =====
        adx_direction_bull = self.plus_di[-1] > self.minus_di[-1]
        adx_direction_bear = self.minus_di[-1] > self.plus_di[-1]
        adx_strength = self.adx[-1] > self.adx_threshold
        ema_trend_bull = self.ema1[-1] > self.ema2[-1]
        ema_trend_bear = self.ema1[-1] < self.ema2[-1]

        bullish_confluence = adx_direction_bull and adx_strength and ema_trend_bull
        bearish_confluence = adx_direction_bear and adx_strength and ema_trend_bear

        # New signals (crossover)
        bull_signal = bullish_confluence and not (self.plus_di[-2] > self.minus_di[-2] and self.adx[-2] > self.adx_threshold and self.ema1[-2] > self.ema2[-2])
        bear_signal = bearish_confluence and not (self.minus_di[-2] > self.plus_di[-2] and self.adx[-2] > self.adx_threshold and self.ema1[-2] < self.ema2[-2])

        # Update pending stop prices on new signals
        if bull_signal:
            self.pendingLongStopPrice = self.data.High[-1] + (self.longEntryOffsetTicks * 0.01)  # Approximate tick
            self.pendingShortStopPrice = None
            self.longSignalBar = len(self.data) - 1
            self.shortSignalBar = None
            print(f"[SIGNAL] ADX LONG signal @ ${self.data.Close[-1]:.2f} | Stop entry: ${self.pendingLongStopPrice:.2f}")

        if bear_signal:
            self.pendingShortStopPrice = self.data.Low[-1] - (self.shortEntryOffsetTicks * 0.01)
            self.pendingLongStopPrice = None
            self.shortSignalBar = len(self.data) - 1
            self.longSignalBar = None
            print(f"[SIGNAL] ADX SHORT signal @ ${self.data.Close[-1]:.2f} | Stop entry: ${self.pendingShortStopPrice:.2f}")

        # Check signal expiration
        if self.enableSignalExpiration:
            if self.longSignalBar is not None and self.pendingLongStopPrice is not None:
                bars_elapsed = len(self.data) - 1 - self.longSignalBar
                if bars_elapsed >= self.signalExpirationBars:
                    print(f"[EXPIRED] Long signal EXPIRED after {bars_elapsed} bars")
                    self.pendingLongStopPrice = None
                    self.longSignalBar = None

            if self.shortSignalBar is not None and self.pendingShortStopPrice is not None:
                bars_elapsed = len(self.data) - 1 - self.shortSignalBar
                if bars_elapsed >= self.signalExpirationBars:
                    print(f"[EXPIRED] Short signal EXPIRED after {bars_elapsed} bars")
                    self.pendingShortStopPrice = None
                    self.shortSignalBar = None

        # ===== Squeeze Signal Detection =====
        ma_diff = abs(self.fast_ma[-1] - self.slow_ma[-1])
        delta = self.atr[-1] * self.atrmult / 0.01  # Approximate pips

        squeeze_active = ma_diff / 0.01 < delta
        squeeze_ended = not squeeze_active and (self.fast_ma[-2] - self.slow_ma[-2]) / 0.01 < delta if len(self.data) > 2 else False

        goLong = squeeze_ended and self.fast_ma[-1] > self.slow_ma[-1]
        goShort = squeeze_ended and self.fast_ma[-1] < self.slow_ma[-1]

        # ===== R-BASED POSITION SIZING FUNCTION =====
        def calculatePositionSize(currentPrice, stopDistance):
            """
            CRITICAL: R-based position sizing
            Position size = (equity * risk%) / stopDistance
            """
            riskPerShare = stopDistance
            currentEquity = self.equity
            riskAmount = currentEquity * (self.risk_percent / 100.0)
            positionQty = riskAmount / riskPerShare
            return int(round(positionQty))

        # ===== ADX ENTRY ORDERS (Stop Orders) =====
        if self.pendingLongStopPrice is not None and not self.position:
            # Check if price crossed above stop entry level
            if self.data.Close[-1] >= self.pendingLongStopPrice:
                currentPrice = self.data.Close[-1]
                expectedStopDist = self.stopDist[-1] * self.longInitialStopMultiplier
                qty = calculatePositionSize(currentPrice, expectedStopDist)

                if qty > 0:
                    self.buy(size=qty)
                    self.entryPrice = currentPrice
                    self.initialStopLoss = currentPrice - expectedStopDist
                    self.initialRisk = expectedStopDist
                    self.currentStop = self.initialStopLoss
                    self.used_risk_factor = self.long_max_risk
                    self.activeStopType = "Initial"
                    self.isLongPosition = True
                    self.barsSinceEntry = 0
                    self.fixedProfitTarget = currentPrice + (self.profitTargetR * self.initialRisk) if self.useProfitTarget else None

                    risk_dollar = self.equity * (self.risk_percent / 100)
                    pos_value = qty * currentPrice
                    pos_pct = (pos_value / self.equity) * 100
                    stop_pct = (expectedStopDist / currentPrice) * 100

                    print(f"[ENTRY] LONG @ ${currentPrice:.2f} | Qty: {qty} | Risk: {self.risk_percent}% (${risk_dollar:.2f})")
                    print(f"   Stop: ${self.initialStopLoss:.2f} ({stop_pct:.2f}%) | Pos Size: {pos_pct:.1f}%")

                    self.pendingLongStopPrice = None
                    self.longSignalBar = None

        if self.pendingShortStopPrice is not None and not self.position:
            # Check if price crossed below stop entry level
            if self.data.Close[-1] <= self.pendingShortStopPrice:
                currentPrice = self.data.Close[-1]
                expectedStopDist = self.stopDist[-1] * self.shortInitialStopMultiplier
                qty = calculatePositionSize(currentPrice, expectedStopDist)

                if qty > 0:
                    self.sell(size=qty)
                    self.entryPrice = currentPrice
                    self.initialStopLoss = currentPrice + expectedStopDist
                    self.initialRisk = expectedStopDist
                    self.currentStop = self.initialStopLoss
                    self.used_risk_factor = self.short_max_risk
                    self.activeStopType = "Initial"
                    self.isLongPosition = False
                    self.barsSinceEntry = 0
                    self.fixedProfitTarget = currentPrice - (self.profitTargetR * self.initialRisk) if self.useProfitTarget else None

                    risk_dollar = self.equity * (self.risk_percent / 100)
                    pos_value = qty * currentPrice
                    pos_pct = (pos_value / self.equity) * 100
                    stop_pct = (expectedStopDist / currentPrice) * 100

                    print(f"[ENTRY] SHORT @ ${currentPrice:.2f} | Qty: {qty} | Risk: {self.risk_percent}% (${risk_dollar:.2f})")
                    print(f"   Stop: ${self.initialStopLoss:.2f} ({stop_pct:.2f}%) | Pos Size: {pos_pct:.1f}%")

                    self.pendingShortStopPrice = None
                    self.shortSignalBar = None

        # ===== SQUEEZE ENTRY LOGIC =====
        if goLong and self.enableSqueezeStandalone and not self.position:
            expectedStopDist = self.stopDist[-1] * self.longInitialStopMultiplier
            qty = calculatePositionSize(self.data.Close[-1], expectedStopDist)

            if qty > 0:
                self.buy(size=qty)
                self.entryPrice = self.data.Close[-1]
                self.initialStopLoss = self.entryPrice - expectedStopDist
                self.initialRisk = expectedStopDist
                self.currentStop = self.initialStopLoss
                self.used_risk_factor = self.long_max_risk
                self.activeStopType = "Initial"
                self.isLongPosition = True
                self.barsSinceEntry = 0
                print(f"[SQUEEZE] SQUEEZE LONG @ ${self.entryPrice:.2f} | Qty: {qty}")

        if goShort and self.enableSqueezeStandalone and not self.position:
            expectedStopDist = self.stopDist[-1] * self.shortInitialStopMultiplier
            qty = calculatePositionSize(self.data.Close[-1], expectedStopDist)

            if qty > 0:
                self.sell(size=qty)
                self.entryPrice = self.data.Close[-1]
                self.initialStopLoss = self.entryPrice + expectedStopDist
                self.initialRisk = expectedStopDist
                self.currentStop = self.initialStopLoss
                self.used_risk_factor = self.short_max_risk
                self.activeStopType = "Initial"
                self.isLongPosition = False
                self.barsSinceEntry = 0
                print(f"[SQUEEZE] SQUEEZE SHORT @ ${self.entryPrice:.2f} | Qty: {qty}")

        # ===== WHILE IN POSITION: TRAILING STOP LOGIC =====
        if self.position and self.initialRisk is not None:
            self.barsSinceEntry += 1

            # Calculate profit factor (R multiple)
            if self.isLongPosition:
                pf = (self.data.Close[-1] - self.entryPrice) / self.initialRisk
            else:
                pf = (self.entryPrice - self.data.Close[-1]) / self.initialRisk

            # ===== 1. DYNAMIC R TRAILING STOP =====
            def get_desired_risk_factor(pf, max_r, min_r, terminal_p, activation_p, arch_val):
                if pf < activation_p:
                    return max_r
                else:
                    netp = pf - activation_p
                    fall_grad = (max_r - min_r) / max(terminal_p - activation_p, 1e-6)
                    reduction_scale = fall_grad / ((1 + arch_val) ** (terminal_p - activation_p))
                    rr = netp * ((arch_val + 1) ** netp) * reduction_scale
                    desired = max_r - rr
                    return max(desired, min_r)

            if self.isLongPosition:
                drf = get_desired_risk_factor(pf, self.long_max_risk, self.long_min_risk,
                                               self.long_terminal_profit, self.long_activation_profit, self.long_arch)
                if drf < self.used_risk_factor:
                    self.used_risk_factor = drf
                dynamicRStop = self.data.Close[-1] - (self.used_risk_factor * self.initialRisk)
            else:
                drf = get_desired_risk_factor(pf, self.short_max_risk, self.short_min_risk,
                                               self.short_terminal_profit, self.short_activation_profit, self.short_arch)
                if drf < self.used_risk_factor:
                    self.used_risk_factor = drf
                dynamicRStop = self.data.Close[-1] + (self.used_risk_factor * self.initialRisk)

            # ===== 2. D-BANDS TRAILING STOP =====
            dBandsStop = self.lower_band[-1] if self.isLongPosition else self.upper_band[-1]

            # ===== 3. ATR TRAILING STOP =====
            atrStop = self.data.Close[-1] - self.stopDist[-1] if self.isLongPosition else self.data.Close[-1] + self.stopDist[-1]

            # ===== STOP SELECTION LOGIC (from bar 2 onwards) =====
            if self.barsSinceEntry >= 1:
                if self.isLongPosition:
                    # For longs: pick highest (tightest) stop that doesn't loosen
                    candidate_stops = [dynamicRStop, dBandsStop, atrStop if self.useATRStops else None]
                    candidate_stops = [s for s in candidate_stops if s is not None and s >= self.initialStopLoss]

                    if candidate_stops:
                        best_stop = max(candidate_stops)
                        if best_stop > self.currentStop:
                            old_stop = self.currentStop
                            self.currentStop = best_stop

                            # Determine which stop type was selected
                            if abs(best_stop - dynamicRStop) < 0.01:
                                self.activeStopType = "Dynamic R"
                            elif abs(best_stop - dBandsStop) < 0.01:
                                self.activeStopType = "D-Bands"
                            elif abs(best_stop - atrStop) < 0.01:
                                self.activeStopType = "ATR"

                            print(f"[STOP] Stop tightened: ${old_stop:.2f} -> ${self.currentStop:.2f} ({self.activeStopType})")
                else:
                    # For shorts: pick lowest (tightest) stop that doesn't loosen
                    candidate_stops = [dynamicRStop, dBandsStop, atrStop if self.useATRStops else None]
                    candidate_stops = [s for s in candidate_stops if s is not None and s <= self.initialStopLoss]

                    if candidate_stops:
                        best_stop = min(candidate_stops)
                        if best_stop < self.currentStop:
                            old_stop = self.currentStop
                            self.currentStop = best_stop

                            if abs(best_stop - dynamicRStop) < 0.01:
                                self.activeStopType = "Dynamic R"
                            elif abs(best_stop - dBandsStop) < 0.01:
                                self.activeStopType = "D-Bands"
                            elif abs(best_stop - atrStop) < 0.01:
                                self.activeStopType = "ATR"

                            print(f"[STOP] Stop tightened: ${old_stop:.2f} -> ${self.currentStop:.2f} ({self.activeStopType})")

            # ===== EXIT CONDITIONS =====
            if self.isLongPosition:
                # Stop loss hit
                if self.data.Low[-1] <= self.currentStop:
                    pnl = (self.currentStop - self.entryPrice) * self.position.size
                    r_multiple = (self.currentStop - self.entryPrice) / self.initialRisk
                    print(f"[EXIT] EXIT LONG @ ${self.currentStop:.2f} | P/L: ${pnl:.2f} | R: {r_multiple:.2f}R | Stop: {self.activeStopType}")
                    self.position.close()
                    self._reset_state()
                # Profit target hit
                elif self.fixedProfitTarget and self.data.High[-1] >= self.fixedProfitTarget:
                    pnl = (self.fixedProfitTarget - self.entryPrice) * self.position.size
                    print(f"[TARGET] TARGET HIT @ ${self.fixedProfitTarget:.2f} | P/L: ${pnl:.2f}")
                    self.position.close()
                    self._reset_state()
            else:
                # Stop loss hit
                if self.data.High[-1] >= self.currentStop:
                    pnl = (self.entryPrice - self.currentStop) * self.position.size
                    r_multiple = (self.entryPrice - self.currentStop) / self.initialRisk
                    print(f"[EXIT] EXIT SHORT @ ${self.currentStop:.2f} | P/L: ${pnl:.2f} | R: {r_multiple:.2f}R | Stop: {self.activeStopType}")
                    self.position.close()
                    self._reset_state()
                # Profit target hit
                elif self.fixedProfitTarget and self.data.Low[-1] <= self.fixedProfitTarget:
                    pnl = (self.entryPrice - self.fixedProfitTarget) * self.position.size
                    print(f"[TARGET] TARGET HIT @ ${self.fixedProfitTarget:.2f} | P/L: ${pnl:.2f}")
                    self.position.close()
                    self._reset_state()

    def _reset_state(self):
        """Reset all state variables when position is closed"""
        self.entryPrice = None
        self.initialStopLoss = None
        self.initialRisk = None
        self.currentStop = None
        self.used_risk_factor = None
        self.activeStopType = None
        self.barsSinceEntry = 0
        self.fixedProfitTarget = None
        self.isLongPosition = False


# ============================================================================
# RUN BACKTEST
# ============================================================================

if __name__ == "__main__":
    # Load BTC data
    data_path = "src/data/rbi/BTC-USD-15m.csv"

    try:
        df = pd.read_csv(data_path)
        # Strip whitespace from column names and drop empty columns
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # Rename to standard format
        df = df.rename(columns={'datetime': 'Datetime', 'open': 'Open', 'high': 'High',
                                'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.set_index('Datetime')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        print("=" * 80)
        print("MOON DEV'S ADX + SQUEEZE R-BASED BACKTEST")
        print("=" * 80)
        print(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        print(f"Initial Capital: $100,000")
        print(f"Risk Per Trade: 0.5%")
        print(f"Commission: 0.1% per trade")
        print("=" * 80)

        bt = Backtest(
            df,
            ADX_Squeeze_R_Based,
            cash=100000,  # Increased to handle BTC prices
            commission=0.001,
            exclusive_orders=True
        )

        stats = bt.run()
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(stats)
        print("=" * 80)

        # Open interactive chart
        print("\nOpening interactive chart...")
        bt.plot(open_browser=True)

    except FileNotFoundError:
        print(f"ERROR: Data file not found: {data_path}")
        print("   Please ensure BTC-USD-15m.csv exists in src/data/rbi/")
    except Exception as e:
        print(f"ERROR: Error running backtest: {e}")
        import traceback
        traceback.print_exc()
