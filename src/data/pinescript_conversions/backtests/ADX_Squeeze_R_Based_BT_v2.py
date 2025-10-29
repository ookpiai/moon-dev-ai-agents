"""
ADX + Squeeze [R-BASED] Strategy - Exact Pinescript Conversion
Converted from TradingView Pinescript by Claude Code

CRITICAL CONVERSION NOTES:
- Squeeze uses EMA(5) vs EMA(7) with ATR(50)*0.4 threshold
- Source: "Close" (no transformations needed)
- R-based position sizing: qty = (equity * risk%) / stopDistance
- 4 stop types: Initial, Dynamic R, D-Bands, ATR Regime
- Stop selection: tightest valid stop that doesn't loosen

Built by Moon Dev
"""

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import talib
from datetime import datetime

class ADX_Squeeze_R_Based(Strategy):
    # ===== ADX Settings =====
    adx_length = 14
    adx_smooth = 14
    ema_fast = 12
    ema_slow = 50
    adx_threshold = 20

    # ===== Entry Settings =====
    tradingDirection = "Long Only"  # "Long Only", "Short Only", "Both"
    longEntryOffsetTicks = 10
    shortEntryOffsetTicks = 10
    enableSignalExpiration = True
    signalExpirationBars = 13

    enableSqueezeStandalone = True
    enableSqueezeLongAdd = True
    enableSqueezeShortAdd = True
    enableSqueezeReversal = False
    squeezeEntryOffsetTicks = 0

    # ===== Squeeze Settings =====
    # Using EMA as per Pinescript defaults (loxx library with "Close" source = EMA)
    # fsmthtype = "Kaufman" (only used for HAB sources, not "Close")
    # fastsrc = "Close"
    # ftype = "Exponential Moving Average - EMA"
    fastper = 5

    # ssmthtype = "Kaufman" (only used for HAB sources, not "Close")
    # slowsrc = "Close"
    # stype = "Exponential Moving Average - EMA"
    slowper = 7  # EXACT match to Pinescript default

    # Squeeze Filter
    atrper = 50
    atrmult = 0.4
    filttype = "ATR"  # or "Pips"
    pipsfiltin = 36  # Only used if filttype == "Pips"

    # ===== R-Based Position Management =====
    risk_percent = 0.5

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

    # ===== D-Bands Trailing Stop =====
    dbandLength = 30
    dbandMultiplier = 5.0

    # ===== ATR Stop Module =====
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

    # ===== Initial Stop Adjustment =====
    longInitialStopMultiplier = 1.0
    shortInitialStopMultiplier = 1.0

    def init(self):
        """Initialize indicators and state variables"""
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        hlc3 = (high + low + close) / 3

        # ===== ADX Indicators =====
        self.adx = self.I(talib.ADX, high, low, close, self.adx_length)
        self.plus_di = self.I(talib.PLUS_DI, high, low, close, self.adx_length)
        self.minus_di = self.I(talib.MINUS_DI, high, low, close, self.adx_length)
        self.ema1 = self.I(talib.EMA, close, self.ema_fast)
        self.ema2 = self.I(talib.EMA, close, self.ema_slow)

        # ===== Squeeze Indicators =====
        # Using EMA as per strategy defaults (not Kaufman)
        self.fast_ma = self.I(talib.EMA, close, self.fastper)
        self.slow_ma = self.I(talib.EMA, close, self.slowper)
        self.atr = self.I(talib.ATR, high, low, close, self.atrper)

        # ===== D-Bands (Double WMA) =====
        def double_wma(src, length):
            """Double WMA: WMA of WMA"""
            w1 = talib.WMA(src, length)
            w2 = talib.WMA(w1, length)
            return w2

        # Calculate center line
        self.dbands_center = self.I(double_wma, hlc3, self.dbandLength)

        # Calculate asymmetric bands
        dist_up_raw = np.maximum(0, high - self.dbands_center)
        dist_down_raw = np.maximum(0, self.dbands_center - low)

        # WMA of distances
        dist_up = talib.WMA(dist_up_raw, self.dbandLength)
        dist_down = talib.WMA(dist_down_raw, self.dbandLength)

        # Double WMA of standard deviations
        stdev_up_raw = pd.Series(dist_up).rolling(self.dbandLength).std().values
        stdev_down_raw = pd.Series(dist_down).rolling(self.dbandLength).std().values

        dbands_stDevUp = double_wma(stdev_up_raw, self.dbandLength)
        dbands_stDevDown = double_wma(stdev_down_raw, self.dbandLength)

        # Calculate bands
        dbands_alpha = 0.3
        upper_band_raw = (self.dbands_center +
                         ((dbands_alpha * dbands_stDevUp + (1 - dbands_alpha) * self.atr) *
                          self.dbandMultiplier))
        lower_band_raw = (self.dbands_center -
                         ((dbands_alpha * dbands_stDevDown + (1 - dbands_alpha) * self.atr) *
                          self.dbandMultiplier))

        self.upper_band = self.I(lambda: upper_band_raw, name='upper_band')
        self.lower_band = self.I(lambda: lower_band_raw, name='lower_band')

        # ===== ATR Stop Module with Regime Detection =====
        tr = talib.TRANGE(high, low, close)
        ATR_short = pd.Series(talib.RMA(tr, self.atrShortLen) if hasattr(talib, 'RMA')
                              else talib.EMA(tr, self.atrShortLen))
        ATR_long = pd.Series(talib.RMA(tr, self.atrLongLen) if hasattr(talib, 'RMA')
                             else talib.EMA(tr, self.atrLongLen))
        baseline = pd.Series(talib.EMA(ATR_long.values, self.baselineLen))

        # Avoid division by zero
        ratio_raw = ATR_long / baseline.replace(0, 1)
        ratio = pd.Series(talib.EMA(ratio_raw.values, self.ratioSmooth))

        # ATR Regime detection with hysteresis
        regime = pd.Series(data=["NORMAL"] * len(close), dtype=str)
        for i in range(1, len(ratio)):
            prev_regime = regime.iloc[i-1]
            r = ratio.iloc[i]

            # State transitions with hysteresis
            if prev_regime != "HIGH" and r > self.highIn:
                regime.iloc[i] = "HIGH"
            elif prev_regime == "HIGH" and r >= self.highOut:
                regime.iloc[i] = "HIGH"
            elif prev_regime != "LOW" and r < self.lowIn:
                regime.iloc[i] = "LOW"
            elif prev_regime == "LOW" and r <= self.lowOut:
                regime.iloc[i] = "LOW"
            else:
                regime.iloc[i] = "NORMAL"

        # Map regime to multiplier
        mult = regime.map({"LOW": self.multLow, "HIGH": self.multHigh, "NORMAL": self.multNormal})
        stopDist_series = ATR_short * mult

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

        # Track last trade type
        self.lastTradeType = "NONE"

    def next(self):
        """Execute strategy logic on each bar"""

        # ===== Trading Direction Flags =====
        allowLongs = self.tradingDirection in ["Long Only", "Both"]
        allowShorts = self.tradingDirection in ["Short Only", "Both"]

        # ===== ADX Signal Detection =====
        adx_direction_bull = self.plus_di[-1] > self.minus_di[-1]
        adx_direction_bear = self.minus_di[-1] > self.plus_di[-1]
        adx_strength = self.adx[-1] > self.adx_threshold
        ema_trend_bull = self.ema1[-1] > self.ema2[-1]
        ema_trend_bear = self.ema1[-1] < self.ema2[-1]

        bullish_confluence = adx_direction_bull and adx_strength and ema_trend_bull
        bearish_confluence = adx_direction_bear and adx_strength and ema_trend_bear

        # New signals (crossover to confluence state)
        bull_signal = (bullish_confluence and
                      not (self.plus_di[-2] > self.minus_di[-2] and
                           self.adx[-2] > self.adx_threshold and
                           self.ema1[-2] > self.ema2[-2]))

        bear_signal = (bearish_confluence and
                      not (self.minus_di[-2] > self.plus_di[-2] and
                           self.adx[-2] > self.adx_threshold and
                           self.ema1[-2] < self.ema2[-2]))

        # Update pending stop prices on new ADX signals
        if bull_signal and allowLongs:
            self.pendingLongStopPrice = self.data.High[-1] + (self.longEntryOffsetTicks * 0.01)
            self.pendingShortStopPrice = None
            self.longSignalBar = len(self.data) - 1
            self.shortSignalBar = None

        if bear_signal and allowShorts:
            self.pendingShortStopPrice = self.data.Low[-1] - (self.shortEntryOffsetTicks * 0.01)
            self.pendingLongStopPrice = None
            self.shortSignalBar = len(self.data) - 1
            self.longSignalBar = None

        # Check signal expiration
        if self.enableSignalExpiration:
            if self.longSignalBar is not None and self.pendingLongStopPrice is not None:
                bars_elapsed = len(self.data) - 1 - self.longSignalBar
                if bars_elapsed >= self.signalExpirationBars:
                    self.pendingLongStopPrice = None
                    self.longSignalBar = None

            if self.shortSignalBar is not None and self.pendingShortStopPrice is not None:
                bars_elapsed = len(self.data) - 1 - self.shortSignalBar
                if bars_elapsed >= self.signalExpirationBars:
                    self.pendingShortStopPrice = None
                    self.shortSignalBar = None

        # ===== Squeeze Signal Detection =====
        # Calculate pip/tick size (simplified for BTC)
        pip_size = 0.01  # Approximate for BTC

        ma_diff = abs(self.fast_ma[-1] - self.slow_ma[-1])

        # Calculate threshold
        if self.filttype == "ATR":
            delta = self.atr[-1] * self.atrmult / pip_size
        else:
            delta = self.pipsfiltin

        # Squeeze detection
        squeeze_active = (ma_diff / pip_size) < delta
        squeeze_active_prev = False
        if len(self.data) > 2:
            ma_diff_prev = abs(self.fast_ma[-2] - self.slow_ma[-2])
            squeeze_active_prev = (ma_diff_prev / pip_size) < delta

        # Squeeze END signal
        squeeze_ended = not squeeze_active and squeeze_active_prev

        goLong = squeeze_ended and self.fast_ma[-1] > self.slow_ma[-1]
        goShort = squeeze_ended and self.fast_ma[-1] < self.slow_ma[-1]

        # ===== R-BASED POSITION SIZING FUNCTION =====
        def calculatePositionSize(currentPrice, stopDistance):
            """R-based position sizing: qty = (equity * risk%) / stopDistance"""
            if stopDistance <= 0:
                return 0

            riskPerShare = stopDistance
            currentEquity = self.equity
            riskAmount = currentEquity * (self.risk_percent / 100.0)
            positionQty = riskAmount / riskPerShare

            return int(round(positionQty))

        # ===== ADX ENTRY ORDERS (Stop Orders) =====
        if self.pendingLongStopPrice is not None and allowLongs and not self.position:
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
                    self.fixedProfitTarget = (currentPrice + (self.profitTargetR * self.initialRisk)
                                             if self.useProfitTarget else None)
                    self.lastTradeType = "ADX"

                    self.pendingLongStopPrice = None
                    self.longSignalBar = None

        if self.pendingShortStopPrice is not None and allowShorts and not self.position:
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
                    self.fixedProfitTarget = (currentPrice - (self.profitTargetR * self.initialRisk)
                                             if self.useProfitTarget else None)
                    self.lastTradeType = "ADX"

                    self.pendingShortStopPrice = None
                    self.shortSignalBar = None

        # ===== SQUEEZE ENTRY LOGIC =====
        # Standalone entries
        if goLong and self.enableSqueezeStandalone and not self.position and allowLongs:
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
                self.fixedProfitTarget = (self.entryPrice + (self.profitTargetR * self.initialRisk)
                                         if self.useProfitTarget else None)
                self.lastTradeType = "SQUEEZE"

        if goShort and self.enableSqueezeStandalone and not self.position and allowShorts:
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
                self.fixedProfitTarget = (self.entryPrice - (self.profitTargetR * self.initialRisk)
                                         if self.useProfitTarget else None)
                self.lastTradeType = "SQUEEZE"

        # ===== WHILE IN POSITION: TRAILING STOP LOGIC =====
        if self.position and self.initialRisk is not None and self.initialRisk > 0:
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
                                               self.long_terminal_profit, self.long_activation_profit,
                                               self.long_arch)
                if drf < self.used_risk_factor:
                    self.used_risk_factor = drf
                dynamicRStop = self.data.Close[-1] - (self.used_risk_factor * self.initialRisk)
            else:
                drf = get_desired_risk_factor(pf, self.short_max_risk, self.short_min_risk,
                                               self.short_terminal_profit, self.short_activation_profit,
                                               self.short_arch)
                if drf < self.used_risk_factor:
                    self.used_risk_factor = drf
                dynamicRStop = self.data.Close[-1] + (self.used_risk_factor * self.initialRisk)

            # ===== 2. D-BANDS TRAILING STOP =====
            dBandsStop = self.lower_band[-1] if self.isLongPosition else self.upper_band[-1]

            # ===== 3. ATR TRAILING STOP =====
            atrStop = (self.data.Close[-1] - self.stopDist[-1] if self.isLongPosition
                      else self.data.Close[-1] + self.stopDist[-1])

            # ===== STOP SELECTION LOGIC (from bar 2 onwards) =====
            if self.barsSinceEntry >= 1:
                if self.isLongPosition:
                    # For longs: pick highest (tightest) stop that doesn't loosen
                    candidate_stops = []

                    if dynamicRStop >= self.initialStopLoss:
                        candidate_stops.append(('Dynamic R', dynamicRStop))

                    if dBandsStop >= self.initialStopLoss:
                        candidate_stops.append(('D-Bands', dBandsStop))

                    if self.useATRStops and atrStop >= self.initialStopLoss:
                        candidate_stops.append(('ATR', atrStop))

                    if candidate_stops:
                        # Pick highest stop
                        best_stop_name, best_stop = max(candidate_stops, key=lambda x: x[1])

                        if best_stop > self.currentStop:
                            self.currentStop = best_stop
                            self.activeStopType = best_stop_name
                else:
                    # For shorts: pick lowest (tightest) stop that doesn't loosen
                    candidate_stops = []

                    if dynamicRStop <= self.initialStopLoss:
                        candidate_stops.append(('Dynamic R', dynamicRStop))

                    if dBandsStop <= self.initialStopLoss:
                        candidate_stops.append(('D-Bands', dBandsStop))

                    if self.useATRStops and atrStop <= self.initialStopLoss:
                        candidate_stops.append(('ATR', atrStop))

                    if candidate_stops:
                        # Pick lowest stop
                        best_stop_name, best_stop = min(candidate_stops, key=lambda x: x[1])

                        if best_stop < self.currentStop:
                            self.currentStop = best_stop
                            self.activeStopType = best_stop_name

            # ===== EXIT CONDITIONS =====
            if self.isLongPosition:
                # Stop loss hit
                if self.data.Low[-1] <= self.currentStop:
                    self.position.close()
                    self._reset_state()
                # Profit target hit
                elif self.fixedProfitTarget and self.data.High[-1] >= self.fixedProfitTarget:
                    self.position.close()
                    self._reset_state()
            else:
                # Stop loss hit
                if self.data.High[-1] >= self.currentStop:
                    self.position.close()
                    self._reset_state()
                # Profit target hit
                elif self.fixedProfitTarget and self.data.Low[-1] <= self.fixedProfitTarget:
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
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.rename(columns={'datetime': 'Datetime', 'open': 'Open', 'high': 'High',
                                'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.set_index('Datetime')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        print("=" * 80)
        print("ADX + SQUEEZE [R-BASED] STRATEGY - EXACT PINESCRIPT CONVERSION")
        print("=" * 80)
        print(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        print(f"Initial Capital: $100,000")
        print(f"Risk Per Trade: 0.5%")
        print(f"Commission: 0.1% per trade")
        print()
        print("Strategy Settings:")
        print(f"  Squeeze: EMA({5}) vs EMA({7}), ATR({50})*{0.4} threshold")
        print(f"  ADX: Length={14}, Threshold={20}")
        print(f"  Signal Expiration: {13} bars")
        print(f"  Initial Stop Multiplier: Long={1.0}x, Short={1.0}x")
        print(f"  D-Bands: Length={30}, Multiplier={5.0}")
        print(f"  ATR Regime: Short={14}, Long={100}, Baseline={100}")
        print("=" * 80)

        bt = Backtest(
            df,
            ADX_Squeeze_R_Based,
            cash=100000,
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
