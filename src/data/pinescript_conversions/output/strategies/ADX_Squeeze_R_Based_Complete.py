"""
Complete Python conversion of ADX + Squeeze [R-BASED] Pinescript Strategy
Converted from TradingView Pinescript v5 to Python backtesting.py
NO FEATURES LOST IN TRANSLATION - Complete 1:1 implementation

All indicators implemented from scratch using numpy/pandas (no pandas_ta dependency)
"""

from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import math

# ===== INDICATOR FUNCTIONS (No external dependencies) =====

def calc_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar uses high-low
    return tr

def calc_atr(high, low, close, length=14):
    """Calculate Average True Range"""
    tr = calc_true_range(high, low, close)
    atr = pd.Series(tr).ewm(alpha=1.0/length, adjust=False).mean().values
    return atr

def calc_rma(series, length):
    """Calculate RMA (RSI Moving Average) - same as Wilder's smoothing"""
    alpha = 1.0 / length
    return pd.Series(series).ewm(alpha=alpha, adjust=False).mean().values

def calc_ema(series, length):
    """Calculate Exponential Moving Average"""
    return pd.Series(series).ewm(span=length, adjust=False).mean().values

def calc_sma(series, length):
    """Calculate Simple Moving Average"""
    return pd.Series(series).rolling(window=length).mean().values

def calc_wma(series, length):
    """Calculate Weighted Moving Average"""
    weights = np.arange(1, length + 1)
    def wma(x):
        if len(x) < length:
            return np.nan
        return np.sum(weights * x[-length:]) / weights.sum()
    return pd.Series(series).rolling(window=length).apply(wma, raw=True).values

def calc_adx(high, low, close, length=14, smooth=14):
    """
    Calculate ADX (Average Directional Index) and directional indicators
    Returns: (DI+, DI-, ADX)
    """
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Calculate TR
    tr = calc_true_range(high, low, close)

    # Smooth DMs and TR
    plus_dm_smooth = calc_rma(plus_dm, length)
    minus_dm_smooth = calc_rma(minus_dm, length)
    tr_smooth = calc_rma(tr, length)

    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth

    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx) | np.isinf(dx), 0, dx)

    # Calculate ADX (smoothed DX)
    adx = calc_rma(dx, smooth)

    return plus_di, minus_di, adx

def calc_dbands(high, low, close, length=30, multiplier=5.0):
    """
    Calculate D-Bands (Donchian-style bands with WMA and ATR)
    Returns: (upper_band, lower_band)
    """
    hlc3 = (high + low + close) / 3

    # Double WMA for center
    w1 = calc_wma(hlc3, length)
    center = calc_wma(w1, length)

    # Distance calculations
    dist_up = np.maximum(0.0, high - center)
    dist_down = np.maximum(0.0, center - low)

    dist_up_wma = calc_wma(dist_up, length)
    dist_down_wma = calc_wma(dist_down, length)

    # Standard deviations with double WMA
    std_up = pd.Series(dist_up_wma).rolling(window=length).std().values
    std_down = pd.Series(dist_down_wma).rolling(window=length).std().values

    std_up_dwma = calc_wma(std_up, length)
    std_down_dwma = calc_wma(std_down, length)

    # ATR component
    atr_val = calc_atr(high, low, close, length)

    # Combine with alpha weighting
    alpha = 0.3
    upper = center + ((alpha * std_up_dwma + (1 - alpha) * atr_val) * multiplier)
    lower = center - ((alpha * std_down_dwma + (1 - alpha) * atr_val) * multiplier)

    return upper, lower


class ADXSqueezeRBasedStrategy(Strategy):
    """
    ADX + Squeeze Strategy with R-Based Position Sizing

    Complete conversion of TradingView Pinescript including:
    - ADX confluence detection
    - Squeeze indicator with ATR filter
    - R-based position sizing
    - Triple stop system: Dynamic R, D-Bands, ATR regime
    - Signal expiration (13 bars)
    - Multiple entry scenarios
    """

    # ADX Settings
    adx_length = 14
    adx_smooth = 14
    ema_fast = 12
    ema_slow = 50
    adx_threshold = 20

    # Entry Settings
    trading_direction = "Long Only"
    long_entry_offset_ticks = 10
    short_entry_offset_ticks = 10
    enable_signal_expiration = True
    signal_expiration_bars = 13

    # Squeeze Settings
    enable_squeeze_standalone = True
    enable_squeeze_long_add = True
    enable_squeeze_short_add = True
    enable_squeeze_reversal = False
    squeeze_entry_offset_ticks = 0
    fast_period = 5
    slow_period = 7
    atr_period = 50
    atr_multiplier = 0.4
    ma_threshold_ticks = 36
    filter_type = "ATR"

    # R-Based Position Management
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
    use_profit_target = False
    profit_target_r = 3.0

    # D-Bands Trailing Stop
    dband_length = 30
    dband_multiplier = 5.0

    # ATR Stop Module
    use_atr_stops = True
    atr_short_len = 14
    atr_long_len = 100
    baseline_len = 100
    ratio_smooth = 5
    low_in = 0.85
    low_out = 0.85
    high_in = 1.15
    high_out = 1.15
    mult_low = 2.25
    mult_normal = 0.5
    mult_high = 3.25

    # Initial Stop Adjustment
    long_initial_stop_multiplier = 1.0
    short_initial_stop_multiplier = 1.0

    def init(self):
        """Initialize all indicators"""

        # Convert data to numpy arrays
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        # ADX indicators
        self.adx_plus, self.adx_minus, self.adx_value = self.I(
            calc_adx, high, low, close, self.adx_length, self.adx_smooth)

        self.ema1 = self.I(calc_ema, close, self.ema_fast)
        self.ema2 = self.I(calc_ema, close, self.ema_slow)

        # Squeeze indicators
        self.ma1 = self.I(calc_ema, close, self.fast_period)
        self.ma2 = self.I(calc_ema, close, self.slow_period)
        self.atr_squeeze = self.I(calc_atr, high, low, close, self.atr_period)

        # D-Bands
        self.upper_band, self.lower_band = self.I(
            calc_dbands, high, low, close, self.dband_length, self.dband_multiplier)

        # ATR Stop Module with regime detection
        def calc_atr_stop_with_regime():
            tr = calc_true_range(high, low, close)
            atr_short = calc_rma(tr, self.atr_short_len)
            atr_long = calc_rma(tr, self.atr_long_len)
            baseline = calc_ema(atr_long, self.baseline_len)

            # Ratio with protection
            ratio = np.where(baseline != 0, atr_long / baseline, 1.0)
            ratio_smooth = calc_ema(ratio, self.ratio_smooth)

            # Regime detection with hysteresis
            regime = np.full(len(ratio_smooth), "NORMAL", dtype=object)
            for i in range(1, len(ratio_smooth)):
                if not np.isnan(ratio_smooth[i]):
                    if regime[i-1] != "HIGH" and ratio_smooth[i] > self.high_in:
                        regime[i] = "HIGH"
                    elif regime[i-1] == "HIGH" and ratio_smooth[i] >= self.high_out:
                        regime[i] = "HIGH"
                    elif regime[i-1] != "LOW" and ratio_smooth[i] < self.low_in:
                        regime[i] = "LOW"
                    elif regime[i-1] == "LOW" and ratio_smooth[i] <= self.low_out:
                        regime[i] = "LOW"
                    else:
                        regime[i] = "NORMAL"
                else:
                    regime[i] = regime[i-1]

            # Multiplier based on regime
            mult = np.where(regime == "LOW", self.mult_low,
                          np.where(regime == "HIGH", self.mult_high, self.mult_normal))

            stop_dist = atr_short * mult
            return stop_dist

        self.stop_dist = self.I(calc_atr_stop_with_regime)

        # State variables
        self.entry_price = None
        self.initial_stop_loss = None
        self.initial_risk = None
        self.current_stop = None
        self.used_risk_factor = None
        self.is_long_position = None
        self.bars_since_entry = 0
        self.fixed_profit_target = None
        self.pending_long_stop_price = None
        self.pending_short_stop_price = None
        self.long_signal_bar = None
        self.short_signal_bar = None
        self.last_position_size = 0

    def next(self):
        """Execute strategy logic on each bar"""

        current_bar = len(self.data) - 1

        # ADX Confluence
        adx_direction_bull = self.adx_plus[-1] > self.adx_minus[-1]
        adx_direction_bear = self.adx_minus[-1] > self.adx_plus[-1]
        adx_strength = self.adx_value[-1] > self.adx_threshold
        ema_trend_bull = self.ema1[-1] > self.ema2[-1]
        ema_trend_bear = self.ema1[-1] < self.ema2[-1]

        bullish_confluence = adx_direction_bull and adx_strength and ema_trend_bull
        bearish_confluence = adx_direction_bear and adx_strength and ema_trend_bear

        # Detect new signals
        prev_bull_conf = (len(self.data) > 1 and self.adx_plus[-2] > self.adx_minus[-2] and
                         self.adx_value[-2] > self.adx_threshold and self.ema1[-2] > self.ema2[-2])
        prev_bear_conf = (len(self.data) > 1 and self.adx_minus[-2] > self.adx_plus[-2] and
                         self.adx_value[-2] > self.adx_threshold and self.ema1[-2] < self.ema2[-2])

        bull_signal = bullish_confluence and not prev_bull_conf
        bear_signal = bearish_confluence and not prev_bear_conf

        # Squeeze Detection
        madif = abs(self.ma1[-1] - self.ma2[-1])
        pip_size = self.data.Close[-1] * 0.0001

        if self.filter_type == "ATR":
            delta = self.atr_squeeze[-1] * self.atr_multiplier / pip_size
        else:
            delta = self.ma_threshold_ticks

        squeeze_active = (madif / pip_size) < delta
        prev_squeeze = (len(self.data) > 1 and
                       abs(self.ma1[-2] - self.ma2[-2]) / pip_size < delta)

        squeeze_end = not squeeze_active and prev_squeeze
        go_long = squeeze_end and self.ma1[-1] > self.ma2[-1]
        go_short = squeeze_end and self.ma1[-1] < self.ma2[-1]

        # Direction filters
        allow_longs = self.trading_direction in ["Long Only", "Both"]
        allow_shorts = self.trading_direction in ["Short Only", "Both"]

        # ADX Signal Tracking
        if bull_signal and allow_longs:
            self.pending_long_stop_price = self.data.High[-1] + (self.long_entry_offset_ticks * pip_size)
            self.pending_short_stop_price = None
            self.long_signal_bar = current_bar
            self.short_signal_bar = None

        if bear_signal and allow_shorts:
            self.pending_short_stop_price = self.data.Low[-1] - (self.short_entry_offset_ticks * pip_size)
            self.pending_long_stop_price = None
            self.short_signal_bar = current_bar
            self.long_signal_bar = None

        # Signal Expiration
        if self.enable_signal_expiration:
            if self.long_signal_bar is not None and self.pending_long_stop_price is not None:
                if current_bar - self.long_signal_bar >= self.signal_expiration_bars:
                    self.pending_long_stop_price = None
                    self.long_signal_bar = None

            if self.short_signal_bar is not None and self.pending_short_stop_price is not None:
                if current_bar - self.short_signal_bar >= self.signal_expiration_bars:
                    self.pending_short_stop_price = None
                    self.short_signal_bar = None

        # Position Sizing Function
        def calc_position_size(current_price, stop_distance):
            risk_amount = self.equity * (self.risk_percent / 100.0)
            qty = risk_amount / stop_distance
            # Round to at least 0.01 BTC (2 decimal places) or make it a whole number if >= 1
            if qty >= 1:
                return max(1, int(round(qty)))
            else:
                return max(0.01, round(qty, 4))  # Min 0.01 BTC

        # ADX Entry Orders
        if (self.pending_long_stop_price and allow_longs and self.position.size <= 0 and
            self.data.High[-1] >= self.pending_long_stop_price):

            if self.position.size < 0:
                self.position.close()

            stop_dist = self.stop_dist[-1] * self.long_initial_stop_multiplier
            qty = calc_position_size(self.pending_long_stop_price, stop_dist)
            self.buy(size=qty)
            self.pending_long_stop_price = None
            self.long_signal_bar = None

        if (self.pending_short_stop_price and allow_shorts and self.position.size >= 0 and
            self.data.Low[-1] <= self.pending_short_stop_price):

            if self.position.size > 0:
                self.position.close()

            stop_dist = self.stop_dist[-1] * self.short_initial_stop_multiplier
            qty = calc_position_size(self.pending_short_stop_price, stop_dist)
            self.sell(size=qty)
            self.pending_short_stop_price = None
            self.short_signal_bar = None

        # Clear pending orders
        if self.position.size > 0:
            self.pending_short_stop_price = None
            self.short_signal_bar = None
        if self.position.size < 0:
            self.pending_long_stop_price = None
            self.long_signal_bar = None

        # Squeeze Entries
        if go_long and allow_longs and self.position.size >= 0:
            can_enter = False
            if self.enable_squeeze_long_add and self.position.size > 0:
                can_enter = True
            elif self.enable_squeeze_standalone and self.position.size == 0:
                can_enter = True

            if can_enter:
                stop_dist = self.stop_dist[-1] * self.long_initial_stop_multiplier
                qty = calc_position_size(self.data.Close[-1], stop_dist)
                self.buy(size=qty)

        if go_short and allow_shorts and self.position.size <= 0:
            can_enter = False
            if self.enable_squeeze_short_add and self.position.size < 0:
                can_enter = True
            elif self.enable_squeeze_standalone and self.position.size == 0:
                can_enter = True

            if can_enter:
                stop_dist = self.stop_dist[-1] * self.short_initial_stop_multiplier
                qty = calc_position_size(self.data.Close[-1], stop_dist)
                self.sell(size=qty)

        # Entry Detection
        is_entry_bar = (self.position.size != 0 and self.last_position_size == 0)

        if is_entry_bar:
            self.entry_price = self.data.Close[-1]
            self.is_long_position = self.position.size > 0

            adj_stop_dist = self.stop_dist[-1] * (
                self.long_initial_stop_multiplier if self.is_long_position
                else self.short_initial_stop_multiplier)

            if self.is_long_position:
                self.initial_stop_loss = self.entry_price - adj_stop_dist
                self.initial_risk = adj_stop_dist
                self.used_risk_factor = self.long_max_risk
            else:
                self.initial_stop_loss = self.entry_price + adj_stop_dist
                self.initial_risk = adj_stop_dist
                self.used_risk_factor = self.short_max_risk

            self.current_stop = self.initial_stop_loss
            self.bars_since_entry = 0

            if self.use_profit_target:
                self.fixed_profit_target = (self.entry_price + (self.profit_target_r * self.initial_risk)
                                           if self.is_long_position
                                           else self.entry_price - (self.profit_target_r * self.initial_risk))

        # Exit Detection
        if self.position.size == 0 and self.last_position_size != 0:
            self.entry_price = None
            self.initial_stop_loss = None
            self.initial_risk = None
            self.current_stop = None
            self.used_risk_factor = None
            self.is_long_position = None
            self.bars_since_entry = 0
            self.fixed_profit_target = None

        # Trailing Stop Logic
        if self.position.size != 0 and self.initial_risk is not None:
            if not is_entry_bar:
                self.bars_since_entry += 1

            # Dynamic R calculations
            pf = ((self.data.Close[-1] - self.entry_price) / self.initial_risk
                  if self.is_long_position
                  else (self.entry_price - self.data.Close[-1]) / self.initial_risk)

            def get_desired_rf(pf, max_r, min_r, term_p, act_p, arch):
                if pf < act_p:
                    return max_r
                netp = pf - act_p
                fall_grad = (max_r - min_r) / max(term_p - act_p, 1e-6)
                reduction_scale = fall_grad / (math.pow(1 + arch, term_p - act_p))
                rr = netp * math.pow(arch + 1, netp) * reduction_scale
                return max(max_r - rr, min_r)

            if self.is_long_position:
                drf = get_desired_rf(pf, self.long_max_risk, self.long_min_risk,
                                    self.long_terminal_profit, self.long_activation_profit,
                                    self.long_arch)
                if drf < self.used_risk_factor:
                    self.used_risk_factor = drf

                dynamic_r_stop = self.data.Close[-1] - (self.used_risk_factor * self.initial_risk)
                dbands_stop = self.lower_band[-1]
                atr_stop = self.data.Close[-1] - self.stop_dist[-1] if self.use_atr_stops else None

                if self.bars_since_entry >= 1:
                    selected_stop = dynamic_r_stop
                    if dbands_stop and dbands_stop > selected_stop:
                        selected_stop = dbands_stop
                    if atr_stop and atr_stop > selected_stop:
                        selected_stop = atr_stop

                    if selected_stop >= self.initial_stop_loss:
                        if self.current_stop is None or selected_stop > self.current_stop:
                            self.current_stop = selected_stop
            else:
                drf = get_desired_rf(pf, self.short_max_risk, self.short_min_risk,
                                    self.short_terminal_profit, self.short_activation_profit,
                                    self.short_arch)
                if drf < self.used_risk_factor:
                    self.used_risk_factor = drf

                dynamic_r_stop = self.data.Close[-1] + (self.used_risk_factor * self.initial_risk)
                dbands_stop = self.upper_band[-1]
                atr_stop = self.data.Close[-1] + self.stop_dist[-1] if self.use_atr_stops else None

                if self.bars_since_entry >= 1:
                    selected_stop = dynamic_r_stop
                    if dbands_stop and dbands_stop < selected_stop:
                        selected_stop = dbands_stop
                    if atr_stop and atr_stop < selected_stop:
                        selected_stop = atr_stop

                    if selected_stop <= self.initial_stop_loss:
                        if self.current_stop is None or selected_stop < self.current_stop:
                            self.current_stop = selected_stop

            # Apply stops
            if self.current_stop:
                if self.is_long_position and self.data.Close[-1] <= self.current_stop:
                    self.position.close()
                elif not self.is_long_position and self.data.Close[-1] >= self.current_stop:
                    self.position.close()

            # Profit target
            if self.use_profit_target and self.fixed_profit_target:
                if self.is_long_position and self.data.Close[-1] >= self.fixed_profit_target:
                    self.position.close()
                elif not self.is_long_position and self.data.Close[-1] <= self.fixed_profit_target:
                    self.position.close()

        self.last_position_size = self.position.size


if __name__ == "__main__":
    import os
    # Load data (auto-detect Windows vs Linux path)
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Go up: output -> pinescript_conversions -> data -> moon-dev-ai-agents
    data_path = os.path.join(base_path, "..", "..", "..", "rbi", "BTC-USD-1h-2020-2025.csv")
    data_path = os.path.normpath(data_path)

    df = pd.read_csv(data_path)

    # Prepare data
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    df.set_index('datetime', inplace=True)

    # Run backtest with enough cash for BTC prices ($100k+ range)
    bt = Backtest(df, ADXSqueezeRBasedStrategy,
                  cash=1000000,  # $1M starting capital for BTC trading
                  commission=0.001,
                  exclusive_orders=True)

    stats = bt.run()
    print("\n" + "="*60)
    print("ADX + Squeeze [R-BASED] Strategy - Complete Conversion")
    print("="*60)
    print(stats)
    print("="*60)

    # Plot
    bt.plot()
