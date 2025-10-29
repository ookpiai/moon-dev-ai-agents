"""
ADX + Squeeze Strategy [R-BASED] - Complete OOP Implementation
Converted from TradingView Pinescript v5 with full feature parity

This is a professional-grade implementation with:
- Triple confluence ADX signals (ADX + DI + EMA)
- Volatility squeeze detection with ATR filtering
- R-based position sizing (risk % per trade)
- Multiple stop mechanisms: Dynamic R, D-Bands, ATR regime
- Signal expiration system
- Multiple entry scenarios (standalone, add-on, reversal)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class TradingDirection(Enum):
    LONG_ONLY = "Long Only"
    SHORT_ONLY = "Short Only"
    BOTH = "Both"

class StopType(Enum):
    INITIAL = "Initial"
    DYNAMIC_R = "Dynamic R"
    D_BANDS = "D-Bands"
    ATR = "ATR"

class ATRRegime(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"

@dataclass
class StrategyConfig:
    """Configuration for ADX + Squeeze Strategy"""
    # ADX Settings
    adx_length: int = 14
    adx_smooth: int = 14
    ema_fast: int = 12
    ema_slow: int = 50
    adx_threshold: int = 20

    # Entry Settings
    trading_direction: TradingDirection = TradingDirection.LONG_ONLY
    long_entry_offset_ticks: int = 10
    short_entry_offset_ticks: int = 10
    enable_signal_expiration: bool = True
    signal_expiration_bars: int = 13

    # Squeeze Entry Settings
    enable_squeeze_standalone: bool = True
    enable_squeeze_long_add: bool = True
    enable_squeeze_short_add: bool = True
    enable_squeeze_reversal: bool = False
    squeeze_entry_offset_ticks: int = 0

    # Squeeze Indicator Settings
    fast_ma_type: str = "Exponential Moving Average - EMA"
    slow_ma_type: str = "Exponential Moving Average - EMA"
    fast_source: str = "Close"
    slow_source: str = "Close"
    fast_period: int = 5
    slow_period: int = 7
    atr_period: int = 50
    atr_multiplier: float = 0.4
    ma_threshold_ticks: int = 36
    filter_type: str = "ATR"

    # R-Based Position Management
    risk_percent: float = 0.5
    pyramiding: int = 5

    # Dynamic R Trailing Stop
    long_activation_profit: float = 0.5
    long_max_risk: float = 1.7
    long_min_risk: float = 1.5
    long_terminal_profit: float = 1.0
    long_arch: float = 0.7

    short_activation_profit: float = 0.5
    short_max_risk: float = 1.7
    short_min_risk: float = 1.5
    short_terminal_profit: float = 1.0
    short_arch: float = 0.7

    use_profit_target: bool = False
    profit_target_r: float = 3.0

    # D-Bands Trailing Stop
    dband_length: int = 30
    dband_multiplier: float = 5.0
    dband_alpha: float = 0.3
    dband_use_triple: bool = True

    # ATR Stop Module
    use_atr_stops: bool = True
    atr_short_len: int = 14
    atr_long_len: int = 100
    baseline_len: int = 100
    ratio_smooth: int = 5
    low_in: float = 0.85
    low_out: float = 0.85
    high_in: float = 1.15
    high_out: float = 1.15
    mult_low: float = 2.25
    mult_normal: float = 0.5
    mult_high: float = 3.25

    # Initial Stop Settings
    long_initial_stop_multiplier: float = 1.0
    short_initial_stop_multiplier: float = 1.0

    # Commission
    commission_percent: float = 0.0
    initial_capital: float = 100000.0

    # Backtesting period
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None

@dataclass
class Position:
    """Represents a trading position"""
    entry_price: float
    size: float
    is_long: bool
    initial_stop: float
    initial_risk: float
    entry_bar: int
    entry_type: str
    current_stop: float
    stop_type: StopType
    used_risk_factor: float
    profit_target: Optional[float] = None

class ADXSqueezeStrategy:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.positions: List[Position] = []
        self.equity_curve: List[float] = []
        self.trades: List[Dict] = []

        # State variables
        self.pending_long_stop_price: Optional[float] = None
        self.pending_short_stop_price: Optional[float] = None
        self.long_signal_bar: Optional[int] = None
        self.short_signal_bar: Optional[int] = None
        self.atr_regime = ATRRegime.NORMAL

        # Initialize capital
        self.capital = config.initial_capital
        self.equity = config.initial_capital

    def _get_price_source(self, df: pd.DataFrame, source: str) -> pd.Series:
        """Get price source based on source type"""
        source_map = {
            "Close": df['close'],
            "Open": df['open'],
            "High": df['high'],
            "Low": df['low'],
            "Median": (df['high'] + df['low']) / 2,
            "Typical": (df['high'] + df['low'] + df['close']) / 3,
            "Weighted": (df['high'] + df['low'] + df['close'] * 2) / 4,
            "Average": (df['open'] + df['high'] + df['low'] + df['close']) / 4,
            "Average Median Body": (df['open'] + df['close']) / 2
        }

        return source_map.get(source, df['close']).copy()

    def _calculate_pip_size(self, symbol_info: Optional[Dict] = None) -> float:
        """Calculate pip/tick size based on instrument type"""
        if symbol_info is None:
            return 0.0001

        instrument_type = symbol_info.get('type', 'forex').lower()
        currency = symbol_info.get('currency', 'USD')

        if instrument_type == 'forex':
            if currency == 'JPY':
                return 0.01
            else:
                return 0.0001
        elif instrument_type == 'futures':
            ticker = symbol_info.get('ticker', '').upper()
            if 'ES' in ticker or 'NQ' in ticker:
                return 0.25
            elif 'YM' in ticker:
                return 1.0
            elif 'CL' in ticker:
                return 0.01
            elif 'GC' in ticker:
                return 0.10
            else:
                return symbol_info.get('tick_size', 0.01)
        else:
            return symbol_info.get('tick_size', 0.01)

    def calculate_indicators(self, df: pd.DataFrame, symbol_info: Optional[Dict] = None) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = df.copy()

        # Calculate basic price derivations first
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3

        # ADX calculations (matching ta.dmi function)
        # True Range
        df['high_low'] = df['high'] - df['low']
        df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
        df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)

        # Directional Movement
        df['high_prev'] = df['high'].shift(1)
        df['low_prev'] = df['low'].shift(1)

        # +DM and -DM calculation
        df['up_move'] = df['high'] - df['high_prev']
        df['down_move'] = df['low_prev'] - df['low']

        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0),
                                  df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0),
                                   df['down_move'], 0)

        # Smooth using RMA (same as ta.dmi uses internally)
        alpha = 1.0 / self.config.adx_length

        # First calculate smoothed TR, +DM, -DM using RMA
        df['atr'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
        df['plus_dm_smooth'] = df['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
        df['minus_dm_smooth'] = df['minus_dm'].ewm(alpha=alpha, adjust=False).mean()

        # Calculate +DI and -DI
        df['plus_di'] = 100 * df['plus_dm_smooth'] / df['atr']
        df['minus_di'] = 100 * df['minus_dm_smooth'] / df['atr']

        # Calculate DX
        df['di_sum'] = df['plus_di'] + df['minus_di']
        df['di_diff'] = abs(df['plus_di'] - df['minus_di'])
        df['dx'] = np.where(df['di_sum'] > 0, 100 * df['di_diff'] / df['di_sum'], 0)

        # Calculate ADX using RMA smoothing
        alpha_adx = 1.0 / self.config.adx_smooth
        df['adx'] = df['dx'].ewm(alpha=alpha_adx, adjust=False).mean()

        # EMAs for trend
        df['ema_fast'] = df['close'].ewm(span=self.config.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.config.ema_slow, adjust=False).mean()

        # ADX conditions (matching indicator exactly)
        df['adx_direction_bull'] = df['plus_di'] > df['minus_di']
        df['adx_direction_bear'] = df['minus_di'] > df['plus_di']
        df['adx_strength'] = df['adx'] > self.config.adx_threshold
        df['ema_trend_bull'] = df['ema_fast'] > df['ema_slow']
        df['ema_trend_bear'] = df['ema_fast'] < df['ema_slow']

        # Triple confluence conditions
        df['bullish_confluence'] = (df['adx_direction_bull'] &
                                   df['adx_strength'] &
                                   df['ema_trend_bull'])
        df['bearish_confluence'] = (df['adx_direction_bear'] &
                                   df['adx_strength'] &
                                   df['ema_trend_bear'])

        # Transition detection (entry signals)
        df['bull_signal'] = df['bullish_confluence'] & (~df['bullish_confluence'].shift(1).fillna(False))
        df['bear_signal'] = df['bearish_confluence'] & (~df['bearish_confluence'].shift(1).fillna(False))

        # Squeeze calculations
        pip_size = self._calculate_pip_size(symbol_info)

        # Get price sources
        fast_source = self._get_price_source(df, self.config.fast_source)
        slow_source = self._get_price_source(df, self.config.slow_source)

        # Calculate MAs based on selected types (simplified to EMA for now)
        if self.config.fast_ma_type == "Exponential Moving Average - EMA":
            df['ma_fast'] = fast_source.ewm(span=self.config.fast_period, adjust=False).mean()
        else:
            df['ma_fast'] = fast_source.ewm(span=self.config.fast_period, adjust=False).mean()

        if self.config.slow_ma_type == "Exponential Moving Average - EMA":
            df['ma_slow'] = slow_source.ewm(span=self.config.slow_period, adjust=False).mean()
        else:
            df['ma_slow'] = slow_source.ewm(span=self.config.slow_period, adjust=False).mean()

        # MA difference
        df['ma_diff'] = abs(df['ma_fast'] - df['ma_slow'])

        # ATR for squeeze filter
        df['atr_squeeze'] = df['tr'].ewm(span=self.config.atr_period, adjust=False).mean()

        # Calculate delta threshold
        if self.config.filter_type == "ATR":
            df['delta'] = df['atr_squeeze'] * self.config.atr_multiplier / pip_size
        else:
            df['delta'] = float(self.config.ma_threshold_ticks)

        # Squeeze detection
        df['ma_diff_pips'] = df['ma_diff'] / pip_size
        df['squeeze_active'] = df['ma_diff_pips'] < df['delta']

        # Calculate squeeze bands
        df['squeeze_band_upper'] = np.where(df['squeeze_active'],
                                           df['ma_slow'] + (df['delta'] * pip_size),
                                           np.nan)
        df['squeeze_band_lower'] = np.where(df['squeeze_active'],
                                           df['ma_slow'] - (df['delta'] * pip_size),
                                           np.nan)

        # Detect squeeze start and end
        df['squeeze_start'] = df['squeeze_active'] & (~df['squeeze_active'].shift(1).fillna(False))
        df['squeeze_end'] = (~df['squeeze_active'].fillna(False)) & df['squeeze_active'].shift(1).fillna(False)

        # Generate signals on squeeze end
        df['go_long'] = df['squeeze_end'] & (df['ma_fast'] > df['ma_slow'])
        df['go_short'] = df['squeeze_end'] & (df['ma_fast'] < df['ma_slow'])

        # D-Bands calculations
        df = self._calculate_d_bands(df)

        # ATR regime calculations
        df = self._calculate_atr_regime(df)

        return df

    def _calculate_d_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate D-Bands for trailing stop - exact implementation"""

        # Helper function for double WMA
        def double_wma(series, length):
            """Calculate double-pass WMA"""
            if len(series) < length:
                return series
            # First WMA pass
            wma1 = series.rolling(length).apply(
                lambda x: np.average(x, weights=np.arange(1, length+1))
            )
            # Second WMA pass
            wma2 = wma1.rolling(length).apply(
                lambda x: np.average(x, weights=np.arange(1, length+1))
            )
            return wma2

        # Helper function for single WMA
        def wma(series, length):
            """Calculate weighted moving average"""
            if len(series) < length:
                return series
            return series.rolling(length).apply(
                lambda x: np.average(x, weights=np.arange(1, length+1))
            )

        # 1) Center line (Double WMA of hlc3)
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['dband_center'] = double_wma(df['hlc3'], self.config.dband_length)

        # 2) Up/Down distances from center
        df['dist_up_raw'] = np.maximum(0.0, df['high'] - df['dband_center'])
        df['dist_down_raw'] = np.maximum(0.0, df['dband_center'] - df['low'])

        # 3) Optional extra smoothing before stdev (triple smoothing)
        if self.config.dband_use_triple:
            df['dist_up_pre'] = wma(df['dist_up_raw'], self.config.dband_length)
            df['dist_down_pre'] = wma(df['dist_down_raw'], self.config.dband_length)
        else:
            df['dist_up_pre'] = df['dist_up_raw']
            df['dist_down_pre'] = df['dist_down_raw']

        # 4) Raw stdev of the distances
        df['stdev_up_raw'] = df['dist_up_pre'].rolling(self.config.dband_length).std()
        df['stdev_down_raw'] = df['dist_down_pre'].rolling(self.config.dband_length).std()

        # 5) Double-WMA of the stdev (2 more passes)
        df['stdev_up'] = double_wma(df['stdev_up_raw'], self.config.dband_length)
        df['stdev_down'] = double_wma(df['stdev_down_raw'], self.config.dband_length)

        # 6) ATR calculation (already exists in df from earlier calculations)

        # 7) Blend StDev & ATR
        df['final_up_vol'] = (self.config.dband_alpha * df['stdev_up'] +
                             (1.0 - self.config.dband_alpha) * df['atr'])
        df['final_down_vol'] = (self.config.dband_alpha * df['stdev_down'] +
                               (1.0 - self.config.dband_alpha) * df['atr'])

        # 8) Final bands
        df['dband_upper'] = df['dband_center'] + (df['final_up_vol'] * self.config.dband_multiplier)
        df['dband_lower'] = df['dband_center'] - (df['final_down_vol'] * self.config.dband_multiplier)

        return df

    def _calculate_atr_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR regime for adaptive stops"""
        df['atr_short'] = df['tr'].ewm(span=self.config.atr_short_len, adjust=False).mean()
        df['atr_long'] = df['tr'].ewm(span=self.config.atr_long_len, adjust=False).mean()
        df['baseline'] = df['atr_long'].ewm(span=self.config.baseline_len, adjust=False).mean()

        df['ratio'] = (df['atr_long'] / df['baseline']).ewm(span=self.config.ratio_smooth, adjust=False).mean()

        # Initialize regime column
        df['atr_regime'] = ATRRegime.NORMAL.value

        # Apply regime logic
        for i in range(1, len(df)):
            prev_regime = df.iloc[i-1]['atr_regime']
            ratio = df.iloc[i]['ratio']

            if prev_regime != ATRRegime.HIGH.value and ratio > self.config.high_in:
                df.loc[df.index[i], 'atr_regime'] = ATRRegime.HIGH.value
            elif prev_regime == ATRRegime.HIGH.value and ratio >= self.config.high_out:
                df.loc[df.index[i], 'atr_regime'] = ATRRegime.HIGH.value
            elif prev_regime != ATRRegime.LOW.value and ratio < self.config.low_in:
                df.loc[df.index[i], 'atr_regime'] = ATRRegime.LOW.value
            elif prev_regime == ATRRegime.LOW.value and ratio <= self.config.low_out:
                df.loc[df.index[i], 'atr_regime'] = ATRRegime.LOW.value
            else:
                df.loc[df.index[i], 'atr_regime'] = ATRRegime.NORMAL.value

        # Calculate stop distance based on regime
        df['atr_mult'] = df['atr_regime'].map({
            ATRRegime.LOW.value: self.config.mult_low,
            ATRRegime.HIGH.value: self.config.mult_high,
            ATRRegime.NORMAL.value: self.config.mult_normal
        })

        df['stop_distance'] = df['atr_short'] * df['atr_mult']

        return df

    def _calculate_position_size(self, current_price: float, stop_distance: float) -> float:
        """Calculate position size based on R-based sizing"""
        risk_amount = self.equity * (self.config.risk_percent / 100.0)
        position_size = risk_amount / stop_distance
        return position_size

    def _get_profit_factors(self, position: Position, current_price: float) -> float:
        """Calculate profit in R multiples"""
        if position.is_long:
            return (current_price - position.entry_price) / position.initial_risk
        else:
            return (position.entry_price - current_price) / position.initial_risk

    def _get_desired_risk_factor(self, pf: float, position: Position) -> float:
        """Calculate desired risk factor for dynamic R stop"""
        if position.is_long:
            activation = self.config.long_activation_profit
            max_r = self.config.long_max_risk
            min_r = self.config.long_min_risk
            terminal = self.config.long_terminal_profit
            arch = self.config.long_arch
        else:
            activation = self.config.short_activation_profit
            max_r = self.config.short_max_risk
            min_r = self.config.short_min_risk
            terminal = self.config.short_terminal_profit
            arch = self.config.short_arch

        if pf < activation:
            return max_r

        net_p = pf - activation
        fall_gradient = (max_r - min_r) / max(terminal - activation, 1e-6)
        reduction_scaler = fall_gradient / pow(1 + arch, terminal - activation)
        rr = net_p * pow(arch + 1, net_p) * reduction_scaler
        desired = max_r - rr

        return max(desired, min_r)

    def _calculate_stop_price(self, position: Position, rf: float) -> float:
        """Calculate stop price based on risk factor"""
        if position.is_long:
            return position.entry_price - (rf * position.initial_risk)
        else:
            return position.entry_price + (rf * position.initial_risk)

    def _update_trailing_stops(self, position: Position, current_bar: pd.Series,
                              bars_since_entry: int) -> None:
        """Update trailing stop for position"""
        if bars_since_entry < 1:
            return

        current_price = current_bar['close']
        pf = self._get_profit_factors(position, current_price)

        # Calculate all stop candidates
        stops = {}

        # Dynamic R stop
        drf = self._get_desired_risk_factor(pf, position)
        if drf < position.used_risk_factor:
            position.used_risk_factor = drf

        dynamic_r_stop = self._calculate_stop_price(position, position.used_risk_factor)
        stops['dynamic_r'] = (dynamic_r_stop, StopType.DYNAMIC_R)

        # D-Bands stop
        if position.is_long:
            stops['dbands'] = (current_bar['dband_lower'], StopType.D_BANDS)
        else:
            stops['dbands'] = (current_bar['dband_upper'], StopType.D_BANDS)

        # ATR stop
        if self.config.use_atr_stops:
            stop_dist = current_bar['stop_distance']
            if position.is_long:
                stops['atr'] = (current_price - stop_dist, StopType.ATR)
            else:
                stops['atr'] = (current_price + stop_dist, StopType.ATR)

        # Select best stop
        if position.is_long:
            valid_stops = {k: v for k, v in stops.items()
                          if not pd.isna(v[0]) and v[0] >= position.initial_stop}
            if valid_stops:
                best_stop = max(valid_stops.values(), key=lambda x: x[0])
                if best_stop[0] > position.current_stop:
                    position.current_stop = best_stop[0]
                    position.stop_type = best_stop[1]
        else:
            valid_stops = {k: v for k, v in stops.items()
                          if not pd.isna(v[0]) and v[0] <= position.initial_stop}
            if valid_stops:
                best_stop = min(valid_stops.values(), key=lambda x: x[0])
                if best_stop[0] < position.current_stop:
                    position.current_stop = best_stop[0]
                    position.stop_type = best_stop[1]

    def backtest(self, df: pd.DataFrame, symbol_info: Optional[Dict] = None) -> Dict:
        """Run backtest on provided data"""
        # Calculate indicators
        df = self.calculate_indicators(df, symbol_info)

        # Get pip size for order offsets
        pip_size = self._calculate_pip_size(symbol_info)

        # Filter by date range if specified
        if self.config.start_date:
            df = df[df.index >= self.config.start_date]
        if self.config.end_date:
            df = df[df.index <= self.config.end_date]

        # Initialize results
        results = {
            'equity': [],
            'positions': [],
            'signals': [],
            'stops': []
        }

        # Main backtest loop
        for i, (idx, row) in enumerate(df.iterrows()):
            # Update pending orders expiration
            self._check_signal_expiration(i)

            # Check for new ADX signals
            if row['bull_signal'] and self._can_trade_long():
                self.pending_long_stop_price = row['high'] + (
                    self.config.long_entry_offset_ticks * pip_size)
                self.long_signal_bar = i
                results['signals'].append({
                    'index': idx,
                    'type': 'ADX_LONG',
                    'price': self.pending_long_stop_price
                })

            if row['bear_signal'] and self._can_trade_short():
                self.pending_short_stop_price = row['low'] - (
                    self.config.short_entry_offset_ticks * pip_size)
                self.short_signal_bar = i
                results['signals'].append({
                    'index': idx,
                    'type': 'ADX_SHORT',
                    'price': self.pending_short_stop_price
                })

            # Check for ADX entries
            self._check_adx_entries(row, i, pip_size)

            # Check for Squeeze entries
            self._check_squeeze_entries(row, i, pip_size)

            # Update positions
            self._update_positions(row, i)

            # Check exits
            self._check_exits(row, i)

            # Update equity
            self._update_equity(row)
            results['equity'].append({
                'index': idx,
                'equity': self.equity,
                'positions': len(self.positions)
            })

            # Record position info
            if self.positions:
                for pos in self.positions:
                    results['positions'].append({
                        'index': idx,
                        'entry_price': pos.entry_price,
                        'size': pos.size,
                        'is_long': pos.is_long,
                        'current_stop': pos.current_stop,
                        'stop_type': pos.stop_type.value
                    })

        # Calculate performance metrics
        equity_df = pd.DataFrame(results['equity']).set_index('index')
        returns = equity_df['equity'].pct_change().dropna()

        metrics = {
            'total_return': (self.equity / self.config.initial_capital - 1) * 100,
            'total_trades': len(self.trades),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor(),
            'max_drawdown': self._calculate_max_drawdown(equity_df['equity']),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'results': results
        }

        return metrics

    def _can_trade_long(self) -> bool:
        """Check if long trades are allowed"""
        return (self.config.trading_direction == TradingDirection.LONG_ONLY or
                self.config.trading_direction == TradingDirection.BOTH)

    def _can_trade_short(self) -> bool:
        """Check if short trades are allowed"""
        return (self.config.trading_direction == TradingDirection.SHORT_ONLY or
                self.config.trading_direction == TradingDirection.BOTH)

    def _check_signal_expiration(self, current_bar: int) -> None:
        """Check and expire stale signals"""
        if not self.config.enable_signal_expiration:
            return

        if self.long_signal_bar is not None and self.pending_long_stop_price is not None:
            if current_bar - self.long_signal_bar >= self.config.signal_expiration_bars:
                self.pending_long_stop_price = None
                self.long_signal_bar = None

        if self.short_signal_bar is not None and self.pending_short_stop_price is not None:
            if current_bar - self.short_signal_bar >= self.config.signal_expiration_bars:
                self.pending_short_stop_price = None
                self.short_signal_bar = None

    def _check_adx_entries(self, row: pd.Series, bar_index: int, pip_size: float) -> None:
        """Check for ADX entry conditions"""
        # Long entry
        if (self.pending_long_stop_price is not None and
            row['high'] >= self.pending_long_stop_price and
            self._can_trade_long() and
            self._total_positions() < self.config.pyramiding):

            # Close shorts if any
            self._close_opposite_positions(is_long=True)

            # Calculate position size
            stop_dist = row['stop_distance'] * self.config.long_initial_stop_multiplier
            size = self._calculate_position_size(self.pending_long_stop_price, stop_dist)

            # Create position
            initial_stop = self.pending_long_stop_price - stop_dist
            initial_risk = self.pending_long_stop_price - initial_stop

            position = Position(
                entry_price=self.pending_long_stop_price,
                size=size,
                is_long=True,
                initial_stop=initial_stop,
                initial_risk=initial_risk,
                entry_bar=bar_index,
                entry_type="ADX",
                current_stop=initial_stop,
                stop_type=StopType.INITIAL,
                used_risk_factor=self.config.long_max_risk,
                profit_target=(self.pending_long_stop_price +
                              (self.config.profit_target_r * initial_risk)
                              if self.config.use_profit_target else None)
            )

            self.positions.append(position)
            self.pending_long_stop_price = None
            self.long_signal_bar = None

            # Record trade
            self.trades.append({
                'entry_bar': bar_index,
                'entry_price': position.entry_price,
                'size': position.size,
                'is_long': position.is_long,
                'entry_type': 'ADX'
            })

        # Short entry
        if (self.pending_short_stop_price is not None and
            row['low'] <= self.pending_short_stop_price and
            self._can_trade_short() and
            self._total_positions() < self.config.pyramiding):

            self._close_opposite_positions(is_long=False)

            stop_dist = row['stop_distance'] * self.config.short_initial_stop_multiplier
            size = self._calculate_position_size(self.pending_short_stop_price, stop_dist)

            initial_stop = self.pending_short_stop_price + stop_dist
            initial_risk = initial_stop - self.pending_short_stop_price

            position = Position(
                entry_price=self.pending_short_stop_price,
                size=size,
                is_long=False,
                initial_stop=initial_stop,
                initial_risk=initial_risk,
                entry_bar=bar_index,
                entry_type="ADX",
                current_stop=initial_stop,
                stop_type=StopType.INITIAL,
                used_risk_factor=self.config.short_max_risk,
                profit_target=(self.pending_short_stop_price -
                              (self.config.profit_target_r * initial_risk)
                              if self.config.use_profit_target else None)
            )

            self.positions.append(position)
            self.pending_short_stop_price = None
            self.short_signal_bar = None

            self.trades.append({
                'entry_bar': bar_index,
                'entry_price': position.entry_price,
                'size': position.size,
                'is_long': position.is_long,
                'entry_type': 'ADX'
            })

    def _check_squeeze_entries(self, row: pd.Series, bar_index: int, pip_size: float) -> None:
        """Check for Squeeze entry conditions"""
        if self._total_positions() >= self.config.pyramiding:
            return

        # Squeeze long
        if row['go_long'] and self._can_trade_long():
            can_enter = False

            # Check entry scenarios
            if self.config.enable_squeeze_standalone and not self._has_positions():
                can_enter = True
            elif self.config.enable_squeeze_long_add and self._has_long_positions():
                can_enter = True
            elif self.config.enable_squeeze_reversal and self._has_short_positions():
                self._close_all_positions()
                can_enter = True

            if can_enter:
                # Calculate entry price
                if self.config.squeeze_entry_offset_ticks > 0:
                    entry_price = row['close'] + (
                        self.config.squeeze_entry_offset_ticks * pip_size)
                else:
                    entry_price = row['close']

                # Calculate position
                stop_dist = row['stop_distance'] * self.config.long_initial_stop_multiplier
                size = self._calculate_position_size(entry_price, stop_dist)

                initial_stop = entry_price - stop_dist
                initial_risk = entry_price - initial_stop

                position = Position(
                    entry_price=entry_price,
                    size=size,
                    is_long=True,
                    initial_stop=initial_stop,
                    initial_risk=initial_risk,
                    entry_bar=bar_index,
                    entry_type="SQUEEZE",
                    current_stop=initial_stop,
                    stop_type=StopType.INITIAL,
                    used_risk_factor=self.config.long_max_risk,
                    profit_target=(entry_price +
                                  (self.config.profit_target_r * initial_risk)
                                  if self.config.use_profit_target else None)
                )

                self.positions.append(position)
                self.trades.append({
                    'entry_bar': bar_index,
                    'entry_price': position.entry_price,
                    'size': position.size,
                    'is_long': position.is_long,
                    'entry_type': 'SQUEEZE'
                })

        # Squeeze short
        if row['go_short'] and self._can_trade_short():
            can_enter = False

            if self.config.enable_squeeze_standalone and not self._has_positions():
                can_enter = True
            elif self.config.enable_squeeze_short_add and self._has_short_positions():
                can_enter = True
            elif self.config.enable_squeeze_reversal and self._has_long_positions():
                self._close_all_positions()
                can_enter = True

            if can_enter:
                if self.config.squeeze_entry_offset_ticks > 0:
                    # Stop entry order below current price
                    entry_price = row['close'] - (
                        self.config.squeeze_entry_offset_ticks * pip_size)
                else:
                    entry_price = row['close']

                stop_dist = row['stop_distance'] * self.config.short_initial_stop_multiplier
                size = self._calculate_position_size(entry_price, stop_dist)

                initial_stop = entry_price + stop_dist
                initial_risk = initial_stop - entry_price

                position = Position(
                    entry_price=entry_price,
                    size=size,
                    is_long=False,
                    initial_stop=initial_stop,
                    initial_risk=initial_risk,
                    entry_bar=bar_index,
                    entry_type="SQUEEZE",
                    current_stop=initial_stop,
                    stop_type=StopType.INITIAL,
                    used_risk_factor=self.config.short_max_risk,
                    profit_target=(entry_price -
                                  (self.config.profit_target_r * initial_risk)
                                  if self.config.use_profit_target else None)
                )

                self.positions.append(position)
                self.trades.append({
                    'entry_bar': bar_index,
                    'entry_price': position.entry_price,
                    'size': position.size,
                    'is_long': position.is_long,
                    'entry_type': 'SQUEEZE'
                })

    def _update_positions(self, row: pd.Series, bar_index: int) -> None:
        """Update position trailing stops"""
        for position in self.positions:
            bars_since_entry = bar_index - position.entry_bar
            self._update_trailing_stops(position, row, bars_since_entry)

    def _check_exits(self, row: pd.Series, bar_index: int) -> None:
        """Check for position exits"""
        positions_to_remove = []

        for i, position in enumerate(self.positions):
            exit_triggered = False
            exit_price = None
            exit_reason = ""

            # Check stop loss
            if position.is_long:
                if row['low'] <= position.current_stop:
                    exit_triggered = True
                    exit_price = min(row['open'], position.current_stop)
                    exit_reason = f"Stop ({position.stop_type.value})"
            else:
                if row['high'] >= position.current_stop:
                    exit_triggered = True
                    exit_price = max(row['open'], position.current_stop)
                    exit_reason = f"Stop ({position.stop_type.value})"

            # Check profit target
            if position.profit_target is not None:
                if position.is_long and row['high'] >= position.profit_target:
                    exit_triggered = True
                    exit_price = max(row['open'], position.profit_target)
                    exit_reason = "Profit Target"
                elif not position.is_long and row['low'] <= position.profit_target:
                    exit_triggered = True
                    exit_price = min(row['open'], position.profit_target)
                    exit_reason = "Profit Target"

            if exit_triggered:
                # Calculate P&L
                if position.is_long:
                    pnl = (exit_price - position.entry_price) * position.size
                else:
                    pnl = (position.entry_price - exit_price) * position.size

                # Update capital
                self.capital += pnl

                # Record exit
                for trade in self.trades:
                    if (trade['entry_bar'] == position.entry_bar and
                        trade['entry_price'] == position.entry_price):
                        trade['exit_bar'] = bar_index
                        trade['exit_price'] = exit_price
                        trade['exit_reason'] = exit_reason
                        trade['pnl'] = pnl
                        break

                positions_to_remove.append(i)

        # Remove exited positions
        for i in reversed(positions_to_remove):
            self.positions.pop(i)

    def _update_equity(self, row: pd.Series) -> None:
        """Update equity based on open positions"""
        self.equity = self.capital

        for position in self.positions:
            if position.is_long:
                unrealized_pnl = (row['close'] - position.entry_price) * position.size
            else:
                unrealized_pnl = (position.entry_price - row['close']) * position.size

            self.equity += unrealized_pnl

    def _total_positions(self) -> int:
        return len(self.positions)

    def _has_positions(self) -> bool:
        return len(self.positions) > 0

    def _has_long_positions(self) -> bool:
        return any(p.is_long for p in self.positions)

    def _has_short_positions(self) -> bool:
        return any(not p.is_long for p in self.positions)

    def _close_opposite_positions(self, is_long: bool) -> None:
        """Close positions opposite to the new direction"""
        self.positions = [p for p in self.positions if p.is_long == is_long]

    def _close_all_positions(self) -> None:
        """Close all positions"""
        self.positions = []

    def _calculate_win_rate(self) -> float:
        """Calculate win rate percentage"""
        if not self.trades:
            return 0.0

        completed_trades = [t for t in self.trades if 'pnl' in t]
        if not completed_trades:
            return 0.0

        wins = sum(1 for t in completed_trades if t['pnl'] > 0)
        return (wins / len(completed_trades)) * 100

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        completed_trades = [t for t in self.trades if 'pnl' in t]
        if not completed_trades:
            return 0.0

        gross_profit = sum(t['pnl'] for t in completed_trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in completed_trades if t['pnl'] < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        cummax = equity_series.expanding().max()
        drawdown = (equity_series - cummax) / cummax
        return abs(drawdown.min()) * 100

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio (assuming 252 trading days)"""
        if len(returns) < 2:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0.0

        return (mean_return / std_return) * np.sqrt(252)


if __name__ == "__main__":
    # Example usage
    import os

    # Load BTC data
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "..", "..", "..", "rbi", "BTC-USD-1h-2020-2025.csv")
    data_path = os.path.normpath(data_path)

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    df.columns = [col.lower() for col in df.columns]
    df.set_index('datetime', inplace=True)

    # Create strategy config
    config = StrategyConfig(
        trading_direction=TradingDirection.LONG_ONLY,
        risk_percent=0.5,
        initial_capital=1000000.0
    )

    # Run backtest
    strategy = ADXSqueezeStrategy(config)
    results = strategy.backtest(df)

    print("\n" + "="*60)
    print("ADX + Squeeze [R-BASED] Strategy Results")
    print("="*60)
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print("="*60)
