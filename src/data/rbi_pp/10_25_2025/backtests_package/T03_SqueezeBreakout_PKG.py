import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# Load and prepare data
path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Ensure required columns with capital first letter
data.columns = [col.capitalize() if col in ['open', 'high', 'low', 'close', 'volume'] else col for col in data.columns]

class SqueezeBreakout(Strategy):
    bb_period = 20
    bb_std = 2.0
    kc_period = 20
    kc_mult = 1.5
    vol_short = 14
    vol_long = 28
    lookback_squeeze = 12
    risk_per_trade = 0.01
    rr_ratio = 2.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                                               nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)

        # ATR for Keltner and stops
        atr = self.I(talib.ATR, high, low, close, timeperiod=self.kc_period)

        # EMA for Keltner middle
        ema = self.I(talib.EMA, close, timeperiod=self.kc_period)

        # Keltner Channels
        def kc_upper_func(ema_val, atr_val):
            return ema_val + atr_val * self.kc_mult
        def kc_lower_func(ema_val, atr_val):
            return ema_val - atr_val * self.kc_mult
        self.kc_upper = self.I(kc_upper_func, ema, atr)
        self.kc_lower = self.I(kc_lower_func, ema, atr)

        # BB Width
        def bb_width_func(upper, lower):
            return upper - lower
        self.bb_width = self.I(bb_width_func, bb_upper, bb_lower)

        # Volume Oscillator (short SMA - long SMA)
        vol_sma_short = self.I(talib.SMA, volume, timeperiod=self.vol_short)
        vol_sma_long = self.I(talib.SMA, volume, timeperiod=self.vol_long)
        def vol_osc_func(short, long):
            return short - long
        self.vol_osc = self.I(vol_osc_func, vol_sma_short, vol_sma_long)

        # Squeeze condition: BB inside KC
        def squeeze_func(bb_u, bb_l, kc_u, kc_l):
            return (bb_u <= kc_u) & (bb_l >= kc_l)
        self.squeeze = self.I(squeeze_func, bb_upper, bb_lower, self.kc_upper, self.kc_lower)

        # Store indicators for access
        self.bb_upper = bb_upper
        self.bb_lower = bb_lower
        self.bb_middle = bb_middle
        self.atr = atr
        self.sma20 = self.I(talib.SMA, close, timeperiod=20)  # For bias

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.lookback_squeeze + self.kc_period:
            return

        current_price = self.data.Close[-1]
        current_squeeze = self.squeeze[-1]
        prev_squeeze = self.squeeze[-2] if len(self.squeeze) > 1 else False
        current_vol_osc = self.vol_osc[-1]
        current_bb_width = self.bb_width[-1]
        bb_widths = self.bb_width[-self.lookback_squeeze:]
        is_lowest_width = current_bb_width == np.min(bb_widths) if len(bb_widths) == self.lookback_squeeze else False

        # Check for squeeze end and low width
        squeeze_ended = prev_squeeze and not current_squeeze and is_lowest_width

        # No position
        if self.position.is_long == False and self.position.is_short == False:
            # Long entry
            if squeeze_ended and current_price > self.bb_upper[-1] and current_vol_osc > 0 and current_price > self.bb_middle[-1]:
                entry_price = current_price
                sl_price = self.bb_lower[-1]  # Beyond opposite BB
                risk_distance = entry_price - sl_price
                if risk_distance > 0:
                    equity = self._broker.getvalue()
                    risk_amount = self.risk_per_trade * equity
                    position_size = risk_amount / risk_distance
                    size = int(round(position_size))
                    tp_price = entry_price + (self.rr_ratio * risk_distance)
                    self.buy(size=size, sl=sl_price, tp=tp_price)
                    print(f"🌙 Moon Dev: Squeeze Breakout LONG entry at {entry_price:.2f}, size {size}, SL {sl_price:.2f}, TP {tp_price:.2f} 🚀✨")

            # Short entry
            elif squeeze_ended and current_price < self.bb_lower[-1] and current_vol_osc > 0 and current_price < self.bb_middle[-1]:
                entry_price = current_price
                sl_price = self.bb_upper[-1]  # Beyond opposite BB
                risk_distance = sl_price - entry_price
                if risk_distance > 0:
                    equity = self._broker.getvalue()
                    risk_amount = self.risk_per_trade * equity
                    position_size = risk_amount / risk_distance
                    size = int(round(position_size))
                    tp_price = entry_price - (self.rr_ratio * risk_distance)
                    self.sell(size=size, sl=sl_price, tp=tp_price)
                    print(f"🌙 Moon Dev: Squeeze Breakout SHORT entry at {entry_price:.2f}, size {size}, SL {sl_price:.2f}, TP {tp_price:.2f} 🚀✨")

        # Early exit if re-squeeze
        elif (self.position.is_long and current_squeeze) or (self.position.is_short and current_squeeze):
            self.position.close()
            print(f"🌙 Moon Dev: Early exit due to re-squeeze at {current_price:.2f} 😔")

# Run backtest
bt = Backtest(data, SqueezeBreakout, cash=1000000, commission=.002)
stats = bt.run()
print(stats)