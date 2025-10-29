from backtesting import Backtest, Strategy
import pandas as pd
import pandas_ta as ta
import numpy as np

class ADXSqueezeStrategy(Strategy):
    # ADX Settings
    adx_length = 14
    adx_smooth = 14
    ema_fast = 12
    ema_slow = 50
    adx_threshold = 20

    # Entry Settings
    trading_direction = "Long Only"
    long_entry_offset = 10
    short_entry_offset = 10
    signal_expiration = 13
    
    # Squeeze Settings
    fast_period = 5
    slow_period = 7
    
    # Risk Management
    risk_percent = 0.5
    long_max_risk = 1.7
    short_max_risk = 1.7
    long_min_risk = 1.5
    short_min_risk = 1.5
    long_terminal_profit = 1.0
    short_terminal_profit = 1.0
    long_activation_profit = 0.5
    short_activation_profit = 0.5
    long_arch = 0.7
    short_arch = 0.7

    def init(self):
        # Calculate ADX indicators
        self.adx = self.I(ta.adx, high=self.data.High, low=self.data.Low, close=self.data.Close, length=self.adx_length)
        self.ema_fast = self.I(ta.ema, self.data.Close, length=self.ema_fast)
        self.ema_slow = self.I(ta.ema, self.data.Close, length=self.ema_slow)
        
        # Squeeze indicators
        self.fast_ma = self.I(ta.ema, self.data.Close, length=self.fast_period)
        self.slow_ma = self.I(ta.ema, self.data.Close, length=self.slow_period)

    def next(self):
        # ADX trend conditions
        adx_bull = (self.adx.pdi[-1] > self.adx.mdi[-1]) and (self.adx.adx[-1] > self.adx_threshold)
        adx_bear = (self.adx.mdi[-1] > self.adx.pdi[-1]) and (self.adx.adx[-1] > self.adx_threshold)
        ema_bull = self.ema_fast[-1] > self.ema_slow[-1]
        ema_bear = self.ema_fast[-1] < self.ema_slow[-1]

        # Squeeze conditions
        squeeze_long = self.fast_ma[-1] > self.slow_ma[-1]
        squeeze_short = self.fast_ma[-1] < self.slow_ma[-1]

        # Position sizing
        atr = ta.atr(self.data.High, self.data.Low, self.data.Close, length=14)[-1]
        stop_distance = atr * 2  # Simple ATR-based stop
        
        # Entry logic for longs
        if (self.trading_direction in ["Long Only", "Both"] and 
            adx_bull and ema_bull and squeeze_long and not self.position):
            risk_amount = self.equity * (self.risk_percent / 100)
            position_size = risk_amount / stop_distance
            self.buy(size=position_size, stop=self.data.Close[-1] - stop_distance)

        # Entry logic for shorts
        if (self.trading_direction in ["Short Only", "Both"] and 
            adx_bear and ema_bear and squeeze_short and not self.position):
            risk_amount = self.equity * (self.risk_percent / 100)
            position_size = risk_amount / stop_distance
            self.sell(size=position_size, stop=self.data.Close[-1] + stop_distance)

        # Dynamic trailing stop
        if self.position:
            # Simple trailing stop based on ATR
            if self.position.is_long:
                stop_price = self.data.Close[-1] - (stop_distance * self.long_max_risk)
                if stop_price > self.position.stop:
                    self.position.stop = stop_price
            else:
                stop_price = self.data.Close[-1] + (stop_distance * self.short_max_risk)
                if stop_price < self.position.stop:
                    self.position.stop = stop_price

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("path/to/your/data.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Run backtest
    bt = Backtest(df, ADXSqueezeStrategy, cash=10000, commission=0.001)
    stats = bt.run()
    print(stats)
    bt.plot()
