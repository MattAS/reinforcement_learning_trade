from bars import Bars
import enviornment
import numpy as np
from stream import StreamBar
import threading
from dataParser import Parser


class Observation:
    def __init__(self, ticker, env):
        self.bars_data = Bars(ticker)
        self.prices = self.bars_data.get_data()
        self.first_run = True
        self.stream = StreamBar(ticker)
        self.parser = Parser()
        self.called = 0
        self._offset = 0
        self.env = env

    def create_obs(self):
        state_obs = np.ndarray(shape=self.env._state.shape, dtype=np.float32)
        shift = 0  # counter of shift

        # Loop through bars and get price values for ohlc, position, and pnl
        for bar_idx in range(10):
            ofs = self._offset + bar_idx
            print(type(self.prices['open']))
            state_obs[shift] = self.prices['open'][ofs]
            shift += 1
            state_obs[shift] = self.prices['high'][ofs]
            shift += 1
            state_obs[shift] = self.prices['low'][ofs]
            shift += 1
            state_obs[shift] = self.prices['close'][ofs]
            shift += 1

            state_obs[shift] = float(self.env._state.have_position)
            shift += 1

            if not self.env._state.have_position:
                state_obs[shift] = 0.0
            else:
                state_obs[shift] = np.sign(
                    self.env._state.curr_close() - self.env._state.open_price)
            shift += 1

        self.first_run = False
        return state_obs
