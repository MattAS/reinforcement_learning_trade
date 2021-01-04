import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import gym.wrappers
import enum
import numpy as np
import random
from pathlib import Path

BATCH_SIZE = 32
BARS_COUNT = 10

EPS_START = 1.0
EPS_FINAL = 0.1
EPS_STEPS = 1000000

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000

LEARNING_RATE = 0.0001
STATES_TO_EVALUATE = 1000

DEFAULT_ENV_NAME = "StocksEnv-v0"
DEFAULT_BAR_COUNT = 10
DEFAULT_COMMISION = 0.0

EPS_DECAY = 0.99
TGT_NET_SYNC = 10


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    spec = EnvSpec("StocksEnv-v0")

    # Constructor: Initialize price, state, observation space, and action space
    def __init__(self, prices, bar_count=DEFAULT_BAR_COUNT, commision=DEFAULT_COMMISION):
        assert isinstance(prices, dict)
        self._prices = prices
        self._state = State(bar_count, commision)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.seed()

    # Take a step in the enviornment. return the next observation, the reward, done flag and other info

    def step(self, action_idx):
        action = Actions(action_idx)
        done, reward = self._state.step(action)
        obs = self._state.encode()
        info = {
            "choice": self._random_choice,
            "offset": self._state._offset
        }

        return obs, reward, done, info

    # Reset to give one observation to the Agent.
    def reset(self):
        self._random_choice = random.randrange(
            len(self._prices['date'])-self._state.bars_count-1)
        # prices = {'open': [self._prices['open'][self._random_choice]], 'high': [self._prices['high'][self._random_choice]],
        #           'low': [self._prices['low'][self._random_choice]], 'close': [self._prices['close'][self._random_choice]]}
        offset = self._random_choice
        self._state.reset(self._prices, offset)
        return self._state.encode()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]


class State:
    # Initialize the bar count and commision
    def __init__(self, bars_count, commision):
        assert bars_count > 0
        assert isinstance(bars_count, int)
        assert isinstance(commision, float)
        assert commision >= 0.0
        self.bars_count = bars_count
        self.commision = commision
        self.profit = 0.0

    # Reset values for first observation
    def reset(self, prices, offset):
        assert isinstance(prices, dict)
        assert isinstance(offset, int)
        self.have_position = False  # Whether or not we bought yet
        self._offset = offset
        self.open_price = 0.0  # The price we bought at
        self._prices = prices

    # Return the shape of the np array
    @property
    def shape(self):
        # [o,h,l,c] + bar_count + position + pnl
        return 6 * (self.bars_count),

    # Create the observation
    def encode(self):
        # create an empty array for observation
        state_obs = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0  # counter of shift

        # Loop through bars and get price values for ohlc, position, and pnl
        for bar_idx in range(-self.bars_count+1, 1):
            ofs = self._offset + bar_idx
            state_obs[shift] = self._prices['open'][ofs]
            shift += 1
            state_obs[shift] = self._prices['high'][ofs]
            shift += 1
            state_obs[shift] = self._prices['low'][ofs]
            shift += 1
            state_obs[shift] = self._prices['close'][ofs]
            shift += 1

            state_obs[shift] = float(self.have_position)
            shift += 1

            if not self.have_position:
                state_obs[shift] = 0.0
            else:
                state_obs[shift] = np.sign(self.curr_close() - self.open_price)
            shift += 1
        return state_obs

    # get the current closing price
    def curr_close(self):
        assert self._offset <= len(self._prices['close'])
        try:
            return self._prices['close'][self._offset]
        except:
            print(self._prices['close'][self._offset])
            quit

    # step of an enviornment
    # Check the action and perform said action
    def step(self, action):
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self.curr_close()

        # If the action is buy, take the closing price of the current position and set as open
        # set position = True
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close

        # If action is close, take the reward as the signed pnl (1, -1, 0) and set have_position is false, and done= True
        elif action == Actions.Close and self.have_position:
            self.have_position = False
            done = True
            reward += np.sign(self.curr_close() - self.open_price)
            self.profit += (self.curr_close()-self.open_price)/self.open_price

        self._offset += 1
        done |= self._offset >= len(
            self._prices['close'])-1

        if not done:
            close = self.curr_close()

        return done, reward
