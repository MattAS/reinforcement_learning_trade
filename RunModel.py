import torch
import enviornment
import model
from data import Data
import numpy as np
from datetime import datetime
import time


class RunModel():
    def __init__(self, ticker, epsilon=0.02, epoch=70):
        self.ticker = ticker
        self.epsilon = epsilon
        self.epoch = epoch

    def run_model(self, obs):
        data = Data('day', 'minute', '1', 'true', self.ticker, '10')
        prices = data.get_prices_formatted()

        env = enviornment.StocksEnv(prices)

        net = model.DQN(env.observation_space.shape[0], env.action_space.n)
        net.load_state_dict(torch.load('RL\saves\(episode-60800.000.data'))
        total_reward = 0.00
        total_balance = 10000
        step_idx = 0

        balance = []
        rewards = []
        profit = []

        epochs = self.epoch
        epoch_step = 1

        while epoch_step <= epochs:
            obs = env.reset()
            while True:
                step_idx += 1
                obs_v = torch.tensor([obs])
                out_v = net(obs_v)
                action_idx = out_v.max(dim=1)[1].item()

                if np.random.random() < self.epsilon:
                    action_idx = env.action_space.sample()

                action = enviornment.Actions(action_idx)

                if action == enviornment.Actions.Buy and not env._state.have_position:
                    start_price = env._state.curr_close()
                    total_balance -= start_price
                    balance.append(total_balance)

                obs, reward, done, _ = env.step(action_idx)

                total_reward += reward
                rewards.append(total_reward)

                if step_idx % 100 == 0:
                    print("Epoch %d, Step_idx: %d reward = %.3f" %
                          (epoch_step, step_idx, total_reward))

                if done:
                    profit_recieved = (env._state.curr_close() -
                                       start_price)/start_price
                    profit.append(profit_recieved)
                    total_balance += env._state.curr_close()
                    balance.append(total_balance)
                    break
            epoch_step += 1

        file_name_profit = 'RL\logs\profits\profit_%s_%s.txt' % (
            self.ticker, str(time.time()))
        file_name_reward = 'RL\logs\\rewards\\reward_{}_{}.txt'.format(
            self.ticker, str(time.time()))

        file_name_balance = 'RL\logs\\balances\\balance_{}_{}.txt'.format(
            self.ticker, str(time.time()))

        with open(file_name_profit, 'w') as f:
            f.writelines('%s,' % x for x in profit)

        with open(file_name_reward, 'w') as f:
            f.writelines('%s,' % x for x in rewards)

        with open(file_name_balance, 'w') as f:
            f.writelines('%s,' % x for x in balance)


if __name__ == "__main__":
    goog_model = RunModel("TSLA")
    goog_model.run_model()
