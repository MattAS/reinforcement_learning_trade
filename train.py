import ptan
import torch
import torch.optim as optim
from data import Data
from model import DQN
from enviornment import Actions, StocksEnv, State
import common
import argparse
from ignite.engine import Engine
import pathlib
from datetime import datetime
import time
from ignite.engine import Engine
from ignite.contrib.handlers import tensorboard_logger as tb_logger
import pickle
import pandas as pd

SAVES_DIR = pathlib.Path("saves")

NAME = 'Stocks01'

if __name__ == '__main__':
    params = common.HYPERPARAMS['stocks']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    data = Data('day', 'minute', '1', 'true', 'GOOG', '10')
    val_data = Data('day', 'minute', '1', '10', 'true', 'AAPL')
    val_price = val_data.get_prices_formatted()
    prices = data.get_prices_formatted()
    prices_df = pd.DataFrame(prices)
    prices_df.to_csv('prices_df.csv')
    print('complete')
    env = StocksEnv(prices)
    env_val = StocksEnv(val_price)

    net = DQN(env.observation_space.shape[0], env.action_space.n)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params.epsilon_start)
    epsilon_tracker = ptan.actions.EpsilonTracker(
        selector, params.epsilon_start, params.epsilon_final, params.epsilon_steps)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=args.gamma, steps_count=args.reward_steps)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(
            batch, net, tgt_net.target_model, gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            'loss': loss_v.item(),
            'epsilon': selector.epsilon
        }

    engine = Engine(process_batch)
    tb = common.setup_ignite(engine, params, exp_source,
                             NAME, net, extra_metrics=('values_mean',))

    engine.run(common.batch_generator(
        buffer, params.replay_initial, params.batch_size))
