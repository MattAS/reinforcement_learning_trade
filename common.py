from types import SimpleNamespace
import ptan
import numpy as np
import torch
import torch.nn as nn
import ptan.ignite as ptan_ignite
import warnings
from typing import Iterable
from ptan.ignite import Engine
from datetime import datetime, timedelta
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ignite.handlers import Checkpoint, DiskSaver

# Parameters for RL Agent
HYPERPARAMS = {
    'stocks': SimpleNamespace(**{
        'env_name': 'StocksEnv-v0',
        'run_name': 'stocks',
        'replay_size': 100000,
        'replay_initial': 10000,
        'target_net_sync': 10,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'epsilon_steps': 1000000,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 32,
        'reward_steps': 2
    })
}


# Convert batch to set of NumPy Arrays suitable for training
def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state
        else:
            lstate = np.array(exp.last_state, copy=False)
        last_states.append(lstate)
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


# Calculate loss
def calc_loss_dqn(batch, net, tgt_net, gamma, device='cpu'):
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0
    bellman_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, bellman_vals)


# Class to decay epsilon
class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector, params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - frame_idx/self.params.epsilon_frames
        self. selector.epsilon = max(self.params.epsilon_final, eps)


# Generate training batches sampled from buffer
def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer, initial: int, batch_size: int):
    buffer.populate(1)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


# Write metrics to TensorBoard
def setup_ignite(engine: Engine, params: SimpleNamespace, exp_source, run_name: str, net, extra_metrics: Iterable[str] = ()):
    warnings.simplefilter("ignore", category=UserWarning)
    handler = ptan_ignite.EndOfEpisodeHandler(
        exp_source, subsample_end_of_episode=100
    )
    handler.attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.0f, steps=%s, "
              "elapsed=%s" % (trainer.state.episode,
                              trainer.state.episode_reward,
                              trainer.state.episode_steps,
                              timedelta(seconds=int(passed))))
        path = './saves/(episode-%.3f.data' % trainer.state.episode
        torch.save(net.state_dict(), path)

    now = datetime.now().isoformat(timespec='minutes').replace(':', '')
    logdir = f"runs2/{now}-{params.run_name}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, 'avg_loss')

    metrics = ['reward', 'steps', 'avg_reward']
    handler = tb_logger.OutputHandler(
        tag='episodes', metric_names=metrics
    )
    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # write to tb every 100 Iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(
        tag="train", metric_names=metrics, output_transform=lambda a: a
    )
    event = ptan_ignite.PeriodEvents.ITERS_1000_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)
    return tb


@torch.no_grad()
def calc_values_of_states(states, net, device='cpu'):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)
