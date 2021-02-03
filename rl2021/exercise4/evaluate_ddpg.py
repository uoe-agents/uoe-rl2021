import gym
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt

from rl2021.exercise3.replay import ReplayBuffer
from rl2021.exercise4.agents import DDPG
from rl2021.exercise4.train_ddpg import PENDULUM_CONFIG


RENDER = False
CONFIG = PENDULUM_CONFIG


def play_episode(
        env,
        agent,
        replay_buffer,
        train=True,
        explore=True,
        render=False,
        max_steps=200,
        batch_size=64,
):
    obs = env.reset()
    done = False
    losses = []
    if render:
        env.render()

    episode_timesteps = 0
    episode_return = 0

    while not done:
        action = agent.act(obs, explore=explore)
        nobs, reward, done, _ = env.step(action)
        if train:
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.update(batch)["q_loss"]
                losses.append(loss)

        episode_timesteps += 1
        episode_return += reward

        if render:
            env.render()

        if max_steps == episode_timesteps:
            break
        obs = nobs

    return episode_timesteps, episode_return, losses


def evaluate(env: gym.Env, config, output: bool = True) -> Tuple[List[float], List[float]]:
    """
    Execute training of DDPG on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): eval returns during training, times of evaluation
    """
    timesteps_elapsed = 0

    agent = DDPG(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    agent.restore(config['save_filename'])

    eval_returns_all = []
    eval_times_all = []


    eval_returns = 0
    for _ in range(config["eval_episodes"]):
        _, episode_return, _ = play_episode(
            env,
            agent,
            0,
            train=False,
            explore=False,
            render=True,
            max_steps=config["episode_length"],
            batch_size=config["batch_size"],
        )
        eval_returns += episode_return / config["eval_episodes"]

    return eval_returns


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    returns = evaluate(env, CONFIG)
    print(returns)
    env.close()
