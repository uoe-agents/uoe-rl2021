import gym
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple

from rl2021.exercise3.agents import Reinforce

RENDER = False # FALSE FOR FASTER TRAINING / TRUE TO VISUALIZE ENVIRONMENT DURING EVALUATION

CARTPOLE_MAX_EPISODE_STEPS = 200 # USED FOR EVALUATION / DO NOT CHANGE

### TUNE HYPERPARAMETERS HERE ###
CARTPOLE_CONFIG = {
    "env": "CartPole-v1",
    "episode_length": 200,
    "max_timesteps": 200000,
    "eval_freq": 5000,
    "eval_episodes": 5,
    "max_time": 30 * 60,
    "gamma": 0.99,
    "hidden_size": (16,16),
    "learning_rate": 1e-2,
    "save_filename": None,
}

CONFIG = CARTPOLE_CONFIG


def play_episode(
    env: gym.Env,
    agent: Reinforce,
    train: bool = True,
    explore=True,
    render=False,
    max_steps=200,
) -> Tuple[int, float]:
    """
    Play one episode and train reinforce algorithm

    :param env (gym.Env): gym environment
    :param agent (Reinforce): Reinforce agent
    :param train (bool): flag whether training should be executed
    :param explore (bool): flag whether exploration is used
    :param render (bool): flag whether environment should be visualised
    :param max_steps (int): max number of timesteps for the episode
    :return (Tuple[int, float]): total number of executed steps and received reward
    """
    obs = env.reset()

    if render:
        env.render()

    done = False
    num_steps = 0
    episode_return = 0

    observations = []
    actions = []
    rewards = []

    while not done and num_steps < max_steps:
        action = agent.act(np.array(obs), explore=explore)
        nobs, rew, done, _ = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(rew)

        if render:
            env.render()

        num_steps += 1
        episode_return += rew

        obs = nobs

    if train:
        loss = agent.update(rewards, observations, actions)

    return num_steps, episode_return


def train(env: gym.Env, config, output: bool = True) -> Tuple[List[float], List[float]]:
    """
    Execute training of REINFORCE on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): mean average returns during training, times of evaluation
    """
    timesteps_elapsed = 0

    agent = Reinforce(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )

    total_steps = config["max_timesteps"]
    eval_returns_all = []
    eval_times_all = []

    start_time = time.time()
    with tqdm(total=total_steps) as pbar:
        while timesteps_elapsed < total_steps:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break
            agent.schedule_hyperparameters(timesteps_elapsed, total_steps)
            num_steps, _ = play_episode(
                env,
                agent,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
            )
            timesteps_elapsed += num_steps
            pbar.update(num_steps)

            if timesteps_elapsed % config["eval_freq"] < num_steps:
                eval_return = 0
                if config["env"] == "CartPole-v1":
                    max_steps = CARTPOLE_MAX_EPISODE_STEPS
                else:
                    raise ValueError(f"Unknown environment {config['env']}")

                for _ in range(config["eval_episodes"]):
                    _, total_reward = play_episode(
                        env,
                        agent,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=max_steps,
                    )
                    eval_return += total_reward / (config["eval_episodes"])
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean return of {eval_return}"
                    )
                eval_returns_all.append(eval_return)
                eval_times_all.append(time.time() - start_time)


    if config["save_filename"]:
        print("Saving to: ", agent.save(config["save_filename"]))

    return np.array(eval_returns_all), np.array(eval_times_all)


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    returns, times = train(env, CONFIG)
    env.close()
