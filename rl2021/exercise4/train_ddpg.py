import gym
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt

from rl2021.exercise4.agents import DDPG
from rl2021.exercise3.replay import ReplayBuffer

RENDER = False

PENDULUM_CONFIG = {
    "env": "Pendulum-v0",
    "target_return": -300.0,
    "episode_length": 200,
    "max_timesteps": 400000,
    "max_time": 120 * 60,
    "eval_freq": 2000,
    "eval_episodes": 5,
    "policy_learning_rate": 1e-1,
    "critic_learning_rate": 1e-1,
    "critic_hidden_size": [64, 64],
    "policy_hidden_size": [64, 64],
    "tau": 0.01,
    "batch_size": 10,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "save_filename": "pendulum_latest.pt",
}


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
                agent.update(batch)

        episode_timesteps += 1
        episode_return += reward

        if render:
            env.render()

        if max_steps == episode_timesteps:
            break
        obs = nobs

    return episode_timesteps, episode_return


def train(env: gym.Env, config, output: bool = True) -> Tuple[List[float], List[float]]:
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
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    eval_returns_all = []
    eval_times_all = []

    start_time = time.time()
    with tqdm(total=config["max_timesteps"]) as pbar:
        while timesteps_elapsed < config["max_timesteps"]:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break
            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
            episode_timesteps, _ = play_episode(
                env,
                agent,
                replay_buffer,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )
            timesteps_elapsed += episode_timesteps
            pbar.update(episode_timesteps)

            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                eval_returns = 0
                for _ in range(config["eval_episodes"]):
                    _, episode_return = play_episode(
                        env,
                        agent,
                        replay_buffer,
                        train=False,
                        explore=False,
                        render=False,
                        max_steps=config["episode_length"],
                        batch_size=config["batch_size"],
                    )
                    eval_returns += episode_return / config["eval_episodes"]
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean returns of {eval_returns}"
                    )
                    # pbar.write(f"Epsilon = {agent.epsilon}")
                eval_returns_all.append(eval_returns)
                eval_times_all.append(time.time() - start_time)
                if eval_returns >= config["target_return"]:
                    pbar.write(
                        f"Reached return {eval_returns} >= target return of {config['target_return']}"
                    )
                    break

    if config["save_filename"]:
        print("Saving to: ", agent.save(config["save_filename"]))

    return np.array(eval_returns_all), np.array(eval_times_all)


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    _ = train(env, CONFIG)
    env.close()