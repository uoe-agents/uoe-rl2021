"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import pytest
import gym
import os.path
import numpy as np

def test_imports_0():
    from rl2021.exercise3 import DQN, Reinforce, ReplayBuffer
    from rl2021.exercise3.train_dqn import CARTPOLE_CONFIG as DQN_CARTPOLE_CONFIG
    from rl2021.exercise3.train_dqn import LUNARLANDER_CONFIG as DQN_LUNARLANDER_CONFIG

    from rl2021.exercise3.train_reinforce import CARTPOLE_CONFIG as REINF_CARTPOLE_CONFIG


def test_config_0():
    from rl2021.exercise3.train_dqn import CARTPOLE_CONFIG
    assert "eval_freq" in CARTPOLE_CONFIG
    assert "eval_episodes" in CARTPOLE_CONFIG
    assert "episode_length" in CARTPOLE_CONFIG
    assert "max_timesteps" in CARTPOLE_CONFIG

    assert "batch_size" in CARTPOLE_CONFIG
    assert "buffer_capacity" in CARTPOLE_CONFIG

def test_config_1():
    from rl2021.exercise3.train_dqn import LUNARLANDER_CONFIG
    assert "eval_freq" in LUNARLANDER_CONFIG
    assert "eval_episodes" in LUNARLANDER_CONFIG
    assert "episode_length" in LUNARLANDER_CONFIG
    assert "max_timesteps" in LUNARLANDER_CONFIG

    assert "batch_size" in LUNARLANDER_CONFIG
    assert "buffer_capacity" in LUNARLANDER_CONFIG
    

def test_config_2():
    from rl2021.exercise3.train_reinforce import CARTPOLE_CONFIG
    assert "eval_freq" in CARTPOLE_CONFIG
    assert "eval_episodes" in CARTPOLE_CONFIG
    assert "episode_length" in CARTPOLE_CONFIG
    assert "max_timesteps" in CARTPOLE_CONFIG


def test_restore_file_0():
    from rl2021.exercise3 import DQN
    from rl2021.exercise3.train_dqn import LUNARLANDER_CONFIG as DQN_LUNARLANDER_CONFIG
    env = gym.make("LunarLander-v2")
    agent = DQN(
        action_space=env.action_space,
        observation_space=env.observation_space,
        **DQN_LUNARLANDER_CONFIG
    )
    agent.restore("dqn_lunarlander_latest.pt")