"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import pytest
import gym
import os.path
import numpy as np

def test_imports():
    from rl2021.exercise4 import DDPG
    from rl2021.exercise4.train_ddpg import PENDULUM_CONFIG as CONFIG

def test_config():
    from rl2021.exercise4.train_ddpg import PENDULUM_CONFIG
    assert "episode_length" in PENDULUM_CONFIG
    assert "max_timesteps" in PENDULUM_CONFIG
    assert "eval_freq" in PENDULUM_CONFIG
    assert "eval_episodes" in PENDULUM_CONFIG
    assert "policy_learning_rate" in PENDULUM_CONFIG
    assert "critic_learning_rate" in PENDULUM_CONFIG
    assert "policy_hidden_size" in PENDULUM_CONFIG
    assert "critic_hidden_size" in PENDULUM_CONFIG
    assert "tau" in PENDULUM_CONFIG
    assert "batch_size" in PENDULUM_CONFIG
    assert "gamma" in PENDULUM_CONFIG
    assert "buffer_capacity" in PENDULUM_CONFIG
    assert "save_filename" in PENDULUM_CONFIG

def test_restore_file():
    from rl2021.exercise4 import DDPG
    from rl2021.exercise4.train_ddpg import PENDULUM_CONFIG
    env = gym.make("Pendulum-v0")
    agent = DDPG(
        action_space=env.action_space,
        observation_space=env.observation_space,
        **PENDULUM_CONFIG
    )
    agent.restore("pendulum_latest.pt")
