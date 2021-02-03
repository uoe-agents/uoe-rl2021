"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import pytest
import numpy as np
from rl2021.utils import MDP, Transition
from rl2021.exercise1 import ValueIteration, PolicyIteration
from rl2021.exercise2 import QLearningAgent, MonteCarloAgent

from gym.spaces import Discrete


def test_qagent_0():
    agent = QLearningAgent(
        action_space=Discrete(3),
        obs_space=Discrete(3),
        gamma=0.99,
        alpha=1.0,
        epsilon=0.9,
    )
    agent.schedule_hyperparameters(0, 10)


    assert hasattr(agent, "epsilon")
    assert hasattr(agent, "alpha")
    assert hasattr(agent, "q_table")
    assert hasattr(agent, "gamma")
    assert type(agent.epsilon) == float
    assert type(agent.alpha) == float
    assert agent.epsilon >= 0.0
    assert agent.epsilon <= 1.0

def test_qagent_1():
    agent = QLearningAgent(
        action_space=Discrete(3),
        obs_space=Discrete(3),
        gamma=0.99,
        alpha=1.0,
        epsilon=0.9,
    )
    space = Discrete(10)
    action = space.sample()
    obs = space.sample()
    reward = 0.0
    obs_n = space.sample()

    agent.learn(obs, action, reward, obs_n, False)

    assert (obs, action) in agent.q_table
    assert type(agent.q_table[(obs, action)]) == float


def test_montecarlo_0():
    agent = MonteCarloAgent(
        action_space=Discrete(3), obs_space=Discrete(3), gamma=0.99, epsilon=0.9,
    )
    agent.schedule_hyperparameters(0, 10)

    assert hasattr(agent, "epsilon")
    assert hasattr(agent, "q_table")
    assert hasattr(agent, "gamma")
    assert type(agent.epsilon) == float
    assert agent.epsilon >= 0.0
    assert agent.epsilon <= 1.0
