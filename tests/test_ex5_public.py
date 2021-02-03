"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import pytest
import gym
import numpy as np
from rl2021.exercise5.agents import IndependentQLearningAgents, JointActionLearning
from rl2021.exercise5.matrix_game import MatrixGame

from gym.spaces import Discrete


def test_iqlagents_0():
    n_agents = 2
    act_space = gym.spaces.Tuple([gym.spaces.Discrete(3) for _ in range(n_agents)])
    obs_space = gym.spaces.Tuple([gym.spaces.Discrete(2) for _ in range(n_agents)])

    agents = IndependentQLearningAgents(
        num_agents=n_agents,
        action_spaces=act_space,
        observation_spaces=obs_space,
        gamma=0.99,
        learning_rate=0.1,
        epsilon=0.9,
    )
    agents.schedule_hyperparameters(0, 10)
    assert hasattr(agents, "epsilon")
    assert hasattr(agents, "learning_rate")
    assert hasattr(agents, "gamma")
    assert hasattr(agents, "q_tables")

    assert type(agents.epsilon) == float
    assert type(agents.learning_rate) == float
    assert type(agents.gamma) == float
    assert type(agents.q_tables) == list

    assert agents.epsilon >= 0.0
    assert agents.epsilon <= 1.0
    assert agents.learning_rate == 0.1
    assert agents.gamma == 0.99
    assert len(agents.q_tables) == n_agents

def test_iqlagents_1():
    n_agents = 2
    act_space = gym.spaces.Tuple([gym.spaces.Discrete(3) for _ in range(n_agents)])
    obs_space = gym.spaces.Tuple([gym.spaces.Discrete(2) for _ in range(n_agents)])
    agents = IndependentQLearningAgents(
        num_agents=n_agents,
        action_spaces=act_space,
        observation_spaces=obs_space,
        gamma=0.99,
        learning_rate=0.1,
        epsilon=0.9,
    )
    action = list(act_space.sample())
    obs = list(obs_space.sample())
    reward = [0.0] * n_agents
    obs_n = list(obs_space.sample())
    dones = [False] * n_agents

    agents.learn(obs, action, reward, obs_n, dones)

    for i, (o, a) in enumerate(zip(obs, action)):
        assert (o, a) in agents.q_tables[i]
        assert type(agents.q_tables[i][(o, a)]) == float


def test_jalagents_0():
    n_agents = 2
    act_space = gym.spaces.Tuple([gym.spaces.Discrete(3) for _ in range(n_agents)])
    obs_space = gym.spaces.Tuple([gym.spaces.Discrete(2) for _ in range(n_agents)])
    agents = JointActionLearning(
        num_agents=n_agents,
        action_spaces=act_space,
        observation_spaces=obs_space,
        gamma=0.99,
        learning_rate=0.1,
        epsilon=0.9,
    )
    agents.schedule_hyperparameters(0, 10)

    assert hasattr(agents, "epsilon")
    assert hasattr(agents, "learning_rate")
    assert hasattr(agents, "gamma")
    assert hasattr(agents, "q_tables")
    assert hasattr(agents, "models")
    assert hasattr(agents, "c_obss")

    assert type(agents.epsilon) == float
    assert type(agents.learning_rate) == float
    assert type(agents.gamma) == float
    assert type(agents.q_tables) == list
    assert type(agents.models) == list
    assert type(agents.c_obss) == list

    assert agents.epsilon >= 0.0
    assert agents.epsilon <= 1.0
    assert agents.learning_rate == 0.1
    assert agents.gamma == 0.99
    assert len(agents.q_tables) == n_agents
    assert len(agents.models) == n_agents
    assert len(agents.c_obss) == n_agents

def test_jalagents_1():
    n_agents = 2
    act_space = gym.spaces.Tuple([gym.spaces.Discrete(3) for _ in range(n_agents)])
    obs_space = gym.spaces.Tuple([gym.spaces.Discrete(2) for _ in range(n_agents)])
    agents = JointActionLearning(
        num_agents=n_agents,
        action_spaces=act_space,
        observation_spaces=obs_space,
        gamma=0.99,
        learning_rate=0.1,
        epsilon=0.9,
    )

    action = list(act_space.sample())
    obs = [0] * n_agents
    reward = [0.0] * n_agents
    obs_n = list(obs_space.sample())
    dones = [False] * n_agents

    agents.learn(obs, action, reward, obs_n, dones)

    for i, o in enumerate(obs):
        assert (o, tuple(action)) in agents.q_tables[i]
        assert type(agents.q_tables[i][(o, tuple(action))]) == float

        # check observation count
        assert o in agents.c_obss[i]
        assert agents.c_obss[i][o] == 1

        # check model count
        other_i = 0 if i == 1 else 1
        assert agents.models[i][o][action[other_i]] == 1