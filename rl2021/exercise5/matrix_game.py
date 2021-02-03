from collections.abc import Iterable
import numpy as np
import gym


def matrix_shape(matrix, shape=[]):
    """
    Get shape of matrix
    :param matrix: list of lists
    :return: shape of matrix
    """
    if not isinstance(matrix, Iterable):
        return shape
    else:
        shape.append(len(matrix))
        return matrix_shape(matrix[0], shape)


def actions_to_onehot(num_actions, actions):
    """
    Transfer actions to onehot representation
    :param num_actions: list of number of actions of each agent
    :param actions: list of actions (int) for each agent
    :return: onehot representation of actions
    """
    onehot = [[0] * num_action for num_action in num_actions]
    for ag, act in enumerate(actions):
        onehot[ag][act] = 1
    return onehot


class MatrixGame(gym.Env):
    def __init__(self, payoff_matrix, ep_length, last_action_state=True):
        """
        Create matrix game
        :param payoff_matrix: list of lists or numpy array for payoff matrix of all agents
        :param ep_length: length of episode (before done is True)
        :param last_action_state: boolean flag indicating whether last actions should be returned
                                  as state of the environment or just 0s for all agents
        """
        self.payoff = payoff_matrix
        self.num_actions = matrix_shape(payoff_matrix, [])
        self.n_agents = len(self.num_actions)
        self.ep_length = ep_length
        self.last_action_state = last_action_state

        self.last_actions = [-1 for _ in range(self.n_agents)]
        self.t = 0

        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(num_action) for num_action in self.num_actions])
        if last_action_state:
            shape = (self.n_agents, self.num_actions[0])
            low = np.zeros(shape)
            high = np.ones(shape)
            obs_space = gym.spaces.Box(shape=shape, low=low, high=high)
            self.observation_space = gym.spaces.Tuple([obs_space for _ in range(self.n_agents)])
        else:
            self.observation_space = gym.spaces.Tuple([gym.spaces.Discrete(1) for _ in range(self.n_agents)])

    def reset(self):
        self.t = 0
        self.last_actions = actions_to_onehot(self.num_actions, [0] * self.n_agents)
        if self.last_action_state:
            return [self.last_actions for _ in range(self.n_agents)]
        else:
            return [0] * self.n_agents

    def step(self, action):
        self.t += 1
        self.last_actions = actions_to_onehot(self.num_actions, action)
        reward = self.payoff
        for a in action:
            reward = reward[a]

        if self.last_action_state:
            obs = [self.last_actions for _ in range(self.n_agents)]
        else:
            obs = [0] * self.n_agents

        if self.t >= self.ep_length:
            done = True
        else:
            done = False

        return obs, [reward] * self.n_agents, [done] * self.n_agents, {}

    def render(self):
        print(f"Step {self.t}:")
        for i in range(self.n_agents):
            print(f"\tAgent {i + 1} action: {self.last_actions[i]}")


# penalty game
def create_penalty_game(penalty, ep_length, last_action_state=True):
    assert penalty <= 0
    payoff = [
        [penalty, 0, 10],
        [0, 2, 0],
        [10, 0, penalty],
    ]
    game = MatrixGame(payoff, ep_length, last_action_state)
    return game

# climbing game
def create_climbing_game(ep_length, last_action_state=True):
    payoff = [
        [11, -30, 0],
        [-30, 7, 0],
        [0, 6, 5],
    ]
    game = MatrixGame(payoff, ep_length, last_action_state)
    return game
