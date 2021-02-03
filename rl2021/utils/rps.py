import gym
from gym import spaces


class RPS(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.outcome_table = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
        self.latest_actions = [None, None]
        self.action_space = spaces.Tuple([spaces.Discrete(3) for _ in range(2)])

    def reset(self):
        self.latest_actions = [None, None]
        return [0, 0]

    def step(self, action):
        self.latest_actions = action
        rewards = [
            self.outcome_table[action[0]][action[1]],
            self.outcome_table[action[1]][action[0]],
        ]
        return [0, 0], rewards, [True, True], [{}, {}]

    def render(self, mode="human", close=False):
        print(
            "Agent 1 Latest Action : ",
            self.latest_actions[0],
            " Agent 2 Latest Action : ",
            self.latest_actions[1],
        )
