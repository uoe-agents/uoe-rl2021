import time

import matplotlib.pyplot as plt
import numpy as np


def evaluate(env, agent, max_steps, eval_episodes, render):
    """
    Evaluate configuration on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param agent (Agent): agent to act in environment
    :param max_steps (int): max number of steps per evaluation episode
    :param eval_episodes (int): number of evaluation episodes
    :param render (bool): flag whether evaluation runs should be rendered
    :return (float, int): mean of returns received over episodes and number of negative
        return evaluation, episodes
    """
    episodic_returns = []
    for eps_num in range(eval_episodes):
        obs = env.reset()
        if (eps_num == eval_episodes - 1) and render:
            env.render()
            time.sleep(0.5)
        episodic_return = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            act = agent.act(obs)
            n_obs, reward, done, info = env.step(act)
            if (eps_num == eval_episodes - 1) and render:
                env.render()
                print(reward)
                time.sleep(0.5)

            episodic_return += reward
            steps += 1

            obs = n_obs

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns)
    negative_returns = sum([ret < 0 for ret in episodic_returns])

    return mean_return, negative_returns
