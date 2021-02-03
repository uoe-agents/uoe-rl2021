import time

import matplotlib.pyplot as plt
import numpy as np


def visualise_q_table(env, q_table, i):
    print(q_table)
    for a1 in range(3):
        q_value = q_table[(0, a1)]
        actual_value = 0
        if i == 0:
            print(env.payoff[a1])
        else:
             print(list(np.array(env.payoff).T[a1]))
        for a2 in range(3):
            if i == 0:
                actual_value += env.payoff[a1][a2]
            else:
                 actual_value += env.payoff[a2][a1]               
        # expectation over all three values
        actual_value /= 3
        print(f"Q({a1 + 1}) = {q_value:.2f}\t\tActual Value: {actual_value}")
    print()


def visualise_joint_q_table(env, q_table, i):
    for a1 in range(3):
        for a2 in range(3):
            q_value = q_table[(0, (a1, a2))]
            actual_value = env.payoff[a1][a2]
            print(f"Q({a1},{a2}) = {q_value:.2f}\t\tActual Value: {actual_value}")
    print()


def evaluate(env, agents, max_steps, eval_episodes, render, output=True):
    """
    Evaluate configuration on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param agents (MultiAgent): agent to act in environment
    :param max_steps (int): max number of steps per evaluation episode
    :param eval_episodes (int): number of evaluation episodes
    :param render (bool): flag whether evaluation runs should be rendered
    :param output (bool): flag whether mean evaluation results should be printed
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    episodic_returns = []
    for eps_num in range(eval_episodes):
        obss = env.reset()
        if (eps_num == eval_episodes - 1) and render:
            env.render()
            time.sleep(0.5)
        episodic_return = 0
        dones = [False] * agents.num_agents
        steps = 0

        while not all(dones) and steps < max_steps:
            acts = agents.act(obss)
            n_obss, rewards, dones, _ = env.step(acts)
            if (eps_num == eval_episodes - 1) and render:
                env.render()
                print(rewards[0])
                time.sleep(0.5)

            episodic_return += rewards[0]
            steps += 1

            obss = n_obss

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns)
    std_return = np.std(episodic_returns)

    if output:
        # print(f"EVALUATION: EPISODIC RETURNS: {episodic_returns}")
        print(f"EVALUATION: MEAN RETURN OF {mean_return}")
    return mean_return, std_return
