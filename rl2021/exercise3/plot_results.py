import gym
import matplotlib.pyplot as plt
import numpy as np

from rl2021.exercise3.train_dqn import CARTPOLE_CONFIG as DQN_CARTPOLE_CONFIG
from rl2021.exercise3.train_dqn import LUNARLANDER_CONFIG as DQN_LUNARLANDER_CONFIG
from rl2021.exercise3.train_dqn import train as dqn_train
from rl2021.exercise3.train_reinforce import (
    CARTPOLE_CONFIG as REINFORCE_CARTPOLE_CONFIG,
)
from rl2021.exercise3.train_reinforce import train as reinforce_train


plt.style.use("seaborn-darkgrid")
plt.rcParams.update({"font.size": 15})

TRAINING_RUNS = 3

CARTPOLE_CONFIGS = [
    (DQN_CARTPOLE_CONFIG, "DQN", dqn_train),
    (REINFORCE_CARTPOLE_CONFIG, "REINFORCE", reinforce_train),
]

LUNARLANDER_CONFIGS = [
    (DQN_LUNARLANDER_CONFIG, "DQN", dqn_train),
]

CONFIGS = CARTPOLE_CONFIGS
# CONFIGS = LUNARLANDER_CONFIGS

def prepare_config(config, alg_name, train_f):
    """
    Add further parameters to configuration file used in evaluation for plots

    :param config (Dict): configuration file to extend
    :param alg_name (str): name of algorithm for this configuration
    :param train_f (Callable): training function of algorithm
    """
    env_name = config['env'][:-3]

    config["alg"] = alg_name
    config["train"] = train_f
    config["plot_loss"] = False


def plot_timesteps(values: np.ndarray, stds: np.ndarray, xlabel: str, ylabel: str, legend_name: str, eval_freq: int):
    """
    Plot values with respect to timesteps
    
    :param values (np.ndarray): numpy array of values to plot as y values
    :param std (np.ndarray): numpy array of stds of y values to plot as shading
    :param xlabel (str): label of x-axis
    :param ylabel (str): label of y-axis
    :param legend_name (str): name of algorithm
    """
    x_values = eval_freq + np.arange(len(values)) * eval_freq
    plt.plot(x_values, values, "-", alpha=0.7, label=f"{legend_name}")
    plt.fill_between(
        x_values,
        values - stds,
        values + stds,
        alpha=0.2,
        antialiased=True,
    )
    plt.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=0.3)


if __name__ == "__main__":
    # execute training and evaluation to generate return plots
    env_name = None
    plt.figure(figsize=(8, 8))
    axes = plt.gca()

    for config, name, train in CONFIGS:
        env_name = config['env'][:-3]
        prepare_config(config, name, train)


        print(f"{config['alg']} performance on {env_name}")

        env = gym.make(config["env"])

        num_returns = int(config["max_timesteps"] / config["eval_freq"])

        eval_returns = np.zeros((TRAINING_RUNS, num_returns))
        for i in range(TRAINING_RUNS):
            print(f"Executing training for {name} - run {i + 1}")
            returns, _ = config["train"](env, config, output=True)
            # correct for missing returns (repeat last one)
            if returns.shape[-1] < eval_returns.shape[-1]:
                returns_extended = np.zeros(num_returns)
                returns_extended[: returns.shape[-1]] = returns
                returns_extended[returns.shape[-1] :] = returns[-1]
                returns = returns_extended
            eval_returns[i, :] = returns


        hlines = False

        plt.title(f"Average Returns on {env_name}")
        # draw goal line
        if hlines == False:
            x_min = 0
            x_max = config["max_timesteps"]
            if env_name.lower() == "lunarlander":
                plt.hlines(y=195, xmin=x_min, xmax=x_max, colors='k', linestyles='dotted', label="LunarLander threshold")
                axes.set_ylim([-200,200])
            elif env_name.lower() == "cartpole":
                plt.hlines(y=195, xmin=x_min, xmax=x_max, colors='k', linestyles='dotted', label="Cartpole threshold")
                axes.set_ylim([0,200])
            hlines = True

        returns_total = eval_returns.mean(axis=0)
        returns_std = eval_returns.std(axis=0)
        plot_timesteps(returns_total, returns_std, "Timestep", "Mean Eval Returns", name, config["eval_freq"])

    assert env_name is not None
    plt.savefig(f"{env_name.lower()}_results.pdf", format="pdf")
    plt.show()
