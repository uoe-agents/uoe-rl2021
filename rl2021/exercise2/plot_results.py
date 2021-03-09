import gym
import matplotlib.pyplot as plt
import numpy as np

from rl2021.exercise2.train_q_learning import CONFIG as QL_CONFIG
from rl2021.exercise2.train_q_learning import train as ql_train
from rl2021.exercise2.train_monte_carlo import CONFIG as MC_CONFIG
from rl2021.exercise2.train_monte_carlo import train as mc_train


plt.style.use("seaborn-darkgrid")
plt.rcParams.update({"font.size": 15})

TRAINING_RUNS = 3 # NUMBER OF SEEDS TO RUN / YOU MAY WANT TO INCREASE THIS IF YOU HAVE TIME TO GET MORE CONSISTENT MEAN/VARIANCE PLOTS
EVAL_FREQ = 1000
EVAL_EPISODES = 100

CONFIGS = [
    (MC_CONFIG, "Monte Carlo", mc_train),
    (QL_CONFIG, "Q-learning", ql_train),
]

def prepare_config(config, alg_name, train_f):
    """
    Add further parameters to configuration file used in evaluation for plots

    :param config (Dict): configuration file to extend
    :param alg_name (str): name of algorithm for this configuration
    :param train_f (Callable): training function of algorithm
    """
    config["alg"] = alg_name
    config["train"] = train_f
    config["eval_freq"] = EVAL_FREQ
    config["eval_episodes"] = EVAL_EPISODES


def plot_timesteps(values: np.ndarray, stds: np.ndarray, xlabel: str, ylabel: str, legend_name: str):
    """
    Plot values with respect to timesteps
    
    :param values (np.ndarray): numpy array of values to plot as y values
    :param std (np.ndarray): numpy array of stds of y values to plot as shading
    :param xlabel (str): label of x-axis
    :param ylabel (str): label of y-axis
    :param legend_name (str): name of algorithm
    """
    x_values = EVAL_FREQ + np.arange(len(values)) * EVAL_FREQ
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
    plt.figure(figsize=(8, 8))
    axes = plt.gca()
    hlines = False

    env_name = None
    for config, name, train in CONFIGS:
        env_name = "Taxi"
        prepare_config(config, name, train)

        plt.title(f"Average Returns on {env_name}")

        # draw goal line
        if hlines == False:
            x_min = 0
            x_max = config["total_eps"]
            plt.hlines(y=7, xmin=x_min, xmax=x_max, colors='k', linestyles='dotted', label="Taxi threshold = 7")
            axes.set_ylim([-200,20])
            hlines = True

        print(f"{config['alg']} performance on {env_name}")

        env = gym.make("Taxi-v3")

        num_returns = config["total_eps"] // config["eval_freq"]

        eval_returns = np.zeros((TRAINING_RUNS, num_returns))
        for i in range(TRAINING_RUNS):
            print(f"Executing training for {name} - run {i + 1}")
            env.seed(i * 100)
            _, returns, _, _ = config["train"](env, config)
            returns = np.array(returns)
            # correct for missing returns (repeat last one)
            if returns.shape[-1] < eval_returns.shape[-1]:
                returns_extended = np.zeros(num_returns)
                returns_extended[: returns.shape[-1]] = returns
                returns_extended[returns.shape[-1] :] = returns[-1]
                returns = returns_extended
            eval_returns[i, :] = returns
        returns_total = eval_returns.mean(axis=0)
        returns_std = eval_returns.std(axis=0)
        plot_timesteps(returns_total, returns_std, "Episodes", "Mean Eval Returns", name)

    assert env_name is not None
    plt.savefig(f"{env_name.lower()}_results.pdf", format="pdf")
    plt.show()
