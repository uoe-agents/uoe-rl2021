import gym

from rl2021.exercise5.agents import IndependentQLearningAgents
from rl2021.exercise5.utils import visualise_q_table, evaluate
from rl2021.exercise5.matrix_game import create_penalty_game, create_climbing_game


PEN_CONFIG = {
    "env": "penalty",
    "env_args": (-5, 1, False),
    "total_eps": 5000,
    "eps_max_steps": 10,
    "eval_freq": 100,
    "gamma": 0.99,
    "lr": 0.05,
    "epsilon": 0.9,
    "goal_payoff": 10,
}

CLIMBING_CONFIG = {
    "env": "climbing",
    "env_args": (1, False),
    "total_eps": 5000,
    "eps_max_steps": 10,
    "eval_freq": 100,
    "gamma": 0.99,
    "lr": 0.05,
    "epsilon": 0.9,
    "goal_payoff": 11,
}

CONFIG = PEN_CONFIG
# CONFIG = CLIMBING_CONFIG


def iql_eval(env, config, q_tables, max_steps=10, eval_episodes=500, render=False, output=True):
    """
    Evaluate configuration of independent Q-learning on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_tables (List[Dict[(Obs, Act), float]]): Q-tables mapping observation-action to Q-values for each agent
    :param max_steps (int): number of steps per evaluation episode
    :param eval_episodes (int): number of evaluation episodes
    :param render (bool): flag whether evaluation runs should be rendered
    :param output (bool): flag whether mean evaluation performance should be printed
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agents = IndependentQLearningAgents(
            num_agents=env.n_agents,
            action_spaces=env.action_space,
            observation_spaces=env.observation_space,
            gamma=config["gamma"],
            learning_rate=config["lr"],
            epsilon=0.0,
        )
    eval_agents.q_tables = q_tables
    return evaluate(env, eval_agents, max_steps, eval_episodes, render, output)


def train(env, config, output=True):
    """
    Train and evaluate independent Q-learning on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param output (bool): flag if mean evaluation results should be printed
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        total reward over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table
    """
    agents = IndependentQLearningAgents(
            num_agents=env.n_agents,
            action_spaces=env.action_space,
            observation_spaces=env.observation_space,
            gamma=config["gamma"],
            learning_rate=config["lr"],
            epsilon=config["epsilon"],
        )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_return_stds = []

    for eps_num in range(config["total_eps"]):
        obss = env.reset()
        episodic_return = 0
        t = 0

        while t < config["eps_max_steps"]:
            agents.schedule_hyperparameters(step_counter, max_steps)
            acts = agents.act(obss)
            n_obss, rewards, dones, _ = env.step(acts)
            agents.learn(obss, acts, rewards, n_obss, dones)

            t += 1
            step_counter += 1
            # fully cooperative tasks --> only track single reward / all rewards are identical
            episodic_return += rewards[0]

            if all(dones):
                break

            obss = n_obss

        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, std_return = iql_eval(
                env, config, agents.q_tables, output=output
            )
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)
            if mean_return >= config["goal_payoff"]:
                return total_reward, evaluation_return_means, evaluation_return_stds, agents.q_tables

    return total_reward, evaluation_return_means, evaluation_return_stds, agents.q_tables


if __name__ == "__main__":
    if CONFIG["env"] == "penalty":
        env = create_penalty_game(*CONFIG["env_args"])
    else:
        env = create_climbing_game(*CONFIG["env_args"])
    total_reward, _, _, q_tables = train(env, CONFIG)
    print()
    # print(f"Total reward over training: {total_reward}\n")
    print("Q-table:")
    print(q_tables)
    for i, q_table in enumerate(q_tables):
        visualise_q_table(env, q_table, i)
