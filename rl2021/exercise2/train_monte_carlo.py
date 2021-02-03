import gym

from rl2021.exercise2.agents import MonteCarloAgent
from rl2021.exercise2.utils import evaluate

REAL_MAX_EPISODE_STEPS = 100 # CUT OF AN EPISODE THAT RUNS LONGER THAN THAT. DO NOT CHANGE

### TUNE HYPERPARAMETERS HERE ###
CONFIG = {
    "env": "Taxi-v3",
    "total_eps": 100000,
    "eps_max_steps": 100,
    "eval_episodes": 500,
    "eval_freq": 5000,
    "gamma": 0.99,
    "epsilon": 0.0,
}


def monte_carlo_eval(
        env,
        config,
        q_table,
        max_steps=REAL_MAX_EPISODE_STEPS,
        render=False):
    """
    Evaluate configuration of MC on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param max_steps (int): max number of steps per evaluation episode
    :param eval_episodes (int): number of evaluation episodes
    :param render (bool): flag whether evaluation runs should be rendered
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = MonteCarloAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=CONFIG["gamma"],
        epsilon=0.0,
    )
    eval_agent.q_table = q_table
    return evaluate(env, eval_agent, max_steps, config["eval_episodes"], render)


def train(env, config):
    """
    Train and evaluate MC on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        returns over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table, final state-action counts
    """
    agent = MonteCarloAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        epsilon=config["epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []

    for eps_num in range(1, config["total_eps"] + 1):
        obs = env.reset()

        t = 0
        episodic_return = 0

        obs_list, act_list, rew_list = [], [], []
        while t < config["eps_max_steps"]:
            agent.schedule_hyperparameters(step_counter, max_steps)
            act = agent.act(obs)

            n_obs, reward, done, _ = env.step(act)

            obs_list.append(obs)
            rew_list.append(reward)
            act_list.append(act)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = n_obs

        agent.learn(obs_list, act_list, rew_list)
        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, negative_returns = monte_carlo_eval(env, config, agent.q_table)
            print(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return} ({negative_returns}/{config['eval_episodes']} failed episodes)")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)

    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    total_reward, _, _, q_table = train(env, CONFIG)
    # print()
    # print(f"Total reward over training: {total_reward}\n")
    # print("Q-Table:")
    # print(q_table)
