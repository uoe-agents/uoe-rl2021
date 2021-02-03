import gym

from rl2021.exercise2.agents import QLearningAgent
from rl2021.exercise2.utils import evaluate

REAL_MAX_EPISODE_STEPS = 100 # CUT OF AN EPISODE THAT RUNS LONGER THAN THAT. DO NOT CHANGE

### TUNE HYPERPARAMETERS HERE ###
CONFIG = {
    "env": "Taxi-v3",
    "total_eps": 10000,
    "eps_max_steps": 100,
    "eval_episodes": 500,
    "eval_freq": 1000,
    "gamma": 0.99,
    "alpha": 0.5,
    "epsilon": 0.0,
}


def q_learning_eval(
        env,
        config,
        q_table,
        max_steps=REAL_MAX_EPISODE_STEPS,
        render=False,
        output=True):
    """
    Evaluate configuration of Q-learning on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param max_steps (int): max number of steps per evaluation episode
    :param eval_episodes (int): number of evaluation episodes
    :param render (bool): flag whether evaluation runs should be rendered
    :param output (bool): flag whether mean evaluation performance should be printed
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=0.0,
    )
    eval_agent.q_table = q_table
    return evaluate(env, eval_agent, max_steps, config["eval_episodes"], render)


def train(env, config, output=True):
    """
    Train and evaluate Q-Learning on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param output (bool): flag if mean evaluation results should be printed
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        total reward over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table
    """
    agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=config["epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []

    for eps_num in range(1, config["total_eps"]+1):
        obs = env.reset()
        episodic_return = 0
        t = 0

        while t < config["eps_max_steps"]:
            agent.schedule_hyperparameters(step_counter, max_steps)
            act = agent.act(obs)
            n_obs, reward, done, _ = env.step(act)
            agent.learn(obs, act, reward, n_obs, done)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = n_obs

        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, negative_returns = q_learning_eval(env, config, agent.q_table)
            print(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return} ({negative_returns}/{config['eval_episodes']} failed episodes)")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)

    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    total_reward, _, _, q_table = train(env, CONFIG)
    # print()
    # print(f"Total reward over training: {total_reward}\n")
    # print("Q-table:")
    # print(q_table)
