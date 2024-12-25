from numpy.linalg import inv, norm, pinv
from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from dataclasses import dataclass
def search_routine(env: Env, model: Model, episodes: float, batch_size: int = 10, aug_factor: int = 8) -> float:
    """
    Uses a pre-trained model to determine the minimum total distance for the TSP problem,
    incorporating heuristics and multiple restarts for better performance.

    Args:
        env: The TSP environment.
        model: The pre-trained TSP model.
        episodes: Number of episodes to run.
        batch_size: The batch size for processing problems.
        aug_factor: Augmentation factor for data.

    Returns:
        The average total distance of the minimum valid solutions computed over the specified episodes.
    """

    min_distances = []
    for _ in range(int(episodes)):
        env.load_problems(batch_size, aug_factor)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        episode_min_distance = float('inf')
        for _ in range(batch_size):
            done = False
            total_distance = 0
            while not done:
                state, _, _ = env.pre_step()
                selected, _ = model.forward(state)
                step_state, reward, done = env.step(selected)
                total_distance += -reward.item()  # reward is negative distance

            episode_min_distance = min(episode_min_distance, total_distance)

        min_distances.append(episode_min_distance)

    return sum(min_distances) / len(min_distances)
def _run_episode(env: Env, model: Model, batch_size: int, aug_factor: int) -> float:
    """
    Runs a single episode of the TSP problem.

    Args:
        env: The TSP environment.
        model: The pre-trained TSP model.
        batch_size: The batch size for processing problems.
        aug_factor: Augmentation factor for data.

    Returns:
        The minimum travel distance found in the episode.
    """
    env.load_problems(batch_size, aug_factor)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    done = False
    while not done:
        state, _, _ = env.pre_step()
        selected, _ = model.forward(state)
        step_state, reward, done = env.step(selected)

    return -reward.min().item()  # reward is negative distance