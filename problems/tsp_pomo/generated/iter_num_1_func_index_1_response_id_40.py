from numpy.linalg import inv, norm, pinv
from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from dataclasses import dataclass
def search_routine(env: Env, model: Model, episodes: float, batch_size: int = 10, aug_factor: int = 8) -> float:
    """
    Uses a pre-trained model to determine the minimum total distance for the TSP problem.

    Args:
        env: The TSP environment.
        model: The pre-trained TSP model.
        episodes: Number of episodes to run.
        batch_size: The batch size for processing problems.
        aug_factor: Augmentation factor for data.

    Returns:
        The total distance of the minimum valid solution.
    """

    total_distance = 0
    for _ in range(int(episodes)):
        total_distance += _run_episode(env, model, batch_size, aug_factor)

    return total_distance / episodes
def _run_episode(env: Env, model: Model, batch_size: int, aug_factor: int) -> float:
    """
    Runs a single episode of the TSP problem with additional heuristics.

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
    min_distance = float('inf')
    current_path = []

    while not done:
        state, _, _ = env.pre_step()
        selected, _ = model.forward(state)
        step_state, reward, done = env.step(selected)

        # Update the current path
        current_path.append(selected)

        # Calculate the current distance
        current_distance = -reward.item()

        # Update the minimum distance if the current distance is smaller
        if current_distance < min_distance:
            min_distance = current_distance

        # Apply a simple heuristic: Nearest Neighbor
        if len(current_path) < env.problem_size:
            nearest_neighbor = env.get_nearest_neighbor(selected)
            if nearest_neighbor not in current_path:
                selected = nearest_neighbor
                step_state, reward, done = env.step(selected)
                current_path.append(selected)
                current_distance = -reward.item()
                if current_distance < min_distance:
                    min_distance = current_distance

    return min_distance