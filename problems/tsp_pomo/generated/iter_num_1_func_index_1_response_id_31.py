from numpy.linalg import inv, norm, pinv
from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from dataclasses import dataclass
import torch
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
    while not done:
        state, _, _ = env.pre_step()
        selected, _ = model.forward(state)
        step_state, reward, done = env.step(selected)

        # Apply heuristic: Nearest Neighbor
        if not done:
            current_node = selected.item()
            remaining_nodes = [i for i in range(env.problem_size) if i not in env.visited_nodes]
            nearest_node = min(remaining_nodes, key=lambda x: env.distance_matrix[current_node, x])
            selected = torch.tensor([nearest_node])
            step_state, reward, done = env.step(selected)

        # Update minimum distance
        min_distance = min(min_distance, -reward.item())

    return min_distance