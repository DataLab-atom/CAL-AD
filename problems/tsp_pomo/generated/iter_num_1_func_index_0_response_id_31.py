from numpy.linalg import inv, norm, pinv
from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from dataclasses import dataclass
def search_routine(env: Env, model: Model, episodes: float, batch_size: int = 10, aug_factor: int = 8) -> float:
    """
    Uses a pre-trained model to determine the minimum total distance for the TSP problem,
    incorporating heuristic-based initial solutions and iterative refinement.

    Args:
        env: The TSP environment.
        model: The pre-trained TSP model.
        episodes: Number of episodes to run.
        batch_size: The batch size for processing problems.
        aug_factor: Augmentation factor for data.

    Returns:
        The average total distance of the minimum valid solutions computed over the specified episodes.
    """

    total_distances = []

    for _ in range(int(episodes)):
        env.load_problems(batch_size, aug_factor)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        # Heuristic-based initial solution
        initial_solution = _heuristic_initial_solution(env)
        current_state = env.step(initial_solution)[0]

        # Iterative refinement using the model
        for _ in range(10):  # Refine the solution 10 times
            state, _, _ = env.pre_step()
            selected, _ = model.forward(state)
            current_state, reward, done = env.step(selected)
            if done:
                break

        total_distances.append(-reward.min().item())  # reward is negative distance

    return sum(total_distances) / len(total_distances)
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