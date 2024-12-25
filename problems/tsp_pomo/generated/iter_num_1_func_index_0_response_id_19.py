from numpy.linalg import inv, norm, pinv
from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from dataclasses import dataclass
import torch
def _run_episode(env: Env, model: Model) -> torch.Tensor:
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    total_reward = torch.zeros(reset_state.problems.size(0), 1, device=reset_state.problems.device)
    visited_mask = torch.zeros_like(reset_state.problems, dtype=torch.bool)
    current_node = torch.zeros(reset_state.problems.size(0), 1, dtype=torch.long, device=reset_state.problems.device)

    done = False
    while not done:
        state = Step_State(
            BATCH_IDX=torch.arange(reset_state.problems.size(0), device=reset_state.problems.device),
            POMO_IDX=torch.arange(reset_state.problems.size(1), device=reset_state.problems.device),
            current_node=current_node,
            ninf_mask=visited_mask
        )
        selected, log_prob = model(state)
        selected = selected.squeeze(-1)

        # Update visited mask
        visited_mask.scatter_(1, selected.unsqueeze(-1), True)

        # Calculate reward (distance)
        reward = env.calculate_distance(current_node, selected)
        total_reward += reward

        # Update current node
        current_node = selected.unsqueeze(-1)

        # Check if all nodes are visited
        done = visited_mask.all()

    # Return to the starting node
    final_reward = env.calculate_distance(current_node, torch.zeros_like(current_node))
    total_reward += final_reward

    return total_reward
def search_routine(env: Env, model: Model, episodes: float, batch_size: int = 10, aug_factor: int = 8) -> float:
    """
    Executes the POMO algorithm for the TSP.

    Args:
        env: The TSP environment.
        model: The pre-trained TSP model.
        episodes: Number of episodes to run.
        batch_size: Size of the problem batch.
        aug_factor: Augmentation factor for data.

    Returns:
        The total distance of the minimum valid solution.
    """
    model.eval()
    model.to(model.device)  # Ensure model is on the correct device

    env.load_problems(batch_size, aug_factor)
    total_reward = 0

    for _ in range(int(episodes)):
        rewards = _run_episode(env, model)
        total_reward += rewards.mean().item()
        torch.cuda.empty_cache()  # Clear GPU cache

    total_distance = -total_reward / episodes  # Optimization goal is negative reward
    return total_distance