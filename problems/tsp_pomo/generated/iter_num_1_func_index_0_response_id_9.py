from numpy.linalg import inv, norm, pinv
from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from dataclasses import dataclass
import torch
def _run_episode(env: Env, model: Model) -> torch.Tensor:
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    total_reward = 0
    done = False
    while not done:
        state, _, _ = env.pre_step()
        selected, log_prob = model(state)
        step_state, reward, done = env.step(selected)
        total_reward += reward
        
        # Apply some heuristics to guide the model
        if step_state.current_node is not None:
            heuristic_mask = heuristic_function(step_state.current_node, env.problems)
            state.ninf_mask = torch.logical_or(state.ninf_mask, heuristic_mask)

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