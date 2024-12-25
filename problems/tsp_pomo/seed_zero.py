from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
import torch
from dataclasses import dataclass
import numpy as np

@dataclass
class SearchResult:
    total_distance: float

def _run_batch(env: Env, model: Model) -> float:
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    done = False
    while not done:
        state, _, _ = env.pre_step()
        selected, _ = model(state)
        _, reward, done = env.step(selected)

    return reward.mean().item()


def search_routine(env: Env, model: Model, episodes: float, batch_size: int = 10, aug_factor: int = 8) -> float:
    """
    Executes the POMO search routine to find the minimum total distance for the TSP.

    Args:
        env: The TSP environment.
        model: The pre-trained TSP model.
        episodes: Number of episodes to run.
        batch_size: Batch size for processing problems.
        aug_factor: Augmentation factor for data augmentation.

    Returns:
        The total distance of the minimum valid solution.
    """
    model.eval()
    model.to(model.device)  # Ensure model is on the correct device
    total_distance = 0
    num_batches = int(np.ceil(episodes / batch_size))

    with torch.no_grad():
        for _ in range(num_batches):
            env.load_problems(batch_size, aug_factor)
            batch_reward = _run_batch(env, model)
            total_distance += batch_reward * batch_size
    total_distance /= episodes
    return SearchResult(total_distance=-total_distance) # Return positive distance as requested


if __name__ == "__main__":
    # Test code here
    model_params = {'embedding_dim': 128, 'sqrt_embedding_dim': 128**0.5, 'encoder_layer_num': 6,
                    'qkv_dim': 16, 'head_num': 8, 'logit_clipping': 10, 'ff_hidden_dim': 512, 'eval_type': 'softmax'}

    env_params = {'problem_size': 50, 'pomo_size': 100}
    env = Env(**env_params)

    model = Model(**model_params)

    # dummy pre-trained weights - replace with actual pre-trained weights
    for p in model.parameters():
        p.data = torch.randn_like(p)


    episodes = 100
    result = search_routine(env, model, episodes)
    print(f"Total distance: {result.total_distance}")


