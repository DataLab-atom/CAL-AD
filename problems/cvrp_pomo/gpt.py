import torch
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    cap = 0.5  # Vehicle capacity
    
    # Calculate demand differences and check demand exceeding capacity
    demand_diff = torch.abs(demands.unsqueeze(-1) - demands)
    demand_exceed = demands.unsqueeze(-1) + demands > 2 * cap
    
    # Add a small value to distance matrix to prevent division by zero
    distance_matrix = distance_matrix + 1e-6  
    
    # Calculate closeness as the inverse of distance
    closeness = 1 / distance_matrix
    closeness[torch.isinf(closeness)] = 0
    
    # Enhanced prioritization by combining demand difference with inverted distance
    heuristic_values = demand_diff * closeness
    heuristic_values[demand_exceed] = -heuristic_values[demand_exceed]  # Penalize exceeding demand constraints
    
    # Balance diversity by introducing a random noise factor
    diversity_factor = 0.2
    noise = torch.randn_like(heuristic_values)  # Generate random noise with the same shape
    heuristic_values = heuristic_values + diversity_factor * noise  # Introduce noise to promote diversity
    
    return heuristic_values
