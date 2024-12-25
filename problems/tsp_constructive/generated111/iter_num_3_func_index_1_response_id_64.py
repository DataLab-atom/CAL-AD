from dataclasses import dataclass
import random
import numpy as np
def initialize_pheromone_matrix(num_points: int, initial_pheromone: float) -> np.ndarray:
    """Initialize the pheromone matrix with a constant value."""
    return np.full((num_points, num_points), initial_pheromone)
def select_next_node(current_node: int, unvisited_nodes: np.ndarray, pheromone_matrix: np.ndarray, 
                       distance_matrix: np.ndarray, alpha: float, beta: float) -> int:
    """Select the next node based on the ACO probability formula with dynamic tuning and controlled randomness."""
    pheromone = pheromone_matrix[current_node, unvisited_nodes]
    distance = distance_matrix[current_node, unvisited_nodes]
    
    # Dynamic tuning of alpha and beta
    exploration_bias = np.random.uniform(0.5, 1.5)
    alpha *= exploration_bias
    beta /= exploration_bias
    
    attractiveness = (pheromone ** alpha) * ((1.0 / distance) ** beta)
    probabilities = attractiveness / np.sum(attractiveness)
    
    # Introduce controlled randomness
    if np.random.rand() < 0.05:  # 5% chance to select a random node
        next_node = np.random.choice(unvisited_nodes)
    else:
        next_node = np.random.choice(unvisited_nodes, p=probabilities)
    
    return next_node
def construct_solution(start_node: int, pheromone_matrix: np.ndarray, distance_matrix: np.ndarray, 
                          alpha: float, beta: float) -> np.ndarray:
    """Construct a solution for the TSP using the ACO algorithm with optimized heuristics."""
    num_points = pheromone_matrix.shape[0]
    solution = [start_node]
    unvisited_nodes = np.arange(num_points)
    unvisited_nodes = np.delete(unvisited_nodes, start_node)
    
    current_node = start_node
    while unvisited_nodes.size > 0:
        # Calculate the attractiveness with an additional heuristic based on node centrality
        pheromone = pheromone_matrix[current_node, unvisited_nodes]
        distance = distance_matrix[current_node, unvisited_nodes]
        centrality = np.sum(distance_matrix[:, unvisited_nodes], axis=0) / (num_points - 1)
        attractiveness = (pheromone ** alpha) * ((1.0 / distance) ** beta) * (1.0 / centrality)
        probabilities = attractiveness / np.sum(attractiveness)
        
        # Select the next node based on the enhanced probabilities
        next_node = np.random.choice(unvisited_nodes, p=probabilities)
        solution.append(next_node)
        unvisited_nodes = np.delete(unvisited_nodes, np.where(unvisited_nodes == next_node))
        current_node = next_node
    
    return np.array(solution)
def update_pheromone_matrix(pheromone_matrix: np.ndarray, solutions: np.ndarray, distances: np.ndarray, 
                            evaporation_rate: float, Q: float) -> np.ndarray:
    """Update the pheromone matrix based on the solutions found by the ants."""
    num_points = pheromone_matrix.shape[0]
    pheromone_matrix *= (1.0 - evaporation_rate)
    
    for solution, distance in zip(solutions, distances):
        for i in range(num_points - 1):
            pheromone_matrix[solution[i], solution[i + 1]] += Q / distance
        pheromone_matrix[solution[-1], solution[0]] += Q / distance
    
    return pheromone_matrix
def search_routine(cal_total_distance, distance_matrix: np.ndarray, start_node: int, pop_size: int = 100, 
                   num_generations: int = 1000, mutation_rate: float = 0.01, alpha: float = 1.0, 
                   beta: float = 5.0, evaporation_rate: float = 0.5, Q: float = 100.0) -> np.ndarray:
    """Search for the optimal routine using the Ant Colony Optimization algorithm."""
    num_points = distance_matrix.shape[0]
    initial_pheromone = 1.0 / (num_points * np.mean(distance_matrix))
    pheromone_matrix = initialize_pheromone_matrix(num_points, initial_pheromone)
    
    best_solution = None
    best_distance = float('inf')
    
    for generation in range(num_generations):
        solutions = []
        distances = []
        
        for _ in range(pop_size):
            solution = construct_solution(start_node, pheromone_matrix, distance_matrix, alpha, beta)
            distance = cal_total_distance(solution, distance_matrix)
            solutions.append(solution)
            distances.append(distance)
            
            if distance < best_distance:
                best_distance = distance
                best_solution = solution
        
        pheromone_matrix = update_pheromone_matrix(pheromone_matrix, np.array(solutions), np.array(distances), 
                                                   evaporation_rate, Q)
    
    return best_solution