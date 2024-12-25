from dataclasses import dataclass
import random
import numpy as np
def initialize_pheromone_matrix(num_points: int, initial_pheromone: float) -> np.ndarray:
    """Initialize the pheromone matrix with a constant value."""
    return np.full((num_points, num_points), initial_pheromone)
def select_next_node(current_node: int, unvisited_nodes: np.ndarray, pheromone_matrix: np.ndarray, 
                     distance_matrix: np.ndarray, alpha: float, beta: float) -> int:
    """Select the next node based on the ACO probability formula."""
    pheromone = pheromone_matrix[current_node, unvisited_nodes]
    distance = distance_matrix[current_node, unvisited_nodes]
    attractiveness = (pheromone ** alpha) * ((1.0 / distance) ** beta)
    probabilities = attractiveness / np.sum(attractiveness)
    next_node = np.random.choice(unvisited_nodes, p=probabilities)
    return next_node
def construct_solution(start_node: int, pheromone_matrix: np.ndarray, distance_matrix: np.ndarray, 
                       alpha: float, beta: float) -> np.ndarray:
    """Construct a solution for the TSP using the ACO algorithm."""
    num_points = pheromone_matrix.shape[0]
    solution = [start_node]
    unvisited_nodes = np.arange(num_points)
    unvisited_nodes = np.delete(unvisited_nodes, start_node)
    
    current_node = start_node
    while unvisited_nodes.size > 0:
        next_node = select_next_node(current_node, unvisited_nodes, pheromone_matrix, distance_matrix, alpha, beta)
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
def initialize_pheromone_matrix(num_points: int, initial_pheromone: float) -> np.ndarray:
    """Initialize the pheromone matrix with a constant value."""
    return np.full((num_points, num_points), initial_pheromone)