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
                               evaporation_rate: float, Q: float, elitist_factor: float = 0.1) -> np.ndarray:
    """Update the pheromone matrix based on the solutions found by the ants with an elitist strategy."""
    num_points = pheromone_matrix.shape[0]
    pheromone_matrix *= (1.0 - evaporation_rate)
    
    # Find the best solution and its distance
    best_solution_idx = np.argmin(distances)
    best_solution = solutions[best_solution_idx]
    best_distance = distances[best_solution_idx]
    
    # Apply elitist strategy to reinforce the best solution more heavily
    for i in range(num_points - 1):
        pheromone_matrix[best_solution[i], best_solution[i + 1]] += elitist_factor * Q / best_distance
    pheromone_matrix[best_solution[-1], best_solution[0]] += elitist_factor * Q / best_distance
    
    # Update pheromone levels for all solutions
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