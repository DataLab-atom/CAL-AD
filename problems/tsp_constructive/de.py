
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
    
    for generation in tqdm(range(num_generations)):
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

names = ["pr264", "pr226", "pr439"]

opt = {
    'ts225': 126643,
    'rat99': 1211,
    'rl1889': 316536,
    'u1817': 57201,
    'd1655': 62128,
    'bier127': 118282,
    'lin318': 42029,
    'eil51': 426,
    'd493': 35002,
    'kroB100': 22141,
    'kroC100': 20749,
    'ch130': 6110,
    'pr299': 48191,
    'fl417': 11861,
    'd657': 48912,
    'kroA150': 26524,
    'fl1577': 22249,
    'u724': 41910,
    'pr264': 49135,
    'pr226': 80369,
    'pr439': 107217
 }


import numpy as np
from scipy.spatial import distance_matrix
from copy import copy
from tqdm import tqdm

def cal_total_distance(routine,distance_matrix):
    next_points = np.roll(routine, -1)
    distances = distance_matrix[routine, next_points]
    return np.sum(distances)

def eval_heuristic(node_positions: np.ndarray, start_node: int) -> float:
    problem_size = node_positions.shape[0]
    # calculate distance matrix
    dist_mat = distance_matrix(node_positions, node_positions)

    # run the heuristic
    solution = search_routine(cal_total_distance,dist_mat,start_node,200,50)
    
    # calculate the length of the tour
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj

for name in names:
    with open(f'E:/all_works/iclr2025/AEL-P-SNE(1)/AEL-P-SNE/problems/tsp_constructive/test/tsplib/{name}.tsp') as f:
        lines = f.readlines()

    # Parse the data
    data = lines[6:-1]
    data = [x.strip().split() for x in data]
    data = [[float(x) for x in row[1:]] for row in data]

    # Scale the data to [0, 1]^2 to align with the training data
    data = np.array(data)
    scale = max(np.max(data, axis=0) - np.min(data, axis=0))
    data = (data - np.min(data, axis=0)) / scale

    # Evaluate the heuristic
    objs = []
    for start_node in range(3):
        obj = eval_heuristic(data, start_node) * scale
        objs.append(obj)
    mean, std = np.mean(objs), np.std(objs)
    print(name)
    print(f"\tObjective: {mean}+-{std}")
    print(f"\tOpt. Gap: {(mean - opt[name]) / opt[name] * 100}%")
    print()
