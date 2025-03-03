from dataclasses import dataclass
import random
from typing import Callable
from typing import Tuple
import numpy as np
def initialize_population(num_points: int, pop_size: int, start_node: int) -> np.ndarray:
    """
    Initialize a population of routines for the TSP.
    
    Parameters:
    - num_points: Number of points in the TSP.
    - pop_size: Size of the population.
    - start_node: The starting node of the routine.
    
    Returns:
    - population: A 2D numpy array where each row is a routine.
    """
    population = []
    for _ in range(pop_size):
        routine = np.random.permutation(num_points)
        routine[routine == start_node] = routine[0]
        routine[0] = start_node
        population.append(routine)
    return np.array(population)
def evaluate_population(population: np.ndarray, distance_matrix: np.ndarray, cal_total_distance: Callable) -> np.ndarray:
    """
    Evaluate the total distance for each routine in the population using vectorized operations for efficiency.
    
    Parameters:
    - population: A 2D numpy array where each row is a routine.
    - distance_matrix: The distance matrix for the TSP.
    - cal_total_distance: The function to calculate the total distance.
    
    Returns:
    - fitness: A 1D numpy array with the fitness (total distance) of each routine.
    """
    num_routines, num_points = population.shape
    
    # Create a matrix where each row is the routine shifted by one position to the left
    next_points = np.roll(population, -1, axis=1)
    
    # Use broadcasting to get the distances for all routines at once
    distances = distance_matrix[population, next_points]
    
    # Sum the distances along each row to get the total distance for each routine
    fitness = np.sum(distances, axis=1)
    
    return fitness
def select_parents(population: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select parents for crossover based on fitness.
    
    Parameters:
    - population: A 2D numpy array where each row is a routine.
    - fitness: A 1D numpy array with the fitness (total distance) of each routine.
    
    Returns:
    - parents: A tuple of two numpy arrays, each containing selected parents.
    """
    sorted_indices = np.argsort(fitness)
    parents = (population[sorted_indices[0]], population[sorted_indices[1]])
    return parents
def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """
    Perform crossover between two parents to generate a child.
    
    Parameters:
    - parent1: The first parent routine.
    - parent2: The second parent routine.
    
    Returns:
    - child: The child routine generated from crossover.
    """
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child = np.zeros_like(parent1)
    child[:crossover_point] = parent1[:crossover_point]
    for i in range(len(parent2)):
        if parent2[i] not in child:
            for j in range(len(child)):
                if child[j] == 0:
                    child[j] = parent2[i]
                    break
    return child
def mutate(routine: np.ndarray, mutation_rate: float = 0.01) -> np.ndarray:
    """
    Mutate a routine by swapping two random points.
    
    Parameters:
    - routine: The routine to mutate.
    - mutation_rate: The probability of mutation.
    
    Returns:
    - mutated_routine: The mutated routine.
    """
    mutated_routine = routine.copy()
    for i in range(len(routine)):
        if np.random.rand() < mutation_rate:
            swap_with = np.random.randint(len(routine))
            mutated_routine[i], mutated_routine[swap_with] = mutated_routine[swap_with], mutated_routine[i]
    return mutated_routine
def search_routine(cal_total_distance: Callable, distance_matrix: np.ndarray, start_node: int, pop_size: int = 100, 
                    num_iterations: int = 1000, alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """
    Search for the optimal routine using a heuristic algorithm.
    
    Parameters:
    - cal_total_distance: The function to calculate the total distance.
    - distance_matrix: The distance matrix for the TSP.
    - start_node: The starting node of the routine.
    - pop_size: Size of the population.
    - num_iterations: Number of iterations to run the algorithm.
    - alpha: Parameter for controlling the selection pressure.
    - beta: Parameter for controlling the mutation rate.
    
    Returns:
    - best_ind: The best routine found.
    """
    num_points = distance_matrix.shape[0]
    population = initialize_population(num_points, pop_size, start_node)
    
    for _ in range(num_iterations):
        fitness = evaluate_population(population, distance_matrix, cal_total_distance)
        parents = select_parents(population, fitness)
        child = crossover(parents[0], parents[1])
        child = mutate(child, mutation_rate=beta)
        population[np.argmax(fitness)] = child
    
    fitness = evaluate_population(population, distance_matrix, cal_total_distance)
    best_ind = population[np.argmin(fitness)]
    return best_ind