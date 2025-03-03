from dataclasses import dataclass
import random
from typing import Callable
import numpy as np
def initialize_pheromones(distance_matrix: np.ndarray, initial_pheromone: float = 1.0) -> np.ndarray:
    """Initialize the pheromone matrix with a constant value."""
    return np.full_like(distance_matrix, initial_pheromone, dtype=float)
def select_next_city(current_city: int, unvisited_cities: np.ndarray, pheromone_matrix: np.ndarray, distance_matrix: np.ndarray, alpha: float, beta: float) -> int:
    """Select the next city based on the pheromone and distance information."""
    pheromone = pheromone_matrix[current_city, unvisited_cities]
    distance = distance_matrix[current_city, unvisited_cities]
    attractiveness = (pheromone ** alpha) * ((1.0 / distance) ** beta)
    probabilities = attractiveness / np.sum(attractiveness)
    next_city = np.random.choice(unvisited_cities, p=probabilities)
    return next_city
def construct_solution(pheromone_matrix: np.ndarray, distance_matrix: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Construct a solution (tour) for a single ant."""
    num_cities = distance_matrix.shape[0]
    start_city = np.random.randint(num_cities)
    tour = [start_city]
    unvisited_cities = np.delete(np.arange(num_cities), start_city)
    
    for _ in range(num_cities - 1):
        current_city = tour[-1]
        next_city = select_next_city(current_city, unvisited_cities, pheromone_matrix, distance_matrix, alpha, beta)
        tour.append(next_city)
        unvisited_cities = np.delete(unvisited_cities, np.where(unvisited_cities == next_city))
    
    return np.array(tour)
def update_pheromones(pheromone_matrix: np.ndarray, tours: np.ndarray, distances: np.ndarray, evaporation_rate: float, Q: float, elitist_weight: float = 2.0) -> np.ndarray:
    """Update the pheromone matrix based on the tours and their distances with an elitist strategy."""
    num_cities = pheromone_matrix.shape[0]
    
    # Evaporation
    pheromone_matrix *= (1.0 - evaporation_rate)
    
    # Pheromone update for each tour
    for tour, distance in zip(tours, distances):
        for i in range(len(tour) - 1):
            current_city = tour[i]
            next_city = tour[i + 1]
            pheromone_matrix[current_city, next_city] += Q / distance
            pheromone_matrix[next_city, current_city] += Q / distance
    
    # Elitist strategy: reinforce the best tour more heavily
    best_tour_index = np.argmin(distances)
    best_tour = tours[best_tour_index]
    best_distance = distances[best_tour_index]
    
    for i in range(len(best_tour) - 1):
        current_city = best_tour[i]
        next_city = best_tour[i + 1]
        pheromone_matrix[current_city, next_city] += elitist_weight * Q / best_distance
        pheromone_matrix[next_city, current_city] += elitist_weight * Q / best_distance
    
    return pheromone_matrix
def search_routine(cal_total_distance: Callable, distance_matrix: np.ndarray, pop_size: int = 100, num_generations: int = 1000, mutation_rate: float = 0.01, alpha: float = 1.0, beta: float = 5.0, evaporation_rate: float = 0.5, Q: float = 100.0) -> np.ndarray:
    """Search for the optimal TSP routine using Ant Colony Optimization."""
    num_cities = distance_matrix.shape[0]
    pheromone_matrix = initialize_pheromones(distance_matrix)
    best_tour = None
    best_distance = np.inf
    
    for generation in range(num_generations):
        tours = np.array([construct_solution(pheromone_matrix, distance_matrix, alpha, beta) for _ in range(pop_size)])
        distances = np.array([cal_total_distance(tour, distance_matrix) for tour in tours])
        
        if np.min(distances) < best_distance:
            best_distance = np.min(distances)
            best_tour = tours[np.argmin(distances)]
        
        pheromone_matrix = update_pheromones(pheromone_matrix, tours, distances, evaporation_rate, Q)
    
    return best_tour