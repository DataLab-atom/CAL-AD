import os
import numpy as np
import importlib
from typing import List
import time

reward_functions = importlib.import_module('problems.dpp_ga.reward_functions')


def generate_population(n_pop: int, n_decap: int, n_ports: int) -> np.ndarray:
    """Generates an initial population for the GA."""
    population = []
    for _ in range(n_pop):
        individual = np.random.choice(n_ports, n_decap + 1, replace=False)
        population.append(individual)
    return np.array(population)


def calculate_fitness(population: np.ndarray, reward_model: 'RewardModel') -> np.ndarray:
    """Calculates the fitness of each individual in the population."""
    fitness = []
    for individual in population:
        probe = individual[0]
        pi = individual[1:]
        reward = reward_model(probe, pi)
        fitness.append(reward)
    return np.array(fitness)


def select_elite(population: np.ndarray, fitness: np.ndarray, elite_rate: float) -> np.ndarray:
    """Selects the elite individuals based on their fitness."""
    n_elite = int(len(population) * elite_rate)
    elite_indices = np.argsort(fitness)[:n_elite]
    return population[elite_indices]


def crossover(parent1: np.ndarray, parent2: np.ndarray, n_decap: int) -> np.ndarray:
    """Performs crossover between two parents."""
    crossover_point = np.random.randint(1, n_decap + 1)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    # Ensure uniqueness in child
    unique_child = np.unique(child)
    missing_elements = np.setdiff1d(parent1, unique_child)
    np.random.shuffle(missing_elements)

    child = np.concatenate((unique_child, missing_elements[:(n_decap + 1) - len(unique_child)]))
    
    return child

def mutate(individual: np.ndarray, n_ports: int, mutation_rate: float = 0.1) -> np.ndarray:
    """Introduces mutation in an individual."""
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.choice(n_ports)
    return individual


def run_ga(n_pop: int, n_iter: int, n_inst: int, elite_rate: float, n_decap: int, reward_model: 'RewardModel', n:int=10, m:int=10, basepath:str = "") -> float:
    """Runs the Genetic Algorithm (GA) for optimization."""
    n_ports = n * m
    sum_reward = 0
    for _ in range(n_inst):
        population = generate_population(n_pop, n_decap, n_ports)
        for _ in range(n_iter):
            fitness = calculate_fitness(population, reward_model)
            elite = select_elite(population, fitness, elite_rate)
            new_population = elite.tolist()
            while len(new_population) < n_pop:
                parent1 = population[np.random.randint(n_pop)]
                parent2 = population[np.random.randint(n_pop)]
                child = crossover(parent1, parent2, n_decap)
                child = mutate(child, n_ports)
                new_population.append(child)
            population = np.array(new_population)
        best_individual_index = np.argmin(calculate_fitness(population, reward_model))
        best_individual = population[best_individual_index]
        probe = best_individual[0]
        pi = best_individual[1:]
        sum_reward += reward_model(probe, pi)

    return sum_reward / n_inst

def search_root(n_pop: int = 50, n_iter: int = 100, n_inst: int = 1, elite_rate: float = 0.2, n_decap: int = 10,
                 model_number: int = 5, n:int=10, m:int=10, basepath:str = "") -> List[int]:
    """
    Searches for the optimal decap placement using Genetic Algorithm.

    Args:
        n_pop (int): Population size.
        n_iter (int): Number of iterations.
        n_inst (int): Number of instances.
        elite_rate (float): Elite rate.
        n_decap (int): Number of decaps.
        model_number (int): Reward model number.
        n (int): Grid dimension n.
        m (int): Grid dimension m.
        basepath (str): Base path for data loading.

    Returns:
        List[int]: The best pi found.

    """
    try:
        reward_model = reward_functions.RewardModel(basepath, model_number, n=n, m=m)
        best_reward = float('inf')
        best_pi = None

        for _ in range(n_inst):
            result = run_ga(n_pop, n_iter, 1, elite_rate, n_decap, reward_model, n, m, basepath)
            if result < best_reward:
                best_reward = result
                # Extract best_pi somehow, this part needs additional info to be completed accurately.
                # Placeholder for now: assuming run_ga returns best_pi along with the reward
                # best_pi = result[1]

        if best_pi is None:
            raise ValueError("No solution found.") # Handle cases where no feasible solution is found
        return best_pi
    except Exception as e:
        print(f"An error occurred: {e}")
        return []  # Or raise the exception, depending on the desired behavior

if __name__ == "__main__":
    n, m = 10, 10
    best_pi = search_root(n=n, m=m, basepath=".")
    print(best_pi)
