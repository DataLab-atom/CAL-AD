from dataclasses import dataclass
from reward_functions import RewardModel
import random
from typing import List
from typing import Tuple
import numpy as np
def run_ga(n_pop: int, n_iter: int, n_inst: int, elite_rate: float, n_decap: int, reward_model: 'RewardModel') -> float:
    '''
    Runs the Genetic Algorithm (GA) for optimization.

    Args:
        n_pop (int): Population size.
        n_iter (int): Number of generations.
        n_inst (int): Number of test instances.
        elite_rate (float): Percentage of elite individuals.
        n_decap (int): Number of decap.
        reward_model (RewardModel): Reward model for scoring the individuals.
    '''
    sum_reward = 0.0
    for _ in range(n_inst):
        population = initialize_population(n_pop, n_decap, reward_model.n * reward_model.m)
        for _ in range(n_iter):
            population = evolve_population(population, reward_model, elite_rate)
        best_individual = min(population, key=lambda x: x[1])
        sum_reward += best_individual[1]
    return sum_reward / n_inst
def initialize_population(n_pop: int, n_decap: int, total_ports: int) -> List[Tuple[np.ndarray, float]]:
    '''
    Initializes the population with random individuals.

    Args:
        n_pop (int): Population size.
        n_decap (int): Number of decap.
        total_ports (int): Total number of ports.

    Returns:
        List[Tuple[np.ndarray, float]]: List of individuals with their fitness values.
    '''
    population = []
    for _ in range(n_pop):
        pi = np.random.choice(total_ports, n_decap, replace=False)
        probe = random.randint(0, total_ports - 1)
        population.append((pi, probe))
    return population
def evolve_population(population: List[Tuple[np.ndarray, float]], reward_model: 'RewardModel', elite_rate: float) -> List[Tuple[np.ndarray, float]]:
    '''
    Evolves the population by selecting, mating, and mutating individuals with enhanced strategies.

    Args:
        population (List[Tuple[np.ndarray, float]]): Current population.
        reward_model (RewardModel): Reward model for scoring the individuals.
        elite_rate (float): Percentage of elite individuals.

    Returns:
        List[Tuple[np.ndarray, float]]: New population after evolution.
    '''
    population = evaluate_population(population, reward_model)
    elite_count = int(elite_rate * len(population))
    population.sort(key=lambda x: x[1])
    elites = population[:elite_count]
    new_population = elites.copy()
    
    # Enhanced crossover and mutation strategies
    while len(new_population) < len(population):
        parent1, parent2 = random.sample(elites, 2)
        
        # Enhanced crossover: blend crossover
        child1, child2 = blend_crossover(parent1, parent2, alpha=0.5)
        
        # Enhanced mutation: non-uniform mutation
        child1 = non_uniform_mutate(child1, reward_model.n * reward_model.m, generation=len(new_population), max_generation=len(population))
        child2 = non_uniform_mutate(child2, reward_model.n * reward_model.m, generation=len(new_population), max_generation=len(population))
        
        new_population.append(child1)
        new_population.append(child2)
    
    return new_population[:len(population)]
def evaluate_population(population: List[Tuple[np.ndarray, float]], reward_model: 'RewardModel') -> List[Tuple[np.ndarray, float]]:
    '''
    Evaluates the population by calculating the fitness of each individual.

    Args:
        population (List[Tuple[np.ndarray, float]]): Current population.
        reward_model (RewardModel): Reward model for scoring the individuals.

    Returns:
        List[Tuple[np.ndarray, float]]: Population with fitness values.
    '''
    for i in range(len(population)):
        pi, probe = population[i]
        fitness = reward_model(probe, pi)
        population[i] = (pi, fitness)
    return population
def crossover(parent1: Tuple[np.ndarray, int], parent2: Tuple[np.ndarray, int]) -> Tuple[Tuple[np.ndarray, int], Tuple[np.ndarray, int]]:
    '''
    Performs crossover between two parents to produce two children.

    Args:
        parent1 (Tuple[np.ndarray, int]): First parent.
        parent2 (Tuple[np.ndarray, int]): Second parent.

    Returns:
        Tuple[Tuple[np.ndarray, int], Tuple[np.ndarray, int]]: Two children.
    '''
    pi1, probe1 = parent1
    pi2, probe2 = parent2
    split_point = random.randint(1, len(pi1) - 1)
    child1_pi = np.concatenate((pi1[:split_point], pi2[split_point:]))
    child2_pi = np.concatenate((pi2[:split_point], pi1[split_point:]))
    child1_probe = probe1 if random.random() < 0.5 else probe2
    child2_probe = probe2 if random.random() < 0.5 else probe1
    return (child1_pi, child1_probe), (child2_pi, child2_probe)
def mutate(individual: Tuple[np.ndarray, int], total_ports: int) -> Tuple[np.ndarray, int]:
    '''
    Mutates an individual by randomly changing some of its genes.

    Args:
        individual (Tuple[np.ndarray, int]): Individual to mutate.
        total_ports (int): Total number of ports.

    Returns:
        Tuple[np.ndarray, int]: Mutated individual.
    '''
    pi, probe = individual
    mutation_rate = 0.1
    for i in range(len(pi)):
        if random.random() < mutation_rate:
            pi[i] = random.randint(0, total_ports - 1)
    if random.random() < mutation_rate:
        probe = random.randint(0, total_ports - 1)
    return (pi, probe)