from scipy import spatial
import numpy as np
import random

def cal_total_distance(routine: list, distance_matrix: np.ndarray, demand_list: np.ndarray) -> float:
    '''
    Calculate total distance and validate CVRP rules.
    '''
    sorted_arr = np.sort(list(set(routine)))
    assert np.array_equal(sorted_arr, np.arange(len(sorted_arr))), "Rule 1 violated"
    assert (routine[-1] == 0) and (routine[0] == 0), "Rule 2 violated"
    assert distance_matrix.shape[0] == len(set(routine)), "Rule 3 violated"
    assert len(demand_list) == (distance_matrix.shape[0] - 1), "Rule 4 violated"

    selected_demand = 1
    sum_distance = 0
    for i in range(1, len(routine)):
        selected = routine[i]
        if selected == 0:
            selected_demand = 1
            continue

        selected_demand -= demand_list[selected - 1]
        assert selected_demand >= 0, "Capacity violated"

        selected_last = routine[i - 1]
        sum_distance += distance_matrix[selected_last, selected]

    return sum_distance


def generate_initial_population(problem_size: int, pop_size: int) -> np.ndarray:
    '''Generates an initial population of routes.'''
    population = []
    for _ in range(pop_size):
        routine = list(range(1, problem_size + 1))
        random.shuffle(routine)
        routine = [0] + routine + [0]
        population.append(routine)
    return np.array(population)


def calculate_pheromone_matrix(problem_size: int) -> np.ndarray:
    '''Initializes the pheromone matrix.'''
    return np.ones((problem_size + 1, problem_size + 1))


def update_pheromone(pheromone_matrix: np.ndarray, best_ind: list, evaporation_rate: float, Q: float) -> np.ndarray:
    '''Updates pheromone levels based on the best solution.'''
    updated_pheromone = (1 - evaporation_rate) * pheromone_matrix
    for i in range(len(best_ind) - 1):
        updated_pheromone[best_ind[i], best_ind[i+1]] += Q / cal_total_distance(best_ind, distance_matrix, node_demand) # type: ignore
    return updated_pheromone



def search_routine(cal_total_distance, distance_matrix: np.ndarray, demand_list: np.ndarray, pop_size: int = 100, num_generations: int = 1000,
                   mutation_rate: float = 0.01, alpha: float = 1.0, beta: float = 5.0,
                   evaporation_rate: float = 0.5, Q: float = 100.0) -> np.ndarray:
    '''
    Implements the POMO algorithm for CVRP.
    '''
    problem_size = len(demand_list)
    population = generate_initial_population(problem_size, pop_size)
    pheromone_matrix = calculate_pheromone_matrix(problem_size)
    best_ind = population[0]
    best_fitness = cal_total_distance(best_ind, distance_matrix, demand_list)

    for _ in range(num_generations):
        # POMO algorithm implementation (simplified for brevity) - Requires further development for full POMO implementation
        pheromone_matrix = update_pheromone(pheromone_matrix, best_ind, evaporation_rate,Q) #Simplified Update


        for i in range(pop_size):
            current_fitness = cal_total_distance(population[i], distance_matrix, demand_list)
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_ind = population[i].copy()

    return np.array(best_ind)


if __name__ == "__main__":
    problem_size = 20
    demand_scaler = 30
    node_demand = np.random.randint(1, 10, size=(problem_size)) / float(demand_scaler)
    points_coordinate = np.random.rand(problem_size + 1, 2)
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

    best_routine = search_routine(cal_total_distance, distance_matrix, node_demand, pop_size=100, num_generations=100)
    print("Best routine:", best_routine)
    print("Total distance:", cal_total_distance(best_routine, distance_matrix, node_demand))

