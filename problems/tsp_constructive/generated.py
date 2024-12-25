from scipy import spatial
import numpy as np
import random

num_points = 50

points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

def cal_total_distance(routine:np.ndarray, distance_matrix:np.ndarray) -> float:
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points),distance_matrix)
    '''
    expected = np.arange(len(routine))
    sorted_arr = np.sort(routine)
    if not np.array_equal(sorted_arr, expected):
        raise ValueError("Invalid routine: Not all points are visited exactly once.")
    next_points = np.roll(routine, -1)
    distances = distance_matrix[routine, next_points]
    return np.sum(distances)


def generate_neighbor(routine: np.ndarray) -> np.ndarray:
    """Generates a neighbor solution by swapping two random cities."""
    i, j = random.sample(range(len(routine)), 2)
    neighbor = routine.copy()
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def acceptance_probability(delta_cost: float, temperature: float) -> float:
    """Calculates the acceptance probability based on the Metropolis criterion."""
    return np.exp(-delta_cost / temperature)


def simulated_annealing(distance_matrix: np.ndarray, start_node: int, num_iterations: int, 
                       alpha: float, beta: float) -> np.ndarray:
    """Performs simulated annealing to find a near-optimal solution."""
    num_points = distance_matrix.shape[0]
    current_solution = np.arange(num_points)
    current_cost = cal_total_distance(current_solution, distance_matrix)
    best_solution = current_solution.copy()
    best_cost = current_cost
    temperature = alpha
    for _ in range(num_iterations):
        neighbor = generate_neighbor(current_solution)
        neighbor_cost = cal_total_distance(neighbor, distance_matrix)
        delta_cost = neighbor_cost - current_cost
        if delta_cost < 0 or random.random() < acceptance_probability(delta_cost, temperature):
            current_solution = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
        temperature *= beta  # Cooling schedule
    return best_solution


def search_routine(cal_total_distance, distance_matrix: np.ndarray, start_node: int = 0, pop_size: int = 100,
                    num_iterations: int = 1000, alpha: float = 1.0, beta: float = 0.99) -> np.ndarray:
    """Searches for the optimal route using simulated annealing."""
    best_ind = simulated_annealing(distance_matrix, start_node, num_iterations, alpha, beta)
    return best_ind



if __name__ == "__main__":
    # Test code here
    best_route = search_routine(cal_total_distance, distance_matrix, start_node=0)
    best_distance = cal_total_distance(best_route, distance_matrix)
    print(f"Best route: {best_route}")
    print(f"Best distance: {best_distance}")


