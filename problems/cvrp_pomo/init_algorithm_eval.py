from scipy import spatial
import numpy as np


def cal_total_distance(routine,distance_matrix, demand_list):
    '''
    Calculate the total distance of a given route (routine) and check if it satisfies the basic rules of the Capacitated Vehicle Routing Problem (CVRP).

    Parameters:
    - routine (list or numpy.ndarray): The sequence of nodes visited in the route, including the depot. The depot is represented by 0.
    - distance_matrix (numpy.ndarray): A 2D array where `distance_matrix[i, j]` represents the distance between node i and node j.
    - demand_list (list or numpy.ndarray): A list of demands for each node, excluding the depot.

    Returns:
    - float: The total distance of the route.

    Raises:
    - AssertionError: If the route does not satisfy the CVRP rules.
    '''
    sorted_arr = np.sort(list(set(routine)))
    assert  np.array_equal(sorted_arr, np.arange(len(sorted_arr))),"break cvrp rule1"
    assert (routine[-1] == 0) and (routine[0] == 0),"break cvrp rule2"
    assert distance_matrix.shape[0] == len(set(routine)),"break cvrp rule3"
    assert len(demand_list) == (distance_matrix.shape[0] - 1),"break cvrp rule4"
    
    selected_demand = 1
    sum_distance = 0
    for i in range(1,len(routine)):
        selected = routine[i]
        if selected == 0:
            selected_demand = 1
            continue

        selected_demand -= demand_list[selected-1]
        assert selected_demand  

        selected_last = routine[i-1]
        sum_distance += distance_matrix[selected_last,selected]     

    return sum_distance

if __name__ == "__main__": 
    problem_size = 200
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200:
        demand_scaler = 80
    elif problem_size == 500:
        demand_scaler = 100
    elif problem_size == 1000:
        demand_scaler = 250
    elif problem_size == 5000:
        demand_scaler = 500
    else:
        raise NotImplementedError

    node_demand = np.random.randint(1, 10, size=(problem_size)) / float(demand_scaler)
    points_coordinate = np.random.rand(problem_size+1, 2)  # generate coordinate of points
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    routine = [0]  # Start at the depot
    remaining_nodes = list(range(1, problem_size+1))
    np.random.shuffle(remaining_nodes)  # Shuffle the nodes to create a random order
    routine.extend(remaining_nodes)
    routine.append(0)  # End at the depot

    # Calculate the total distance
    try:
        total_distance = cal_total_distance(routine, distance_matrix, node_demand)
        print(f"Total Distance: {total_distance}")
    except AssertionError as e:
        print(f"Assertion Error: {e}")