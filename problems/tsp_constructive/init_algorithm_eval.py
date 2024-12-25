from scipy import spatial
import numpy as np
num_points = 50

points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

def cal_total_distance(routine,distance_matrix):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points),distance_matrix)
    '''
    expected = np.arange(len(routine))
    sorted_arr = np.sort(routine)
    if not np.array_equal(sorted_arr, expected):
        raise "break tsp rule"
    next_points = np.roll(routine, -1)
    distances = distance_matrix[routine, next_points]
    return np.sum(distances)