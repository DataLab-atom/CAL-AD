from prompts.init_algorithm_code import GENERATE_ALG_PROMPT_EN
import inspect
from scipy import spatial
import numpy as np
import importlib
np.random.seed(1234)
import numpy as np


func_template ="""
def search_routine(cal_total_distance,distance_matrix:np.ndarray, start_node: int, pop_size:int=100, 
                    num_iterations:int=1000,alpha:float=1.0, beta:float=1.0) -> np.ndarray:
    '''
    '''
    return best_ind
"""

GENERATE_DESC_PROMPT_EN = """
You are an expert-level algorithm engineer; please describe the optimization objective in one paragraph.

Optimization Objective:
To obtain the optimal path that minimizes the total distance.

Code related to the optimization objective:
```
{code}
```

Please directly return the description content you have written."""



init_algorithm_eval = importlib.import_module('problems.tsp_constructive.init_algorithm_eval')
cal_total_distance = init_algorithm_eval.cal_total_distance


with open('problems/tsp_constructive/init_algorithm_eval.py', 'r') as f:
    class_code = f.read()
    f.close()

description_prompts = GENERATE_DESC_PROMPT_EN.format(code = inspect.getsource(cal_total_distance))
ALGORITHM_NAME="Efficient and Fast Heuristic Algorithms"
dec_template=GENERATE_ALG_PROMPT_EN
FUNC_NAME= "search_routine"
file_path='generated.py'
docs=''

def check_err(init_eval):
    points_coordinate = np.random.rand(50, 2)  # generate coordinate of points
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    best_routine = init_eval.search_routine(cal_total_distance,distance_matrix,0,pop_size=10,num_iterations=100)
    obj = cal_total_distance(best_routine,distance_matrix)
    assert obj > 0 
    assert obj < 1e23
    #avg_reward = init_eval.run_ga(n_pop, n_iter, n_inst, elite_rate, n_decap, reward_model)
    
    #print("[*] Average:")
    #print(avg_reward)