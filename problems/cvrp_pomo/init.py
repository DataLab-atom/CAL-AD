from prompts.init_algorithm_code import GENERATE_ALG_PROMPT_EN
import inspect
from scipy import spatial
import numpy as np
import importlib
np.random.seed(1234)
import numpy as np

func_template ="""
def search_routine(cal_total_distance,distance_matrix:np.ndarray, pop_size:int=100, num_generations:int=1000, mutation_rate:float=0.01,alpha:float=1.0, beta:float=5.0, evaporation_rate:float=0.5, Q:float=100.0) -> np.ndarray:
    '''
    '''
    return best_ind
"""

GENERATE_DESC_PROMPT_ZH = """你是一个专家级别的算法工程师请你用一段话描述一下优化目标

优化目标：
寻找一个点$x^*$使得QuadraticFunction对象的objective_function函数达到最小
在您的TSP环境中，目标是通过优化路径选择策略来最小化_get_travel_distance函数计算出的旅行总距离。
通过优化路径选择策略来寻找一个step，使得_get_travel_distance函数计算出的旅行总距离达到最小。
优化目标的相关代码:
```
{code}
```

请直接返回你编写的描述内容"""

GENERATE_DESC_PROMPT_EN = """
You are an expert-level algorithm engineer; please describe the optimization objective in one paragraph.

Optimization Objective:
To obtain the optimal path that minimizes the total distance.

Code related to the optimization objective:
```
{code}
```

Please directly return the description content you have written."""


init_algorithm_eval = importlib.import_module('problems.cvrp_pomo.init_algorithm_eval')
class_code = inspect.getsource(init_algorithm_eval)
description_prompts=GENERATE_DESC_PROMPT_EN.format(code = inspect.getsource(init_algorithm_eval.cal_total_distance))
FUNC_NAME= "search_routine"
file_path='generated.py'
ALGORITHM_NAME="POMO algorithm"
dec_template=GENERATE_ALG_PROMPT_EN
docs=''

def check_err(init_eval):
    points_coordinate = np.random.rand(50, 2)  # generate coordinate of points
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    init_eval.search_routine(class_code,distance_matrix,pop_size=10,num_generations=100)

    #avg_reward = init_eval.run_ga(n_pop, n_iter, n_inst, elite_rate, n_decap, reward_model)
        
    #print("[*] Average:")
    #print(avg_reward)