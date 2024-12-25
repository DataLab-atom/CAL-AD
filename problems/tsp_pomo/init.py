from prompts.init_algorithm_code import GENERATE_ALG_PROMPT_WODOCS_EN
import numpy as np
import os
np.random.seed(1234)

func_template = """
from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

def search_routine(env: Env, model : Model,episodes:float,batch_size:int=10,aug_factor:int=8) -> float:
    return total_distance
"""

GENERATE_DESC_PROMPT_EN = """
You are an expert-level algorithm engineer. Please describe the optimization goal in one paragraph.

**Optimization Goal:**
Use the given pre-trained MODEL object to find the total distance of the minimum valid solution for the TSP problem stored in the given ENV object.

### Relevant Code for the Optimization Goal:
```python
{code}
```

Please return the description content you have written directly.
"""

path_head=os.getcwd()
with open(f'{path_head}\problems\\tsp_pomo\TSPEnv.py','r') as f:
    class_code=f.read()

with open(f'{path_head}\problems\\tsp_pomo\TSPModel.py','r') as f:
    class_code+=f.read()


from problems.tsp_pomo.eval import get_testpair
env,model,episodes = get_testpair()

file_path='seed_zero.py'
description_prompts = GENERATE_DESC_PROMPT_EN.format(code = class_code)
ALGORITHM_NAME="POMO algorithm"
dec_template=GENERATE_ALG_PROMPT_WODOCS_EN
FUNC_NAME= "search_routine"
def check_err(init_eval):
    obj = init_eval.search_routine(env,model,episodes,10)
    assert obj < 1e23
    assert obj > 0