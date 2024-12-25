from utils.llm_client.openai import OpenAIClient
from prompts.init_algorithm_code import GENERATE_ALG_PROMPT_EN
import numpy as np
import importlib
np.random.seed(1234)
import os
import numpy as np

#from reward_functions import RewardModel


with open('problems\dpp_ga\doc.tex', 'r', encoding='utf-8') as file:
    docs = file.read()

func_template ="""
reward_functions=importlib.import_module('problems.dpp_ga.reward_functions')

def run_ga(n_pop: int, n_iter: int, n_inst: int, elite_rate: float, n_decap: int, reward_model: 'RewardModel') -> float:
    '''
    Runs the Genetic Algorithm (GA) for optimization.

    Args:
        n_pop (int): Population size.
        n_iter (int): Number of generations.
        n_inst (int): Number of test instances.
        elite_rate (float): Percentage of elite individuals.
        n_decap (int): Number of decap.
        reward_model (reward_functions.RewardModel): Reward model for scoring the individuals.
    '''
    
    return sum_reward / n_inst
"""


GENERATE_DESC_PROMPT_EN = """
You are an expert-level algorithm engineer. Please describe in one paragraph the optimization goal below.

**Optimization Goal:**
Find a placement scheme for a set of decoupling capacitors, denoted as pi, such that the __call__ function of the RewardModel object reaches its minimum for any probe value.

**Relevant code for the optimization goal:**
```
{code}
```

Please return your description directly."""

with open("problems/dpp_ga/reward_functions.py", "r") as f:
    class_code = f.read()

d = 20
n = 100
not_find = True
############################################################################################
# Parameters
n = 10 # PDN shape
m = 10 # PDN shape
model = 5 # Reward model type
freq_pts = 201 # Number of Frequencies

base_path = "problems/dpp_ga"

test_probe_path = os.path.join(base_path, "test_problems", "test_100_probe.npy")
test_prohibit_path = os.path.join(base_path, "test_problems", "test_100_keepout.npy")
keepout_num_path = os.path.join(base_path, "test_problems", "test_100_keepout_num.npy")

    # Model initialization
reward_functions=importlib.import_module('problems.dpp_ga.reward_functions')
reward_model = reward_functions.RewardModel(base_path, n=n, m=m, model_number=model, freq_pts=freq_pts)

    # File reading
with open(test_probe_path, "rb") as f:
    test_probe = np.load(f)  # shape (test,)

with open(test_prohibit_path, "rb") as f1:
    test_prohibit = np.load(f1)  # shape (test, n_keepout)

with open(keepout_num_path, "rb") as f2:
    keepout_num = np.load(f2)  # shape (test,)


elite_rate = 0.2
n_decap = 20
n_pop = 20

n_inst = 3
n_iter = 5
test_probe = test_probe[0: 3]
test_prohibit = test_prohibit[0: 3]
keepout_num = keepout_num[0: 3]

file_path='generated.py'

#init_eval = importlib.import_module('generated')
#importlib.reload(init_eval)
#avg_reward = init_eval.run_ga(n_pop, n_iter, n_inst, elite_rate, n_decap, reward_model)
##############################################################################################
description_prompts=GENERATE_DESC_PROMPT_EN.format(code  = class_code)
FUNC_NAME="run_ga"
ALGORITHM_NAME="Genetic Algorithm"
dec_template=GENERATE_ALG_PROMPT_EN

def check_err(init_eval):
    avg_reward = init_eval.run_ga(n_pop, n_iter, n_inst, elite_rate, n_decap, reward_model)
        
    print("[*] Average:")
    print(avg_reward)