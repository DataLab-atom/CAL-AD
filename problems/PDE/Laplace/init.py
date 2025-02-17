from prompts.init_algorithm_code import GENERATE_ALG_PROMPT_WODOCS_EN
import numpy as np
import os
from problems.dimon.run import static_main
import torch

#np.random.seed(1234)

func_template = """
import torch
import torch.nn as nn

def static_loss(model, f, f_bc, x, y):
    return loss

def static_forward(model, f, f_bc, x):
    return y_out

def static_init(model, branch1_dim, branch2_dim, trunk_dim):

"""

GENERATE_DESC_PROMPT_EN = """
You are an expert-level algorithm engineer. Please describe the optimization goal in one paragraph.

** Goal:**
Please generate three functions (static_init, static_forward, static_loss) to write the model (don't generate anything other than these three functions) in order to minimize the model's loss in the validation set after training.


### Relevant Code for the Optimization Goal:
```python
class opnn(torch.nn.Module):
    def __init__(self, branch1_dim, branch2_dim, trunk_dim):
        super(opnn, self).__init__()
        static_init(self,branch1_dim, branch2_dim, trunk_dim)

    def forward(self, f, f_bc, x):
        return static_forward(self,f, f_bc, x)

    def loss(self, f, f_bc, x, y):
        return static_loss(self, f, f_bc, x, y)
```
Please return the description content you have written directly.
"""

path_head=os.getcwd()
with open('Laplace\opnn.py','r') as f:
    class_code=f.read()

with open('Laplace/DIMON.tex', 'r', encoding='utf-8') as file:
    docs = file.read()

file_path='generated.py'
description_prompts = GENERATE_DESC_PROMPT_EN
ALGORITHM_NAME="DIMON algorithm"
dec_template=GENERATE_ALG_PROMPT_WODOCS_EN
FUNC_NAME= "opnn"

def reset_generated():
    path='Laplace/generated.py'
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Find the index of the line where the first class is defined
    class_start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith('class '):
            class_start_index = i
            break

    # If a class definition was found, trim the content
    if class_start_index is not None:
        # Keep everything up to but not including the class definition
        trimmed_lines = lines[:class_start_index]
    else:
        # No class found, keep the entire content
        trimmed_lines = lines

    # Write the modified content back to the file or a new file
    with open(path, 'w', encoding='utf-8') as file:
        file.writelines(trimmed_lines)

def check_err(init_eval):

    class opnn(torch.nn.Module):
        def __init__(self, branch1_dim, branch2_dim, trunk_dim):
            super(opnn, self).__init__()
            init_eval.static_init(self,branch1_dim, branch2_dim, trunk_dim)

        def forward(self, f, f_bc, x):
            return init_eval.static_forward(self,f, f_bc, x)
    
        def loss(self, f, f_bc, x, y):
            return init_eval.static_loss(self, f, f_bc, x, y)
    
    PODMode = 10
    num_bc = 68 #204
    dim_br1 = [PODMode*2, 100, 100, 100]
    dim_br2 = [num_bc, 150, 150, 150, 100] #150
    dim_tr = [2, 100, 100, 100]
    mean_abs_err,rel_l2_err=static_main(opnn(dim_br1, dim_br2, dim_tr), 50000, 'cuda')
    mean_abs_err=mean_abs_err.mean()
    rel_l2_err=rel_l2_err.mean()
    assert mean_abs_err<1e-2
    assert rel_l2_err<1e-2