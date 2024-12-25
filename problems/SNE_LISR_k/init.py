from prompts.init_algorithm_code import GENERATE_ALG_PROMPT_EN
import inspect
import numpy as np
import importlib

np.random.seed(1234)



with open('problems\SNE_LISR_k\doc.tex', 'r', encoding='utf-8') as file:
    docs = file.read()

func_template = """
def search_root(objective_function: callable, x0: np.ndarray, A_list: List[np.ndarray], b_list: List[np.ndarray],
                tol: float = 1e-6, max_iter: int = 1000, k: int = 5) -> np.ndarray:
    '''
    Implements the optimization algorithm to find the minimum of the objective function.
    
    Parameters:
    - objective_function (callable): The objective function to minimize.
    - x0 (np.ndarray): The initial point for the optimization.
    - A_list (List[np.ndarray]): List of all A_i matrices, each of shape (d,d).
    - b_list (List[np.ndarray]): List of all b_i vectors, each of shape (d,).
    - tol (float): Tolerance for convergence. Default is 1e-6.
    - max_iter (int): Maximum number of iterations. Default is 1000.
    - k (int): Parameter used in the optimization algorithm. Default is 5.
    
    Returns:
    - np.ndarray: The point that minimizes the target function.
    '''
    # Placeholder for the actual optimization algorithm
    x = x0  # Start with the initial point
    for _ in range(max_iter):
        # Implement the optimization steps here
        pass  # Placeholder for the optimization logic
        
    return x
"""

GENERATE_DESC_PROMPT_EN = """
You are an expert-level algorithm engineer. Please describe the optimization goal in one paragraph.

**Optimization Goal:**
Find a point \( x^* \) that minimizes the objective function \( f(x) \). The objective function is defined as:

$ f(x) = \\frac{{1}}{{n}} \\sum_{{i=1}}^n \\left( \\frac{{1}}{{2}} x^T A_i x + b_i^T x \\right) $

where \( A_i \) are positive definite matrices and \( b_i \) are vectors. The goal is to determine the optimal point \( x^* \) that achieves the minimum value of this function.

### Relevant Code for the Optimization Goal:
```python
{code}
```

Please return the description content you have written directly.
"""
file_path='generated.py'
quadratic_function=importlib.import_module('problems.SNE_LISR_k.quadratic_function')
objective_function=quadratic_function.objective_function
class_code = inspect.getsource(quadratic_function.objective_function)
ALGORITHM_NAME="LISR-k"
dec_template=GENERATE_ALG_PROMPT_EN

description_prompts=GENERATE_DESC_PROMPT_EN.format(code=class_code)

def check_err(init_eval):
    d = 50
    n = 1000
    for i in range(50):
        for xi in [4,8,12,16]:
            A_list, b_list = quadratic_function.generate_A_b(xi,d,n)
            x0 = np.random.rand(d)
            x_new_0 =  init_eval.search_root(objective_function,x0,A_list, b_list,max_iter=100,k=5)
            x_new =  init_eval.search_root(objective_function,x0,A_list, b_list,max_iter=1000,k=5)
            assert objective_function(x_new_0,A_list, b_list) < objective_function(x0,A_list, b_list)
            assert objective_function(x_new,A_list, b_list) < objective_function(x0,A_list, b_list)
            assert objective_function(x_new,A_list, b_list) != objective_function(x_new_0,A_list, b_list)