import inspect
import ast
from prompts.init_algorithm_code import GENERATE_ALG_PROMPT_EN
from func_timeout import func_set_timeout
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
import importlib
np.random.seed(1234)



# 数据集及其正则化参数
datasets = {
    'a9a': 1e-3,
    'w8a': 1e-4,
    'ijcnn1': 1e-4,
    'mushrooms': 1e-3,
    'phishing': 1e-4,
    'svmguide3': 1e-3,
    'german.numer': 1e-3,
    'splice': 1e-4,
    'covtype': 1e-3
}

with open('problems/logistic_LISR_k/doc.tex', 'r', encoding='utf-8') as file:
    docs = file.read()

func_template = """
def search_root(logistic_loss: callable,logistic_gradient: callable, X: np.ndarray, y: List[np.ndarray], reg_param: float = 1e-3,
                tol: float = 1e-6, max_iter: int = 1000, k: int = 5) -> np.ndarray:
    '''
    Implements the optimization algorithm to find Weight vector of the model that minimizes the logistic_loss.
    
    Parameters:
    - logistic_loss (callable): The logistic_loss to minimize.
    - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    - y (np.ndarray): Label vector of shape (n_samples,), where labels are -1 or 1.
    - reg_param (float): Regularization parameter λ, used to control model complexity and prevent overfitting.
    - tol (float): Tolerance for convergence. Default is 1e-6.
    - max_iter (int): Maximum number of iterations. Default is 1000.
    - k (int): Parameter used in the optimization algorithm. Default is 5.
    
    Returns:
    - np.ndarray: The Weight vector of the model that minimizes the logistic_loss.
    '''
    # Placeholder for the actual optimization algorithm
    for _ in range(max_iter):
        # Implement the optimization steps here
        pass  # Placeholder for the optimization logic
        
    return w
"""

GENERATE_DESC_PROMPT_EN = """
You are an expert-level algorithm engineer. Please describe the optimization goal in one paragraph.

**Optimization Goal:**
Find a Weight vector of the model that minimizes the logistic_loss.

### Relevant Code for the Optimization Goal:
```python
{code}
```

Please return the description content you have written directly.
"""

def extract_functions_from_file(file_path):
    init_eval = importlib.import_module('seedzero')
    importlib.reload(init_eval)
    """
    从给定的Python文件中提取所有函数定义的名字及其源码。
    
    :param file_path: Python文件的路径
    :return: 包含所有函数名及其源码的字典
    """
    
    with open(file_path, "r", encoding="utf-8") as file:
        # 读取文件内容
        source_code = file.read()
        # 解析成抽象语法树
        tree = ast.parse(source_code, filename=file_path)

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if 'objective_function' in node.name :
                continue
            if 'logistic_loss' in node.name :
                continue
            if 'logistic_gradient' in node.name :
                continue
            function_handl = getattr(init_eval,node.name)
            function_source = inspect.getsource(function_handl)
            functions.append({'func_name':node.name,'func_source':'```python\n' + function_source + '\n```'})
    
    return functions


#from quadratic_function import logistic_loss ,logistic_gradient
quadratic_function=importlib.import_module('problems.logistic_LISR_k.quadratic_function')
class_code = inspect.getsource(quadratic_function.logistic_loss)
# class_code = inspect.getsource(logistic_gradient)

X, y = load_svmlight_file('problems/logistic_LISR_k/a9a')  # 根据需要选择数据集
X = X.toarray()  # 如果数据是稀疏格式，则转换为稠密数组
            
# 将标签从{0, 1}转换为{-1, 1}
y = 2 * y - 1
            
# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

file_path = 'seedzero.py'

@func_set_timeout(120)
def test_func():
    init_eval = importlib.import_module('problems.logistic_LISR_k.seedzero')
    # 强制重新加载以确保使用最新代码
    importlib.reload(init_eval)
    search_root = init_eval.search_root
    # print(GGA[0].message.content,end='\n'*6)
    print("searching",end='\n'*6)
    w_new_0 =  search_root(quadratic_function.logistic_loss,quadratic_function.logistic_gradient,X,y,datasets['a9a'],max_iter=10,k=5)
    w_new =  search_root(quadratic_function.logistic_loss,quadratic_function.logistic_gradient,X,y,datasets['a9a'],max_iter=1000,k=5)
    assert quadratic_function.logistic_loss(X,y, w_new,datasets['a9a']) != quadratic_function.logistic_loss(X,y, w_new_0,datasets['a9a'])
    # 使用这个函数来获取指定文件中的所有函数及其源码
    file_path = 'problems/logistic_LISR_k/seedzero.py'  # 将此处替换为你的Python文件路径
    functions = extract_functions_from_file(file_path)
    assert len(functions) > 3

ALGORITHM_NAME="**LISR-k**"      
description_prompts = GENERATE_DESC_PROMPT_EN.format(code  = class_code)
dec_template=GENERATE_ALG_PROMPT_EN
def check_err(init_eval):
    test_func()