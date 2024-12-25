GENERATE_DESC_PROMPT_ZH = """你是一个专家级别的算法工程师请你用一段话描述一下优化目标

优化目标：
获取最优的路径使得总距离最小

优化目标的相关代码:
```
{code}
```

请直接返回你编写的描述内容"""

GENERATE_GA_PROMPT_ZH="""你是一个专家级别的算法工程师，请你使用{algorithm_name}编写一个名为search_routine的python函数达到优化目标
search_routine函数模板:
def search_routine(cal_total_distance,distance_matrix:np.ndarray, pop_size:int=100, num_generations:int=1000, mutation_rate:float=0.01,alpha:float=1.0, beta:float=5.0, evaporation_rate:float=0.5, Q:float=100.0) -> np.ndarray:
    '''
    '''
    return best_ind

优化目标：
{description}

优化目标的相关代码:
```
{code}
```

如果需要定义额外的帮助函数,请确保它们在`search_routine`函数之外定义,并正确地被调用。
如果有测试代码请放在下方代码块中:
```
if __name__ == "__main__":
```
请为所有函数的输入和输出添加类型注解。
请直接返回你编写的所有代码。"""

GENERATE_ALG_PROMPT_ZH="""你是一个专家级别的算法工程师，请你参考给定资料使用python编写算法{algorithm_name}的实现，并以一个名为search_root的函数作为入口达到优化目标

search_root的函数模板
{func_template}

如果你需要修改此模板以增加新的输入,请为它设置好默认值,并放置到参数列表的最后方。

参考资料:
{docs}

优化目标：
{description}

优化目标的相关代码:
```
{code}
```

针对数值计算场景,应该保证计算的精度。
应该避免在一个函数体中编写过多代码,需要多定义工具函数。
确保所有函数的函数体都独立的在`search_root`函数体之外,且正确地被调用。
如果有测试代码请放在下方代码块中:
```
if __name__ == "__main__":
```
请为所有函数的输入和输出添加类型注解。
请直接返回你编写的所有代码。"""


FUNC_DESC_PROMPT_ZH = """你是一个专家级别的算法工程师,现有一份markdown文档,其中包含了python代码与相关说明,现从中选取了一个目标函数`{func_name}`请你用一段话描述这个函数的输入输出以及作用

目标函数代码：
{func_source}

markdown文档:
```
{doc}
```

请直接返回你编写的描述内容"""

#-------------------------------------------------------------------------------------------------------------------------------------------------------
GENERATE_DESC_PROMPT_EN = """You are an expert-level algorithm engineer; please describe the optimization objective in one paragraph.

Optimization Objective:
To obtain the optimal path that minimizes the total distance.

Code related to the optimization objective:
```
{code}
```

Please directly return the description content you have written."""

GENERATE_GA_PROMPT_EN = """You are an expert-level algorithm engineer. Please write a Python function named `search_routine` using a {algorithm_name} to achieve optimization objective.
`search_routine` function template:
def search_routine(cal_total_distance,distance_matrix:np.ndarray, pop_size:int=100, num_generations:int=1000, mutation_rate:float=0.01,alpha:float=1.0, beta:float=5.0, evaporation_rate:float=0.5, Q:float=100.0) -> np.ndarray:
    '''
    '''
    return best_ind

Optimization Objective:
{description}

Code related to the optimization objective:
```
{code}
```

If there is test code, please place it in the following code block:
```
if __name__ == "__main__":
```

Prohibit defining any functions inside the search_routine,if additional helper functions need to be defined, please ensure they are defined outside of the function and are correctly called.
Please add type annotations for the inputs and outputs of all functions.
Please directly return all the code you have written."""

FUNC_DESC_PROMPT_EN = """You are an expert-level algorithm engineer. There is a Markdown document that contains Python code along with relevant explanations. A target function `{func_name}` has been selected from this document. Please describe in one paragraph the inputs, outputs, and the purpose of this function.

Target function code:
{func_source}

markdown document:
```
{doc}
```

Please directly return the description content you have written."""

GENERATE_ALG_PROMPT_EN="""

You are an expert-level algorithm engineer.Please refer to the provided materials and implement the algorithm `{algorithm_name}` using Python, ensuring that the entry function is named search_root. The implementation must strictly adhere to the specified optimization goal and should include comprehensive error handling, efficient performance, and clear documentation. Additionally, the code should be thoroughly tested to ensure correctness and robustness.

**Coding Rules:**
* For numerical computation scenarios, the precision of calculations should be guaranteed.
* Keep the number of lines of code in each function body to 15 lines or less.
* More utility functions should be defined.
* Ensure that all functions are independent, at the same level as the `search_root` function, and are correctly called.
* If you need to modify this template to add new inputs, please set default values for them and place them at the end of the parameter list.
* If there is test code, please place it in the following code block:
```python
if __name__ == "__main__":
    # Test code here
```
* Please add type annotations for all inputs and outputs of the functions.
* Please return all the code you have written directly.

**Optimization Goal:**
{description}

**Relevant code for the optimization goal:**
```
{code}
```

**Reference Materials:**
{docs}

**Function template for `search_root`:**
```python
{func_template}
```
"""


GENERATE_ALG_PROMPT_WODOCS_EN="""

You are an expert-level algorithm engineer.Please refer to the provided materials and implement the algorithm `{algorithm_name}` using Python, ensuring that the entry function is named `{func_name}`. 
The implementation must strictly adhere to the specified optimization goal and should include comprehensive error handling, efficient performance, and clear documentation. 
Additionally, the code should be thoroughly tested to ensure correctness and robustness.

**Coding Rules:**
* For numerical computation scenarios, the precision of calculations should be guaranteed.
* Keep the number of lines of code in each function body to 15 lines or less.
* More utility functions should be defined.
* Ensure that all functions are independent, at the same level as the `{func_name}` function, and are correctly called.
* If you need to modify this template to add new inputs, please set default values for them and place them at the end of the parameter list.
* If there is test code, please place it in the following code block:
```python
if __name__ == "__main__":
    # Test code here
```
* Please add type annotations for all inputs and outputs of the functions.
* Please return all the code you have written directly.

**Optimization Goal:**
{description}

**Relevant code for the optimization goal:**
```
{code}
```

**Function template for `{func_name}`:**
```python
{func_template}
```
"""



BASE_URL='https://api.agicto.cn/v1'
API_KEY='sk-d4nQkPsuNRkzHZ8vt8ag4Npu1sIOdRwdtV5OhOuUT4GUJh5L'
LLM_MODEL='gpt-4o-mini'
LLM_CODE_MODEL='gemini-1.5-pro'