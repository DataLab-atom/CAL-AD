from utils.llm_client.openai import OpenAIClient
from prompts.init_algorithm_code import FUNC_DESC_PROMPT_EN,BASE_URL,LLM_MODEL,LLM_CODE_MODEL,API_KEY
import inspect
import ast
import re
import numpy as np
import sys
import importlib
import os
import argparse

np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, help='Evolutionary problem')
args = parser.parse_args()

path_head=os.getcwd().replace('\\','/')
problem = args.problem
problem_init = importlib.import_module(f'problems.{problem}.init')

llm = OpenAIClient(LLM_MODEL,0.5,base_url=BASE_URL,api_key=API_KEY)
llm_code = OpenAIClient(LLM_CODE_MODEL,0.5,base_url=BASE_URL,api_key=API_KEY)

import traceback
def set_code_to_file(content):
    for pattern in [r'“““python(.*?)”””',r'```python(.*?)```',r'```python\n"(.*?)"\n```']:
        rcontent = re.findall(pattern, content, re.DOTALL)
        if len(rcontent) == 0:
            continue
        with open(f'{path_head}/problems/{problem}/{problem_init.file_path}','w',encoding='utf-8') as code_file:
            code_file.write(rcontent[0].lstrip())
            code_file.close()
        break
    assert rcontent != 0,"code not find in response"


not_find = True
while not_find :
    description_prompts = problem_init.description_prompts
    description = llm._chat_completion_api(messages  = [{"role": "user", "content": description_prompts}],temperature=1)[0].message.content
    print(description,end='\n'*6)
    if hasattr(problem_init,'docs'):
        if hasattr(problem_init,'FUNC_NAME'):
            GGA_prompts = problem_init.dec_template.format(algorithm_name=problem_init.ALGORITHM_NAME,func_template=problem_init.func_template,docs = problem_init.docs,description=description,func_name = problem_init.FUNC_NAME,code=problem_init.class_code)
        else:
            GGA_prompts = problem_init.dec_template.format(algorithm_name=problem_init.ALGORITHM_NAME,func_template=problem_init.func_template,docs = problem_init.docs,description=description,code=problem_init.class_code)
    else:
        if hasattr(problem_init,'FUNC_NAME'):
            GGA_prompts = problem_init.dec_template.format(algorithm_name=problem_init.ALGORITHM_NAME,func_template=problem_init.func_template,description=description,func_name = problem_init.FUNC_NAME,code=problem_init.class_code)
        else:
            GGA_prompts = problem_init.dec_template.format(algorithm_name=problem_init.ALGORITHM_NAME,func_template=problem_init.func_template,description=description,code=problem_init.class_code)

    messages  = [{"role": "user", "content": GGA_prompts}]
    for i in range(1):
        GGA = llm_code._chat_completion_api(messages = messages,temperature = 1)
        messages.append({"role": GGA[0].message.role, "content": GGA[0].message.content})
        try:
            set_code_to_file(GGA[0].message.content)
            file_name=problem_init.file_path.split('.')[0]
            init_eval = importlib.import_module(f'problems.{problem}.{file_name}')
            # 强制重新加载以确保使用最新代码
            importlib.reload(init_eval)
            print(GGA[0].message.content,end='\n'*6)

            check_err=problem_init.check_err(init_eval)
           
            not_find = False
            break
        except Exception as e:
           # Catch all types of exceptions
           # Print the exception type and value
            tb = traceback.format_exc()
            exception_info = f"Exception type: {type(e).__name__}" \
                            +"\n"+f"Exception value: {e}" \
                            +"Traceback (most recent call last):"+tb
            messages.append({"role": "user", "content": f"When I tried to call the algorithm from the outside according to the given template, an exception occurred:\n{exception_info}\n\nPlease modify your own code to fit my calling code."})
            print(exception_info,end='\n'*6)
    print('-'*160)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------

init_eval = importlib.import_module(f'problems.{problem}.{file_name}')
importlib.reload(init_eval)
def extract_functions_from_file(file_path):
    """
    Extract the names of all function definitions and their source code from the given Python file.
    
    :param file_path: The path to the Python file
    :return: A dictionary containing all function names and their source code
    """
    
    with open(file_path, "r", encoding="utf-8") as file:
        # Read the contents of the file
        source_code = file.read()
        # Parse into an abstract syntax tree
        tree = ast.parse(source_code, filename=file_path)

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_handl = getattr(init_eval,node.name)
            function_source = inspect.getsource(function_handl)

            if 'objective_function' in node.name:
                continue
            functions.append({'func_name':node.name,'func_source':'```python\n' + function_source + '\n```'})
    
    return functions

# Use this function to get all the functions and their source code in the specified file
file_path = f'problems/{problem}/{problem_init.file_path}'  # Replace this with your Python file path
functions = extract_functions_from_file(file_path)
doc = (messages[1]["content"] + messages[-1]["content"] if len(messages)> 1 else messages[1]["content"])
for i in range(len(functions)):
    item = functions[i]
    func_name, func_source = item['func_name'],item['func_source']
    code_description_prompts = FUNC_DESC_PROMPT_EN.format(func_name = func_name,func_source=func_source,doc=doc)
    code_description = llm._chat_completion_api(messages  = [{"role": "user", "content": code_description_prompts}],temperature=1)[0].message.content
    functions[i]['func_description'] = code_description
    functions[i]['doc'] = doc
    print(f"func_name : \t{func_name}\n")
    print(f"func_source :\n{func_source}\n")
    print(f"func_description: \n{code_description}")
    print("-"*50)
    sys.stdout.flush()

np.save(f'problems/{problem}/init_generated_funcs.npy',np.array(functions))