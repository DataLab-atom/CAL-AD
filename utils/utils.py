import logging
import re
import inspect
import hydra
from prompts.init_algorithm_code import API_KEY,BASE_URL

base_url = BASE_URL
api_key = API_KEY


def init_client(cfg):
    global client
    if cfg.get("model", None): # for compatibility
        model: str = cfg.get("model")
        temperature: float = cfg.get("temperature", 1.0)
        if model.startswith("gpt"):
            from utils.llm_client.openai import OpenAIClient
            client = OpenAIClient(model, temperature,base_url=base_url,api_key=api_key)
        elif cfg.model.startswith("GLM"):
            from utils.llm_client.zhipuai import ZhipuAIClient
            client = ZhipuAIClient(model, temperature,base_url=base_url,api_key=api_key)
        else: # fall back to Llama API
            from utils.llm_client.llama_api import LlamaAPIClient
            client = LlamaAPIClient(model, temperature,base_url=base_url,api_key=api_key)
    else:
        client = hydra.utils.instantiate(cfg.llm_client)
    return client
    

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if  len(log) > 0:
            if log_status and "Traceback" in log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            else:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successful!")
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r'<start>(.*?)```python', r'<start>(.*?)<end>']
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end+1])
    
    if code_string is None:
        return None
    # Add import statements if not present
    if "np" in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string:
        code_string = "import torch\n" + code_string
    return code_string

def extract_code_from_generator_onece(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end+1])
        
    if code_string is None:
        return ""
    return code_string

def extract_code_from_generators(contents):
    code_string = '\n'.join([extract_code_from_generator_onece(content) for content in contents])
    # Add import statements if not present
    if "np" in code_string:
        code_string = "import numpy as np\n" + code_string
    if "Tuple" in code_string:
        code_string = "from typing import Tuple\n" + code_string
    if "Callable" in code_string:
        code_string = "from typing import Callable\n" + code_string
    if "List" in code_string:
        code_string = "from typing import List\n" + code_string
        
    if "torch" in code_string:
        code_string = "import torch\n" + code_string
    if "random" in code_string:
        code_string = "import random\n" + code_string
    if "QuadraticFunction" in code_string:
        code_string = "from quadratic_function import QuadraticFunction\n" + code_string

    if "numpy.linalg" in code_string:
        code_string = "from numpy.linalg import inv, norm, pinv\n" + code_string
    
    if "TSPEnv" in code_string:
        code_string = "from TSPEnv import TSPEnv as Env\n" + code_string

    if "TSPModel" in code_string:
        code_string = "from TSPModel import TSPModel as Model\n" + code_string
    
    if "RewardModel" in code_string:
        code_string = "from reward_functions import RewardModel\n" + code_string
    
    code_string = "from dataclasses import dataclass\n" + code_string
    return code_string

def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name