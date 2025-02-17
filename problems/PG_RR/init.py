from prompts.init_algorithm_code import GENERATE_ALG_PROMPT_WODOCS_EN
import numpy as np
import importlib
import numpy as np
from PIL import Image

import unicodedata
import os
import time
import sys
import win32com.client
import pythoncom
import html2text

pythoncom.CoInitialize()
word = win32com.client.Dispatch('Word.Application')
word.Visible=False
current_path = os.path.abspath(__file__)
print('\\'.join(current_path.split('\\')[:-1]))
file = word.Documents.Open(current_path+"\\pgrr.pdf")
file.SaveAs(current_path+"\\pgrr.html",FileFormat=8)
file.Close()
word.Quit()

html_content = open(current_path+'\\pgrr.html', 'r', encoding='gb2312', errors='replace')
text = html_content.read()
html_content.close()


converter = html2text.HTML2Text()
converter.body_width = 0  
converter.ignore_links = False  
converter.ignore_images = False 
converter.ignore_emphasis = False 
converter.ignore_tables = False  
converter.ignore_pre = False  
converter.ignore_strikethrough = False 
markdown_str = converter.handle(text)

with open(current_path+'pgrr.md', 'w', encoding='utf-8') as file:
    file.write(markdown_str)

func_template ="""
def PG_RR(A, y, lambda_, gamma, num_epochs, x):

'''
Run the entry function of the (PG-RR) algorithm.

Parameter:
        A (list of np.ndarray): A list of linear transformation matrices. A shape(10, 100, 784)
        y (list or np.ndarray): A vector list of observations. y shape(10, 784)
        lambda_ (float): L1 regularization intensity.
        gamma (float): learning rate (step size).
        num_epochs (int): Number of training cycles.
        initial_x (np.ndarray): Initial solution vector. initial_x shape(784)

Return value:
        tuple[-1]: The last output containing the list of np.ndarray of the optimal solution. shape is(784,)
'''

"""

GENERATE_PROBLEM_DESC_PROMPT_EN = """
You are an expert-level algorithm engineer; please describe the optimization objective in one paragraph.

Optimization Objective:
Find a point \( x^* \) that minimizes the objective function \( f(x) \). The objective function is defined as:

$ \\frac{{1}}{{n}} \\sum_{{i=1}}^{{n}} \\| y_i - A_i x \\|_2^2 + \\lambda \\| x \\|_1 $

where \( A_i \) are definited matrices and \( y_i \) are definited vector. The goal is to determine the optimal point \( x^* \) that achieves the minimum value of this function.

Code related to the optimization objective:
```
def objective_function(x, A, y, lambda_):
    smooth_part = sum(np.linalg.norm(A[i] @ x - y[i]) ** 2 for i in range(len(y))) / len(y)
    nonsmooth_part = lambda_ * np.linalg.norm(x, ord=1)
    return smooth_part + nonsmooth_part
```

Please directly return the description content you have written."""


description_prompts = GENERATE_PROBLEM_DESC_PROMPT_EN
dec_template=GENERATE_ALG_PROMPT_WODOCS_EN
FUNC_NAME= "PG_RR"
ALGORITHM_NAME="PGRR algorithm"
file_path='generated.py'
docs=''

class_code='''
def objective_function(x, A, y, lambda_):
    smooth_part = sum(np.linalg.norm(A[i] @ x - y[i]) ** 2 for i in range(len(y))) / len(y)
    nonsmooth_part = lambda_ * np.linalg.norm(x, ord=1)
    return smooth_part + nonsmooth_part
'''

with open('problems\PG_RR\pgrr.md', 'r', encoding='utf-8') as file:
    docs = file.read()

def check_err(init_eval):

    # Parameter settings
    np.random.seed(0)  # Fixed random seeds to ensure reproducible results

    m = 10  # The number of data points
    n = 784  # Number of features
    lambda_ = 1e-5  # Regularization parameters
    gamma = 6.5e-8   # step size
    num_epochs=int(0.2*1e5)

    true_x = np.array(Image.open('problems/PG_RR/4.jpg'))
    true_x = true_x.reshape(784)

    x=true_x+np.random.randn(784)*(2.4e-3)

    phi=np.random.randn(100,784)
    O = np.random.randn(n, 100, 784)
    A = [O[i] for i in range(m)]
    y = [A[i] @ true_x for i in range(m)]

    r=np.random.normal(loc=0, scale=1e-2, size=100)

    #y+=3*r
    
    # execute PG-RR
    x_new = init_eval.PG_RR(A, y, lambda_, gamma, num_epochs, x)

    # compare the error 
    assert (np.linalg.norm((phi@(x_new-true_x)))**2)/len(y)<=1e-2