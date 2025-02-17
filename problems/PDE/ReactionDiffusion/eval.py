from os import path
import  os
import sys
# 设置NumPy使用的CPU核心数为1
#os.environ['OMP_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
import numpy as np
import logging
import sys
from PIL import Image
import torch
from importlib import import_module

current_path = os.path.abspath(__file__)
current_path='/'.join(current_path.split('\\')[:-1])

sys.path.insert(0, "../../../")
sys.path.insert(0, "../../../")
file_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的目录
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

#generated 目录的完整路径
generated_dir = os.path.join(file_dir, 'generated')

if generated_dir not in sys.path:
     sys.path.insert(0, generated_dir)

if __name__ == "__main__":
    print("[*] Running ...")
    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    outfile_path = sys.argv[4]
    assert mood in ['train', 'val']
    module = __import__(f'{outfile_path}'.replace('.py',''), fromlist=['static_loss', 'static_forward', 'static_init'])
    run_loss = getattr(module, 'static_loss')
    run_forward = getattr(module, 'static_forward')
    run_init = getattr(module, 'static_init')
    get_run = __import__(f'run', fromlist=['static_main'])
    static_run = getattr(get_run, 'static_main')

    PODMode = 10
    num_bc = 68 #204
    dim_br1 = [PODMode*2, 100, 100, 100]
    dim_br2 = [num_bc, 150, 150, 150, 100] #150
    dim_tr = [2, 100, 100, 100]
   
    # 网络结构
    class opnn(torch.nn.Module):
        def __init__(self, branch1_dim, branch2_dim, trunk_dim):
            super(opnn, self).__init__()
            run_init(self,branch1_dim, branch2_dim, trunk_dim)

        def forward(self, f, f_bc, x):
            return run_forward(self,f, f_bc, x)
        
        def loss(self, f, f_bc, x, y):
            return run_loss(self, f, f_bc, x, y)
        
    torch.cuda.empty_cache()
    if mood == 'train':
        mean_abs_err,rel_l2_err=static_run(opnn(dim_br1, dim_br2, dim_tr), 10000, 'cuda')
        mean_abs_err=mean_abs_err.mean()
        rel_l2_err=rel_l2_err.mean()
        print("[*] Average:")
        print((mean_abs_err+rel_l2_err)/2)
    else:
        mean_abs_err,rel_l2_err=static_run(opnn(dim_br1, dim_br2, dim_tr), 10000, 'cuda')
        mean_abs_err=mean_abs_err.mean()
        rel_l2_err=rel_l2_err.mean()
        print("[*] Average:")
        print((mean_abs_err+rel_l2_err)/2)