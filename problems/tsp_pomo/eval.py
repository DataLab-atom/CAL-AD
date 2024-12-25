##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = True
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
import torch

from problems.tsp_pomo.TSPEnv import TSPEnv as Env
from problems.tsp_pomo.TSPModel import TSPModel as Model

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, "..")  # for problem_def
# sys.path.insert(0, "../..")  # for utils

sys.path.insert(0, "../../../")
file_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的目录
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

# generated 目录的完整路径
generated_dir = os.path.join(file_dir, 'generated')

if generated_dir not in sys.path:
    sys.path.insert(0, generated_dir)

# 添加 set 类型到 PyTorch 的安全列表
torch.serialization.add_safe_globals([set])

##########################################################################################
# import

import logging
from problems.tsp_pomo.utils import create_logger, copy_all_src

from problems.tsp_pomo.TSPTester import TSPTester as Tester

from problems.tsp_pomo.gen_inst import dataset_conf

##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 1,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './checkpoints',  # directory path of pre-trained model and log files saved.
        'epoch': 3100,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 10,
    'test_batch_size': 10,
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 100,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp100_longTrain',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    avg_aug_obj = tester.run()
    return avg_aug_obj


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100

def get_testpair():
    problem_size = 200#int(sys.argv[1])
    mood = 'train'#sys.argv[3]
    basepath = os.path.dirname(__file__)
    dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.pt")
    env_params['test_file_path'] = dataset_path
    env_params['problem_size'] = problem_size
    tester_params['test_episodes'] = 10
    tester_params['test_batch_size'] = 10
    # cuda
    USE_CUDA = False #tester_params['use_cuda']
    if USE_CUDA:
        cuda_device_num = tester_params['cuda_device_num']
        torch.cuda.set_device(cuda_device_num)
        device = torch.device('cuda', cuda_device_num)
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)  # 或者你使用的其他数据类型
    else:
        device = torch.device('cpu')
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)  # 或者你使用的其他数据类型
    

    # ENV and MODEL
    env = Env(**env_params)
    model = Model(**model_params)
    model.device = device
    # Restore
    model_load = tester_params['model_load']
    checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
    checkpoint = torch.load(checkpoint_fullname, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return env,model,tester_params['test_episodes']

##########################################################################################

if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    mood = sys.argv[3]
    outfile_path = sys.argv[4]
    assert mood in ['train', 'val', "test"]
    
    basepath = os.path.dirname(__file__)
    # automacially generate dataset if nonexists
    if not os.path.isfile(os.path.join(basepath, f"dataset/train{dataset_conf['train'][0]}_dataset.pt")):
        from gen_inst import generate_datasets, dataset_conf
        generate_datasets()

    module = __import__(f'generated.{outfile_path}'.replace('.py',''), fromlist=['search_routine'])
    search_routine = getattr(module, 'search_routine')

    if not os.path.isfile(os.path.join(basepath, "checkpoints/checkpoint-3100.pt")):
        raise FileNotFoundError("No checkpoints found. Please see the readme.md and download the checkpoints.")

    if mood == 'train':
        dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.pt")
        env_params['test_file_path'] = dataset_path
        env_params['problem_size'] = problem_size
        tester_params['test_episodes'] = 10
        tester_params['test_batch_size'] = 10
         # cuda
        USE_CUDA = tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_device(device)
            torch.set_default_dtype(torch.float32)  # 或者你使用的其他数据类型
        else:
            device = torch.device('cpu')
            torch.set_default_device(device)
            torch.set_default_dtype(torch.float32)  # 或者你使用的其他数据类型
        device = device

        # ENV and MODEL
        env = Env(**env_params)
        model = Model(**model_params)
        model.device = device
        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        obj = search_routine(env,model,tester_params['test_episodes'],tester_params['test_batch_size'],tester_params['aug_factor'])
        print("[*] Average:")
        print(obj)

        
