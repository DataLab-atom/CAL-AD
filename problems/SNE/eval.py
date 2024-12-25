from os import path
import  os
import sys
import numpy as np
import logging
import sys
from joblib import dump, load

sys.path.insert(0, "../../../")
sys.path.insert(0, "../../../")
file_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的目录
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

# generated 目录的完整路径
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
    module = __import__(f'{outfile_path}'.replace('.py',''), fromlist=['search_root'])
    search_root = getattr(module, 'search_root')
    basepath = path.join(path.dirname(__file__), "dataset")
    if not path.isfile(path.join(basepath, f"{mood}{problem_size}_dataset.joblib")):
        from gen_inst import generate_datasets
        generate_datasets()
    
    if mood == 'train':
        dataset_path = path.join(basepath, f"{mood}{problem_size}_dataset.joblib")
        quadratic_func,x0 = load(dataset_path)
        x_new =  search_root(quadratic_func,x0,max_iter=20)
        print("[*] Average:")
        print(quadratic_func.objective_function(x_new))
    
    else:
        for problem_size in [4, 8, 12]:
            dataset_path = path.join(basepath, f"{mood}{problem_size}_dataset.joblib")
            quadratic_func,x0 = load(dataset_path)
            x_new =  search_root(quadratic_func,x0,max_iter=20)
            print(f"[*] Average for {problem_size}: {quadratic_func.objective_function(x_new)}")