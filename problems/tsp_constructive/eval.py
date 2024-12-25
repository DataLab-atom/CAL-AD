from os import path
import  os
import sys
import numpy as np
import logging
import sys
sys.path.insert(0, "../../../")
file_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的目录
if file_dir not in sys.path:
    sys.path.append(file_dir)
from scipy import spatial

if __name__ == "__main__":
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    outfile_path = sys.argv[4]
    assert mood in ['train', 'val']
    module = __import__(f'generated.{outfile_path}'.replace('.py',''), fromlist=['search_routine'])
    search_routine = getattr(module, 'search_routine')
    basepath = path.join(path.dirname(__file__), "dataset")
    if not path.isfile(path.join(basepath, "train50_dataset.npy")):
        from gen_inst import generate_datasets
        generate_datasets()
    
    if mood == 'train':
        dataset_path = path.join(basepath, f"{mood}{problem_size}_dataset.npy")
        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]
        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
        
        objs = []
        for i, node_pos in enumerate(node_positions):
            distance_matrix = spatial.distance.cdist(node_pos, node_pos, metric='euclidean')
            def cal_total_distance(routine: np.ndarray,distance_matrix: np.ndarray) -> float:
                '''The objective function. input routine, return total distance.
                cal_total_distance(np.arange(num_points))
                '''
                expected = np.arange(len(routine))
                sorted_arr = np.sort(routine)
                if not np.array_equal(sorted_arr, expected):
                    raise "break tsp rule"
                next_points = np.roll(routine, -1)
                distances = distance_matrix[routine, next_points]
                return np.sum(distances)
            obj = search_routine(cal_total_distance,distance_matrix,0,20,100)
            obj = cal_total_distance(obj,distance_matrix)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj)
        
        print("[*] Average:")
        print(np.mean(objs))
    
    else:
        for problem_size in [20, 50, 100]:
            dataset_path = path.join(basepath, f"{mood}{problem_size}_dataset.npy")
            node_positions = np.load(dataset_path)
            logging.info(f"[*] Evaluating {dataset_path}")
            n_instances = node_positions.shape[0]
            objs = []
            for i, node_pos in enumerate(node_positions):
                distance_matrix = spatial.distance.cdist(node_pos, node_pos, metric='euclidean')
                def cal_total_distance(routine: np.ndarray,distance_matrix: np.ndarray) -> float:
                    '''The objective function. input routine, return total distance.
                    cal_total_distance(np.arange(num_points))
                    '''
                    expected = np.arange(len(routine))
                    sorted_arr = np.sort(routine)
                    if not np.array_equal(sorted_arr, expected):
                        raise "break tsp rule"
                    next_points = np.roll(routine, -1)
                    distances = distance_matrix[routine, next_points]
                    return np.sum(distances)
                obj = search_routine(cal_total_distance,distance_matrix,20,100)
                objs.append(cal_total_distance(obj,distance_matrix).item())
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")