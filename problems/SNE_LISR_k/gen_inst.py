import os
import numpy as np
from quadratic_function import generate_A_b
from joblib import dump, load


def generate_datasets():
    basepath = os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(basepath, exist_ok=True)
    np.random.seed(1234)
    d = 20
    n = 100

    for xi,kappa in zip([8],[3.12e4]):
        test_dataset = [generate_A_b(xi,d, n) for _ in range(50)]
        x0 = [np.random.rand(d) for _ in range(50)]
        dump((test_dataset,x0),os.path.join(basepath, f'train{xi}_dataset.joblib'))
        
    d = 50
    n = 1000
    for xi,kappa in zip([4,8,12],[3.03e2,3.12e4,3.12e6]):
        test_dataset = generate_A_b(xi,d, n)
        x0 = np.random.rand(d)
        dump((test_dataset,x0),os.path.join(basepath, f'val{xi}_dataset.joblib'))
        
    for xi,kappa in zip([4,8,12],[3.03e2,3.12e4,3.12e6]):
        test_dataset = generate_A_b(xi,d, n) 
        x0 = np.random.rand(d)
        dump((test_dataset,x0),os.path.join(basepath, f'test{xi}_dataset.joblib') )

if __name__ == "__main__":
    generate_datasets()