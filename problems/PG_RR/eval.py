from os import path
import  os
import sys
# Set the number of CPU cores used by NumPy to 1
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
import numpy as np
import logging
import sys
from PIL import Image

sys.path.insert(0, "../../../")
sys.path.insert(0, "../../../")
file_dir = os.path.dirname(os.path.abspath(__file__))  
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
    module = __import__(f'{outfile_path}'.replace('.py',''), fromlist=['PG_RR'])
    PG_RR = getattr(module, 'PG_RR')

    '''
    load data
    '''
    np.random.seed(0) 
    m ,n= 10,784
    lambda_ = 1e-5  
    gamma = 6.5e-8   
    num_epochs=int(0.2*1e5)
    true_x = np.array(Image.open(path.join(path.dirname(__file__), "4.jpg")))
    true_x = true_x.reshape(784)
    x=true_x+np.random.randn(784)*(2.4e-3)
    phi=np.random.randn(100,784)
    O = np.random.randn(n, 100, 784)
    A = [O[i] for i in range(m)]
    y = [A[i] @ true_x for i in range(m)]
    r=np.random.normal(loc=0, scale=1e-2, size=100)
    #y+=3*r
    
    if mood == 'train':
        x_new = PG_RR(A, y, lambda_, gamma, num_epochs, x)
        print("[*] Average:")
        print((np.linalg.norm((phi@(x_new-true_x)))**2)/len(y))
    
    else:
        x_new = PG_RR(A, y, lambda_, gamma, num_epochs, x)
        print("[*] Average:")
        print((np.linalg.norm((phi@(x_new-true_x)))**2)/len(y))