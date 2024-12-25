from os import path
import  os
import sys
import numpy as np
import logging
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
from joblib import dump, load
# 数据集及其正则化参数
datasets = {
    'a9a': 1e-3,
    'w8a': 1e-4,
    'ijcnn1': 1e-4,
    'mushrooms': 1e-3,
    'phishing': 1e-4,
    'svmguide3': 1e-3,
    'german.numer': 1e-3,
    'splice': 1e-4,
    'covtype': 1e-3
}

def logistic_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, reg_param: float) -> float:
    """
    Calculate the logistic regression loss function with L2 regularization.
    
    Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Label vector of shape (n_samples,), where labels are -1 or 1.
        w (np.ndarray): Weight vector of the model of shape (n_features,).
        reg_param (float): Regularization parameter λ, used to control model complexity and prevent overfitting.
        
    Returns:
        float: The value of the loss function for the current weights w.
    
    Description:
        This function computes the average log loss plus the L2 regularization term for a given feature matrix X and label vector y.
        The log loss measures the difference between the model's predictions and the actual labels, while the L2 regularization term helps prevent overfitting.
        
        Formula:
        $$
        f(w) = \frac{1}{n} \sum_{i=1}^{n} \log(1 + \exp(-y_i x_i^T w)) + \frac{\lambda}{2} \|w\|^2
        $$
        Where:
        - $ n $ is the number of samples.
        - $ x_i $ is the feature vector of the i-th data point.
        - $ y_i $ is the label of the i-th data point.
        - $ w $ is the weight vector of the model.
        - $ \lambda $ is the regularization parameter.
        - $ \|\cdot\| $ denotes the Euclidean norm.
    """
    # Get the number of samples
    n = X.shape[0]
    
    # Compute the linear combination z = X @ w
    z = X @ w
    
    # Compute the log loss part: sum(log(1 + exp(-y * z))) / n
    log_loss = (1 / n) * np.sum(np.log(1 + np.exp(-y * z)))
    
    # Compute the L2 regularization term: reg_param/2 * ||w||^2
    l2_reg = (reg_param / 2) * np.dot(w, w)
    
    # Total loss is the sum of the log loss and the L2 regularization term
    loss = log_loss + l2_reg
    
    return loss

def logistic_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, reg_param: float) -> np.ndarray:
    """
    Calculate the gradient of the logistic regression loss function with L2 regularization.
    
    Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Label vector of shape (n_samples,), where labels are -1 or 1.
        w (np.ndarray): Weight vector of the model of shape (n_features,).
        reg_param (float): Regularization parameter λ, used to control model complexity and prevent overfitting.
        
    Returns:
        np.ndarray: The gradient of the loss function with respect to the weights w.
    """
    n = X.shape[0]
    z = X @ w
    grad = (1 / n) * X.T @ (-y / (1 + np.exp(y * z))) + reg_param * w
    return grad

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
        X, y = load_svmlight_file('E:/all_works/iclr2025/AEL-P-SNE(1)/AEL-P-SNE/problems/logistic_LISR_k/a9a')  # 根据需要选择数据集
        X = X.toarray()  # 如果数据是稀疏格式，则转换为稠密数组
                    
        # 将标签从{0, 1}转换为{-1, 1}
        y = 2 * y - 1
                    
        # 数据预处理
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        w_new =  search_root(logistic_loss,logistic_gradient,X,y,datasets['a9a'],max_iter=100,k=5)
        print("[*] Average:")
        print(logistic_loss(X,y, w_new,datasets['a9a']))