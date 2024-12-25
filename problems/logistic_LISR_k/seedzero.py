import numpy as np
from typing import List, Callable

def logistic_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, reg_param: float) -> float:
    """Calculates the logistic loss."""
    n = X.shape[0]
    z = X @ w
    log_loss = (1 / n) * np.sum(np.log(1 + np.exp(-y * z)))
    l2_reg = (reg_param / 2) * np.dot(w, w)
    return log_loss + l2_reg

def logistic_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, reg_param: float) -> np.ndarray:
    """Calculates the gradient of the logistic loss."""
    n = X.shape[0]
    z = X @ w
    grad = -(1 / n) * X.T @ (y * (np.exp(-y * z) / (1 + np.exp(-y * z)))) + reg_param * w
    return grad

def hessian_approx(X: np.ndarray, y: np.ndarray, w: np.ndarray, reg_param: float) -> np.ndarray:
    """Approximates the Hessian of the logistic loss."""
    n = X.shape[0]
    z = X @ w
    sigmoid = 1 / (1 + np.exp(-y * z))
    hessian = (1/n) * X.T @ np.diag(sigmoid * (1 - sigmoid)) @ X + reg_param * np.eye(X.shape[1])
    return hessian
    
def srk(G: np.ndarray, A: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Performs the Symmetric Rank-k update."""
    if np.allclose(G @ U, A @ U):
        return G
    try:
        temp = np.linalg.inv(U.T @ (G - A) @ U)  # Efficiently handle the inverse
        update = (G - A) @ U @ temp @ U.T @ (G - A)
        return G - update
    except np.linalg.LinAlgError:  # Handle potential singular matrices
        return G

def greedy_directions(G: np.ndarray, A: np.ndarray, k: int) -> np.ndarray:
    """Selects k greedy directions."""
    diff = np.diag(G - A)
    indices = np.argsort(diff)[::-1][:k]
    U = np.eye(G.shape[0])[:, indices]
    return U
    
def update_params(B_sum_inv: np.ndarray, B_i_old: np.ndarray, B_i_new: np.ndarray, z_i: np.ndarray, grad_diff: np.ndarray) -> np.ndarray:
    """Updates model parameters."""
    update = B_sum_inv @ (B_i_new @ z_i - B_i_old @ z_i - grad_diff)
    return update


def search_root(logistic_loss: callable, logistic_gradient: callable, X: np.ndarray, y: np.ndarray, reg_param: float = 1e-3,
                tol: float = 1e-6, max_iter: int = 1000, k: int = 5) -> np.ndarray:
    """Implements the LISR-k algorithm."""
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    z = np.tile(w, (n_samples, 1))
    B = np.tile(np.eye(n_features), (n_samples, 1, 1)) * reg_param # initialize with small positive value
    B_sum = np.sum(B, axis=0)
    B_sum_inv = np.linalg.inv(B_sum)

    for t in range(max_iter):
        i = t % n_samples
        w_new = w + update_params(B_sum_inv, B[i], B[i], z[i], logistic_gradient(X[i:i+1,:], y[i:i+1], z[i], reg_param))
        
        if np.linalg.norm(w_new - w) < tol:
            return w_new

        w = w_new
        z[i] = w

        A = hessian_approx(X[i:i+1,:], y[i:i+1], z[i], reg_param) # calculate hessian at z[i] which is w
        U = greedy_directions(B[i], A, k)
        B[i] = srk(B[i], A, U)

        B_sum = np.sum(B, axis=0)
        B_sum_inv = np.linalg.inv(B_sum)
    return w


if __name__ == "__main__":
    # Test code here
    np.random.seed(0)
    n_samples, n_features = 100, 5
    X = np.random.rand(n_samples, n_features)
    y = np.random.choice([-1, 1], n_samples)
    reg_param = 0.1
    
    w_opt = search_root(logistic_loss, logistic_gradient, X, y, reg_param, k=2)
    loss = logistic_loss(X,y,w_opt,reg_param)
    grad = logistic_gradient(X, y, w_opt, reg_param)

    print(f"Optimal weight vector: {w_opt}")
    print(f"Loss at optimal w: {loss}")
    print(f"Gradient norm at optimal w: {np.linalg.norm(grad)}")
