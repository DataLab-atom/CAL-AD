import numpy as np

def logistic_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, reg_param: float) -> float:
    n = X.shape[0]
    z = X @ w
    log_loss = (1 / n) * np.sum(np.log(1 + np.exp(-y * z)))
    l2_reg = (reg_param / 2) * np.dot(w, w)
    loss = log_loss + l2_reg
    return loss

def logistic_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, reg_param: float) -> np.ndarray:
    n = X.shape[0]
    z = X @ w
    grad = (1 / n) * X.T @ (-y / (1 + np.exp(y * z))) + reg_param * w
    return grad