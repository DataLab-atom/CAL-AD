import numpy as np
from typing import List, Tuple

def objective_function(x: np.ndarray, A: List[np.ndarray], y: List[np.ndarray], lambda_: float) -> float:
    """
    Compute the combined objective function consisting of a smooth term and a non-smooth term.

    Parameters:
        x (np.ndarray): The solution vector.
        A (List[np.ndarray]): A list of linear transformation matrices.
        y (List[np.ndarray]): A list of observation vectors.
        lambda_ (float): L1 regularization intensity.

    Returns:
        float: The value of the objective function.
    """
    smooth_part = sum(np.linalg.norm(A[i] @ x - y[i]) ** 2 for i in range(len(y))) / len(y)
    nonsmooth_part = lambda_ * np.linalg.norm(x, ord=1)
    return smooth_part + nonsmooth_part

def soft_thresholding(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft thresholding to the input vector.

    Parameters:
        x (np.ndarray): The input vector.
        threshold (float): The threshold value.

    Returns:
        np.ndarray: The thresholded vector.
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def compute_gradient(x: np.ndarray, A: List[np.ndarray], y: List[np.ndarray]) -> np.ndarray:
    """
    Compute the gradient of the smooth part of the objective function.

    Parameters:
        x (np.ndarray): The solution vector.
        A (List[np.ndarray]): A list of linear transformation matrices.
        y (List[np.ndarray]): A list of observation vectors.

    Returns:
        np.ndarray: The gradient vector.
    """
    n = len(y)
    gradient = np.zeros_like(x)
    for i in range(n):
        gradient += 2 * A[i].T @ (A[i] @ x - y[i])
    return gradient / n

def PG_RR(A: List[np.ndarray], y: List[np.ndarray], lambda_: float, gamma: float, num_epochs: int, initial_x: np.ndarray) -> Tuple[np.ndarray]:
    """
    Run the entry function of the (PG-RR) algorithm.

    Parameters:
        A (List[np.ndarray]): A list of linear transformation matrices.
        y (List[np.ndarray]): A list of observation vectors.
        lambda_ (float): L1 regularization intensity.
        gamma (float): Learning rate (step size).
        num_epochs (int): Number of training cycles.
        initial_x (np.ndarray): Initial solution vector.

    Returns:
        Tuple[np.ndarray]: The last output containing the optimal solution vector.
    """
    x = initial_x.copy()
    n = len(y)

    for epoch in range(num_epochs):
        for i in np.random.permutation(n):
            gradient = 2 * A[i].T @ (A[i] @ x - y[i])
            x = soft_thresholding(x - gamma * gradient, gamma * lambda_)

    
    return x

if __name__ == "__main__":
    # Test code
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 10
    n_features = 784
    A = [np.random.randn(100, n_features) for _ in range(n_samples)]
    y = [np.random.randn(100) for _ in range(n_samples)]
    lambda_ = 0.1
    gamma = 0.01
    num_epochs = 100
    initial_x = np.random.randn(n_features)
    
    # Run PG_RR algorithm
    optimal_x = PG_RR(A, y, lambda_, gamma, num_epochs, initial_x)
    
    # Compute objective function value
    obj_value = objective_function(optimal_x, A, y, lambda_)
    
    print(f"Optimal solution: {optimal_x}")
    print(f"Objective function value: {obj_value}")
