from dataclasses import dataclass
import random
from typing import List
from typing import Tuple
import numpy as np
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
    Compute the gradient of the smooth part of the objective function using vectorized operations.

    Parameters:
        x (np.ndarray): The solution vector.
        A (List[np.ndarray]): A list of linear transformation matrices.
        y (List[np.ndarray]): A list of observation vectors.

    Returns:
        np.ndarray: The gradient vector.
    """
    n = len(y)
    # Stack A and y into a single 3D array and 2D array respectively for vectorized computation
    A_stacked = np.stack(A)
    y_stacked = np.stack(y)
    
    # Compute the residuals in a vectorized manner
    residuals = A_stacked @ x - y_stacked
    
    # Compute the gradient using vectorized operations
    gradient = 2 * np.einsum('ijk,ik->j', A_stacked, residuals) / n
    
    return gradient
def PG_RR(A: List[np.ndarray], y: List[np.ndarray], lambda_: float, gamma: float, num_epochs: int, initial_x: np.ndarray) -> Tuple[np.ndarray]:
    """
    Run the entry function of the (PG-RR) algorithm with adaptive learning rate and momentum.

    Parameters:
        A (List[np.ndarray]): A list of linear transformation matrices.
        y (List[np.ndarray]): A list of observation vectors.
        lambda_ (float): L1 regularization intensity.
        gamma (float): Initial learning rate (step size).
        num_epochs (int): Number of training cycles.
        initial_x (np.ndarray): Initial solution vector.

    Returns:
        Tuple[np.ndarray]: The last output containing the optimal solution vector.
    """
    x = initial_x.copy()
    n = len(y)
    momentum = np.zeros_like(x)
    beta = 0.9  # Momentum coefficient
    
    for epoch in range(num_epochs):
        # Adaptive learning rate decay
        current_gamma = gamma / (1 + 0.01 * epoch)
        
        for i in np.random.permutation(n):
            gradient = 2 * A[i].T @ (A[i] @ x - y[i])
            momentum = beta * momentum + (1 - beta) * gradient
            x = soft_thresholding(x - current_gamma * momentum, current_gamma * lambda_)
    
    return x