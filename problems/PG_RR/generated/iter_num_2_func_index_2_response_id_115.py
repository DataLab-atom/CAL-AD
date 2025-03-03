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
    # Stack all A_i matrices and y_i vectors into a single 3D array and 2D array respectively
    A_stacked = np.stack(A)
    y_stacked = np.stack(y)
    
    # Compute the residuals (A_i @ x - y_i) for all i in a vectorized manner
    residuals = np.einsum('ijk,k->ij', A_stacked, x) - y_stacked
    
    # Compute the gradient contributions for all i in a vectorized manner
    gradient_contributions = np.einsum('ijk,ij->k', A_stacked, residuals)
    
    # Average the contributions and multiply by 2
    gradient = 2 * gradient_contributions / n
    
    return gradient
def PG_RR(A: List[np.ndarray], y: List[np.ndarray], lambda_: float, gamma: float, num_epochs: int, initial_x: np.ndarray, momentum: float = 0.9) -> Tuple[np.ndarray]:
    """
    Run the entry function of the (PG-RR) algorithm with momentum.

    Parameters:
        A (List[np.ndarray]): A list of linear transformation matrices.
        y (List[np.ndarray]): A list of observation vectors.
        lambda_ (float): L1 regularization intensity.
        gamma (float): Learning rate (step size).
        num_epochs (int): Number of training cycles.
        initial_x (np.ndarray): Initial solution vector.
        momentum (float): Momentum factor for accelerated convergence.

    Returns:
        Tuple[np.ndarray]: The last output containing the optimal solution vector.
    """
    x = initial_x.copy()
    n = len(y)
    velocity = np.zeros_like(x)
    
    for epoch in range(num_epochs):
        for i in np.random.permutation(n):
            gradient = 2 * A[i].T @ (A[i] @ x - y[i])
            velocity = momentum * velocity - gamma * gradient
            x = soft_thresholding(x + velocity, gamma * lambda_)
    
    return x