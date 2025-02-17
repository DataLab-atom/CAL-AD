from dataclasses import dataclass
import random
from typing import List
from typing import Tuple
import numpy as np
def soft_thresholding(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft thresholding to the input vector with an adaptive thresholding mechanism.

    Parameters:
        x (np.ndarray): The input vector.
        threshold (float): The threshold value.

    Returns:
        np.ndarray: The thresholded vector.
    """
    # Apply adaptive thresholding based on the magnitude of each element
    thresholded = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > threshold:
            thresholded[i] = x[i] - threshold
        elif x[i] < -threshold:
            thresholded[i] = x[i] + threshold
        else:
            thresholded[i] = 0
    return thresholded
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
    A_stacked = np.stack(A, axis=0)  # Shape: (n, m, d)
    y_stacked = np.stack(y, axis=0)  # Shape: (n, m)
    
    # Compute residuals: A_i @ x - y_i for all i
    residuals = np.einsum('nmd,d->nm', A_stacked, x) - y_stacked  # Shape: (n, m)
    
    # Compute gradient contributions: 2 * A_i.T @ (A_i @ x - y_i) for all i
    gradient_contributions = 2 * np.einsum('nmd,nm->d', A_stacked, residuals)  # Shape: (d,)
    
    # Average the contributions
    gradient = gradient_contributions / n
    return gradient
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