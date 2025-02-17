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
    # Apply adaptive thresholding based on the magnitude of the elements
    adaptive_threshold = threshold * (1 + np.abs(x) / np.max(np.abs(x) + 1e-10))
    return np.sign(x) * np.maximum(np.abs(x) - adaptive_threshold, 0)
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
    Run the enhanced version of the Proximal Gradient with Random Reshuffling (PG-RR) algorithm.

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
    
    # Adaptive learning rate decay
    gamma_decay = gamma / (1 + 0.01 * num_epochs)
    
    for epoch in range(num_epochs):
        # Shuffle indices for random reshuffling
        indices = np.random.permutation(n)
        
        for i in indices:
            # Compute gradient for the current sample
            gradient = 2 * A[i].T @ (A[i] @ x - y[i])
            
            # Update solution with adaptive learning rate
            x = soft_thresholding(x - gamma_decay * gradient, gamma_decay * lambda_)
        
        # Update learning rate decay
        gamma_decay = gamma / (1 + 0.01 * (epoch + 1))
    
    return x