from dataclasses import dataclass
from typing import List
import numpy as np
def srk(G: np.ndarray, A: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Compute the symmetric rank-k update."""
    if np.allclose(G @ U, A @ U):
        return G
    temp = U.T @ (G - A) @ U
    if np.linalg.matrix_rank(temp) < U.shape[1]:  # Handle singularity
        return G  # or implement a robust pseudo-inverse
    return G - (G - A) @ U @ np.linalg.inv(temp) @ U.T @ (G - A)
def greedy_matrix(G: np.ndarray, A: np.ndarray, k: int) -> np.ndarray:
    """Select the greedy matrix U using a more sophisticated approach."""
    # Compute the difference matrix
    diff = G - A
    
    # Compute the Frobenius norm of each row in the difference matrix
    row_norms = np.linalg.norm(diff, axis=1)
    
    # Compute the absolute diagonal differences
    diag_diff = np.abs(np.diag(diff))
    
    # Combine row norms and diagonal differences with weighted scores
    # Adjust weights dynamically based on the relative magnitudes of row_norms and diag_diff
    max_row_norm = np.max(row_norms)
    max_diag_diff = np.max(diag_diff)
    
    if max_row_norm > 0 and max_diag_diff > 0:
        weight_row_norm = max_diag_diff / (max_row_norm + max_diag_diff)
        weight_diag_diff = max_row_norm / (max_row_norm + max_diag_diff)
    else:
        weight_row_norm = 0.5
        weight_diag_diff = 0.5
    
    weights = weight_row_norm * row_norms + weight_diag_diff * diag_diff
    
    # Select the indices of the top k rows with the largest combined scores
    indices = np.argpartition(weights, -k)[-k:]
    
    # Construct the selection matrix U
    U = np.zeros((G.shape[0], k))
    U[indices, np.arange(k)] = 1
    
    return U
def sherman_morrison(A_inv: np.ndarray, U: np.ndarray, V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute the Sherman-Morrison update."""
    temp = W - U.T @ A_inv @ V
    if np.linalg.matrix_rank(temp) < U.shape[1]:  # Handle singularity
        return A_inv # or implement a robust pseudo-inverse
    return A_inv + A_inv @ U @ np.linalg.inv(temp) @ V.T @ A_inv
def compute_omega(t: int, n: int, r0: float, rho: float, M: float, L: float) -> float:
    """Compute the scaling parameter omega with exponential decay."""
    if t % n == 0:
        decay_factor = np.exp(-rho * (t // n))
        return (1 + M * np.sqrt(L) * r0 * decay_factor)**2
def search_root(objective_function: callable, x0: np.ndarray, A_list: List[np.ndarray], b_list: List[np.ndarray],
                   tol: float = 1e-6, max_iter: int = 1000, k: int = 5, rho: float = 0.5, M: float = 1.0) -> np.ndarray:
    """Implements an enhanced LISR-k optimization algorithm with adaptive scaling and gradient descent steps."""

    n = len(A_list)
    d = x0.shape[0]
    z_list = [x0.copy() for _ in range(n)]
    B_list = [np.eye(d) for _ in range(n)]  # Initialize B_i^0
    B_bar = np.sum(B_list, axis=0)
    B_bar_inv = np.linalg.inv(B_bar)
    
    L = np.max(np.linalg.eigvals(A_list[0])) # Example, assuming all A_i have similar L
    mu = np.min(np.linalg.eigvals(A_list[0])) # Example, assuming all A_i have similar mu
    if M is None: # Default to this if M is not provided
        M = (L/mu)**(3/2)/mu # Third derivative upper bound, example using L and mu

    r0 = np.linalg.norm(x0)  # Initialize r0

    x = x0.copy()
    for t in range(max_iter):
        i_t = t % n
        omega = compute_omega(t, n, r0, rho, M, L)

        # Adaptive scaling based on the gradient norm
        grad_sum = np.sum([np.dot(A_i, z_i) + b_i for A_i, z_i, b_i in zip(A_list, z_list, b_list)], axis=0)
        grad_norm = np.linalg.norm(grad_sum)
        adaptive_scale = min(1.0, 1.0 / grad_norm) if grad_norm > 0 else 1.0

        U = greedy_matrix(omega * B_list[i_t], A_list[i_t], k)
        B_new = srk(omega * B_list[i_t], A_list[i_t], U)
        
        V = (omega * B_list[i_t] - A_list[i_t]) @ U
        B_bar_inv = sherman_morrison(B_bar_inv, V, V, U.T @ V) / omega  # Update B_bar_inv

        # Gradient descent step with adaptive scaling
        x_new = x - adaptive_scale * B_bar_inv @ grad_sum

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new.copy()
        z_list[i_t] = x.copy()
        B_list[i_t] = B_new.copy()

    return x