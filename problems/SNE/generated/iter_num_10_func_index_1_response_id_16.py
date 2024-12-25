from quadratic_function import QuadraticFunction
import random
import numpy as np
def backtracking_line_search(quadratic_func: QuadraticFunction, x: np.ndarray, direction: np.ndarray, alpha: float = 0.4, beta: float = 0.7) -> float:
    """
    Modified backtracking line search ensuring Armijo condition for decreased objective function.
    
    Parameters:
    - quadratic_func: `QuadraticFunction` instance representing the objective.
    - x: Current point in the optimization space.
    - direction: Search direction for optimization.
    - alpha: Armijo condition parameter (default 0.4).
    - beta: Reduction factor for step size (default 0.7).
    
    Returns:
    - Optimal step size based on the Armijo condition for sufficiently decreased objective function.
    """
    step_size = 1.0  # Initial step size
    current_obj_value = quadratic_func.objective_function(x)
    gradient_x = quadratic_func.gradient(x)

    # Reduce step size until Armijo condition is met
    while quadratic_func.objective_function(x + step_size * direction) > current_obj_value + alpha * step_size * gradient_x.T @ direction:
        step_size *= beta

    return step_size
def search_root(quadratic_fnc: QuadraticFunction, x0: np.ndarray, num_samples: int = 10, max_iter: int = 1000) -> np.ndarray:
    """
    Implements a new optimization algorithm to find a minimum using stochastic sampling.

    Parameters:
    - quadratic_fnc: An instance of QuadraticFunction.
    - x0: The initial point.
    - num_samples: Number of samples to consider at each iteration.
    - max_iter: Maximum number of iterations.

    Returns:
    - The point that minimizes the target function.
    """

    d = quadratic_fnc.d
    n = quadratic_fnc.n
    
    # Choose random initial solutions
    x = np.tile(x0, (num_samples, 1))

    for t in range(max_iter):
        samples = np.random.uniform(0, 1, size=(num_samples, d))

        # Evaluate the objective function at current set of points
        obj_vals = np.array([quadratic_fnc.objective_function(pt) for pt in x])

        # Sort the points based on their objective function values
        sorted_indices = np.argsort(obj_vals)
        
        # Choose the best and worst performing points for updating
        best_pt = x[sorted_indices[0]]
        worst_pt = x[sorted_indices[-1]]

        # Generate a new point by averaging the best performing and a random point
        alpha = np.random.rand()
        new_pt = alpha * best_pt + (1 - alpha) * samples[np.random.randint(num_samples)]

        # Replace the worst performing point with the new point
        x[np.argmax(obj_vals)] = new_pt

    return x[np.argmin(obj_vals)]
def sherman_morrison_update(B_inv: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Apply the Sherman-Morrison formula to update the inverse of a matrix B.
    
    Parameters:
    - B_inv: Current inverse of the sum of Hessians.
    - u: The vector used for the update.
    - v: The vector used for the update.
    
    Returns:
    - Updated inverse of B.
    """
    Bu = B_inv @ u
    uTBu = u.T @ Bu

    if uTBu < 1e-12:  # Handle near-zero denominator cases carefully
        return B_inv

    B_inv_updated = B_inv + np.outer(Bu, Bu) / (v.T @ u - uTBu)
    return B_inv_updated
def sr1_update(B: np.ndarray, grad_diff: np.ndarray, s: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    """
    Perform an updated version of Symmetric Rank 1 (SR1) update considering the given threshold.
    
    Parameters:
    - B: Current Hessian approximation.
    - grad_diff: The difference in gradients.
    - s: The step vector used to compute the update.
    - threshold: Threshold to check the smallness of denominator (default 1e-8).
    
    Returns:
    - Updated Hessian approximation.
    """
    Bs = B @ s
    diff = grad_diff - Bs
    denom = s.T @ diff

    if np.abs(denom) > threshold:  # To avoid division by zero or near-zero values
        B += np.outer(diff, diff) / denom
    
    return B