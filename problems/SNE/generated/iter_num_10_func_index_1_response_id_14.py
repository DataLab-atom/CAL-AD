from quadratic_function import QuadraticFunction
import numpy as np
def backtracking_line_search(quadratic_func: QuadraticFunction, x: np.ndarray, direction: np.ndarray, alpha: float = 0.4, beta: float = 0.7, max_iter: int = 100) -> float:
    """
    An advanced version of backtracking line search with a maximum iteration limit as a safety measure.

    Parameters:
    - quadratic_func: Instance of the QuadraticFunction.
    - x: Current point.
    - direction: Search direction (optimization step).
    - alpha: Armijo condition parameter (default 0.4).
    - beta: Step size reduction factor (default 0.7).
    - max_iter: Maximum number of iterations for line search (default 100).

    Returns:
    - Optimal step size satisfying Armijo condition.
    """
    t = 1.0  # Initial step size
    f_x = quadratic_func.objective_function(x)
    grad_x = quadratic_func.gradient(x)

    for _ in range(max_iter):
        if quadratic_func.objective_function(x + t * direction) <= f_x + alpha * t * grad_x.T @ direction:
            break
        t *= beta

    return t
def search_root(quadratic_func: QuadraticFunction, x0: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """
    Implements an improved version of the LISR-1 optimization algorithm for quadratic function minimization.

    Parameters:
    - quadratic_func: An instance of QuadraticFunction.
    - x0: The initial point.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.

    Returns:
    - The point that minimizes the target function based on stochastic solution sampling.
    """

    d = quadratic_func.d
    n = quadratic_func.n
    
    A_avg_inv = np.linalg.inv(quadratic_func.A_avg)
    B_inv = A_avg_inv.copy()

    x = x0.copy()
    z = [x0.copy() for _ in range(n)]
    B = [A_avg_inv.copy() for _ in range(n)]

    phi = np.sum([B[i] @ z[i] for i in range(n)], axis=0) / n - quadratic_func.gradient(x)
    g = np.sum([quadratic_func.A[i] @ z[i] + quadratic_func.b[i] for i in range(n)], axis=0) / n
    
    prev_obj_val = quadratic_func.objective_function(x0)

    for t in range(max_iter):
        i_t = t % n
        
        direction = B_inv @ (phi - g)

        step_size = backtracking_line_search(quadratic_func, x, direction, alpha=0.2, beta=0.5)
        x_new = x + step_size * direction
        
        obj_val = quadratic_func.objective_function(x_new)
        
        if obj_val > prev_obj_val - tol:
            print(f"Warning: Minimal improvement at iteration {t}. Current Obj Val: {obj_val}, Previous: {prev_obj_val}.")
        
        prev_obj_val = obj_val

        if np.linalg.norm(quadratic_func.gradient(x_new)) < tol:
            print(f"Converged at iteration {t} with tolerance {tol}")
            break
        
        grad_new = quadratic_func.A[i_t] @ x_new + quadratic_func.b[i_t]
        grad_old = quadratic_func.A[i_t] @ z[i_t] + quadratic_func.b[i_t]
        grad_diff = grad_new - grad_old
        
        s = x_new - z[i_t]
        B[i_t] = sr1_update(B[i_t], grad_diff, s)

        u = grad_diff
        v = B[i_t] @ s
        B_inv = sherman_morrison_update(B_inv, u, v)

        z[i_t] = x_new.copy()
        phi += B[i_t] @ z[i_t] - B[i_t] @ s
        g = np.sum([quadratic_func.A[i] @ z[i] + quadratic_func.b[i] for i in range(n)], axis=0) / n

        x = x_new

    return x
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
    Perform the modified SR1 update on the Hessian approximation.
    
    Parameters:
    - B: Current Hessian approximation.
    - grad_diff: The difference in gradients.
    - s: The step vector used to compute the update.
    - threshold: Small threshold value to avoid division by almost zero.
    
    Returns:
    - Updated Hessian approximation.
    """
    Bs = B @ s
    diff = grad_diff - Bs
    denom = s.T @ diff

    if np.abs(denom) > threshold:
        B += np.outer(diff, diff) / denom
    
    return B