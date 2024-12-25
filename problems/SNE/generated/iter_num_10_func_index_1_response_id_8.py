from quadratic_function import QuadraticFunction
import numpy as np
def backtracking_line_search(quadratic_func: QuadraticFunction, x: np.ndarray, direction: np.ndarray, alpha: float = 0.3, beta: float = 0.6) -> float:
    """
    Improved backtracking line search for optimization with additional tuning parameters.

    Parameters:
    - quadratic_func: The QuadraticFunction instance representing the objective function.
    - x: The current point in the search space.
    - direction: The optimization direction (typically the step computed during optimization).
    - alpha: Parameter controlling the Armijo condition for sufficient decrease (default 0.3).
    - beta: The reduction factor for the step size in each iteration (default 0.6).

    Returns:
    - The optimal step size based on the specified parameters.
    """
    t = 1.0  # Initial step size
    f_x = quadratic_func.objective_function(x)
    grad_direction = alpha * (quadratic_func.gradient(x).T @ direction)

    while quadratic_func.objective_function(x + t * direction) > f_x + t * grad_direction:
        t *= beta
    
    return t
def search_root(quadratic_func, x0, tol=1e-6, max_iter=1000):
    d = quadratic_func.d
    n = quadratic_func.n
    A_inv = np.linalg.inv(quadratic_func.A_avg)
    B_inv = A_inv
    
    x = x0.copy()
    z = [x0.copy() for _ in range(n)]
    B = [A_inv.copy() for _ in range(n)]
    phi = np.mean([B[i] @ z[i] for i in range(n)], axis=0) - quadratic_func.gradient(x)
    g = np.mean([quadratic_func.A[i] @ z[i] + quadratic_func.b[i] for i in range(n)], axis=0)
    
    for t in range(max_iter):
        i_t = t % n
        
        x = B_inv @ (phi - g)
        
        if np.linalg.norm(quadratic_func.gradient(x)) < tol:
            break
        
        grad_current = quadratic_func.A[i_t] @ x + quadratic_func.b[i_t]
        grad_prev = quadratic_func.A[i_t] @ z[i_t] + quadratic_func.b[i_t]
        grad_diff = grad_current - grad_prev
        s = x - z[i_t]
        B[i_t] = sr1_update(B[i_t], grad_diff, s)
        
        u = grad_diff
        v = B[i_t] @ s
        B_inv = sherman_morrison_update(B_inv, u, v)

        z[i_t] = x.copy()
        phi += B[i_t] @ z[i_t] - B[i_t] @ z[i_t]
        g = np.mean([quadratic_func.A[i] @ z[i] + quadratic_func.b[i] for i in range(n)], axis=0)

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
    Perform an enhanced version of the SR1 update for Hessian approximation.
    This version allows customization of tolerance threshold.

    Parameters:
    - B: Current Hessian approximation.
    - grad_diff: The difference in gradients.
    - s: The step vector used to compute the update.
    - threshold: Tolerance threshold for the denominator (default: 1e-8).

    Returns:
    - Updated Hessian approximation.
    """
    Bs = B @ s
    diff = grad_diff - Bs
    denom = s.T @ diff

    if np.abs(denom) > threshold:
        B += np.outer(diff, diff) / denom
    
    return B