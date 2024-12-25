from quadratic_function import QuadraticFunction
import numpy as np
def backtracking_line_search(quadratic_func: QuadraticFunction, x: np.ndarray, direction: np.ndarray, alpha: float = 0.4, beta: float = 0.7, max_iter: int = 100) -> float:
    """
    Enhanced backtracking line search ensuring the Armijo condition for sufficient decrease.
    
    Parameters:
    - quadratic_func: Instance of the QuadraticFunction.
    - x: Current point.
    - direction: The search direction vector.
    - alpha: Parameter for the Armijo condition.
    - beta: Reduction factor for step size.
    - max_iter: Maximum number of iterations for line search.
    
    Returns:
    - Optimal step size that satisfies the Armijo condition.
    """
    t = 1.0  # Initial step size
    f_x = quadratic_func.objective_function(x)
    grad_x = quadratic_func.gradient(x)
    d = alpha * (grad_x.T @ direction)

    for _ in range(max_iter):
        if quadratic_func.objective_function(x + t * direction) <= f_x + t * d:
            break
        t *= beta

    return t
def search_root(quadratic_func, x0, tol=1e-6, max_iter=1000):
    d = quadratic_func.d
    n = quadratic_func.n
    A_inv_avg = np.linalg.inv(quadratic_func.A_avg)
    B_inv = A_inv_avg.copy()
    x = x0.copy()
    z = [x0.copy() for _ in range(n)]
    B = [A_inv_avg.copy() for _ in range(n)]
    prev_obj_val = quadratic_func.objective_function(x0)
    
    for t in range(max_iter):
        i_t = t % n
        
        grad_x = quadratic_func.gradient(x)
        grad_diff = quadratic_func.A[i_t] @ x + quadratic_func.b[i_t] - (quadratic_func.A[i_t] @ z[i_t] + quadratic_func.b[i_t])
        
        phi = np.sum([B[j] @ z[j] for j in range(n)], axis=0) / n - grad_x
        direction = B_inv @ phi
        step_size = backtracking_line_search(quadratic_func, x, direction)
        x_new = x + step_size * direction
        obj_val = quadratic_func.objective_function(x_new)
        
        if obj_val > prev_obj_val - tol:
            print(f"Warning: Minimal improvement at iteration {t}. Current Obj Val: {obj_val}, Previous: {prev_obj_val}.")
        prev_obj_val = obj_val
        
        if np.linalg.norm(grad_diff) < tol:
            print(f"Converged at iteration {t} with tolerance {tol}")
            break

        s = x_new - z[i_t]
        B[i_t] = sr1_update(B[i_t], grad_diff, s)
        
        u = grad_diff
        v = B[i_t] @ s
        B_inv = sherman_morrison_update(B_inv, u, v)

        z[i_t] = x_new
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
    Perform an enhanced version of the SR1 update on the Hessian approximation with a specified threshold.
    
    Parameters:
    - B: Current Hessian approximation.
    - grad_diff: The difference in gradients.
    - s: The step vector used to compute the update.
    - threshold: The threshold for the absolute value of the denominator to prevent numerical instabilities.
    
    Returns:
    - Updated Hessian approximation using a refined SR1 update.
    """
    Bs = B @ s
    diff = grad_diff - Bs
    denom = s.T @ diff

    if np.abs(denom) > threshold:  
        B += np.outer(diff, diff) / denom
    
    return B