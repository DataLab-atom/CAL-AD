from quadratic_function import QuadraticFunction
import numpy as np
def backtracking_line_search(quadratic_func: QuadraticFunction, x: np.ndarray, direction: np.ndarray, alpha: float = 0.3, beta: float = 0.8) -> float:
    """
    Perform backtracking line search v2 for sufficient decrease in the objective function.
    
    Parameters:
    - quadratic_func: Instance of the QuadraticFunction.
    - x: Current point in the search space.
    - direction: The search direction (typically the gradient or optimization direction).
    - alpha: Parameter for the Armijo condition (default 0.3).
    - beta: Reduction factor for the step size (default 0.8).
    
    Returns:
    - The optimal step size.
    """
    t = 1.0  # Initial step size
    obj_val = quadratic_func.objective_function(x)
    gradient_x = quadratic_func.gradient(x)

    # Backtrack by reducing step size until a sufficient decrease condition holds
    while quadratic_func.objective_function(x + t * direction) > obj_val + alpha * t * gradient_x.T @ direction:
        t *= beta

    return t
def search_root(quadratic_func: QuadraticFunction, x0: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
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
        step_size = backtracking_line_search(quadratic_func, x, direction)
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
def sr1_update(B: np.ndarray, grad_diff: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Perform an extended version of the SR1 update on the Hessian approximation by incorporating rank-2 updates as well.
    
    Parameters:
    - B: Current Hessian approximation.
    - grad_diff: The difference in gradients.
    - s: The step vector used to compute the update.
    
    Returns:
    - Updated Hessian approximation.
    """
    Bs = B @ s
    diff = grad_diff - Bs
    elem_denom = s.T @ diff
    sec_denom = diff.T @ diff

    if np.abs(elem_denom) > 1e-8 and sec_denom != 0:  # To prevent division by zero or near-zero cases and ensure no rank-1 update
        num_out = np.outer(diff, diff) / sec_denom
        denom_factor = (1 - elem_denom / sec_denom)
        B += num_out * denom_factor
    
    return B