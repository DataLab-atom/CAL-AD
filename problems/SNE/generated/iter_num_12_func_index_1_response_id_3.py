from quadratic_function import QuadraticFunction
import numpy as np
def backtracking_line_search(quadratic_func: QuadraticFunction, x: np.ndarray, direction: np.ndarray, alpha: float = 0.4, beta: float = 0.7) -> float:
    """
    Perform backtracking line search to ensure sufficient decrease in the objective function.
    
    Parameters:
    - quadratic_func: Instance of the QuadraticFunction.
    - x: Current point.
    - direction: The search direction (typically the step computed during optimization).
    - alpha: The parameter controlling the Armijo condition (default 0.4).
    - beta: The reduction factor for the step size (default 0.7).
    
    Returns:
    - Optimal step size.
    """
    t = 1  # Initial step size
    f_x = quadratic_func.objective_function(x)
    grad_x = quadratic_func.gradient(x)

    # Backtrack by reducing step size until a sufficient decrease condition is met
    while quadratic_func.objective_function(x + t * direction) > f_x + alpha * t * grad_x.T @ direction:
        t *= beta

    return t
def search_root(quadratic_func, x0, tol=1e-6, max_iter=1000):
    """
    Implement the LISR-1 optimization algorithm with enhanced stability and convergence properties.
    
    Parameters:
    - quadratic_func : Instance of the QuadraticFunction.
    - x0 : Initial point for optimization.
    - tol : Convergence tolerance (default 1e-6).
    - max_iter : Maximum number of iterations (default 1000).
    
    Returns:
    - Optimized point minimizing the quadratic function.
    """
    
    d = quadratic_func.d
    n = quadratic_func.n
    A_inv = np.linalg.inv(quadratic_func.A_avg)  # Inverse of the average Hessian approximation
    B_inv = A_inv.copy()  # Start with B_inv as the inverse of the average Hessian
    x = x0.copy()
    z = [x0.copy() for _ in range(n)]  # Copy of x for each component
    B = [A_inv.copy() for _ in range(n)]  # Initialize all Hessian approximations
    phi = np.sum([B[i] @ z[i] for i in range(n)], axis=0) / n - quadratic_func.gradient(x)
    g = np.sum([quadratic_func.A[i] @ z[i] + quadratic_func.b[i] for i in range(n)], axis=0) / n
    obj_val_prev = quadratic_func.objective_function(x0)

    for t in range(max_iter):
        i_t = t % n
        direction = B_inv @ (phi - g)

        step_size = backtracking_line_search(quadratic_func, x, direction)
        x_new = x + step_size * direction
        obj_val = quadratic_func.objective_function(x_new)

        if obj_val > obj_val_prev - tol:
            print(f"Warning: Limited improvement at iteration {t}. Current Obj Val: {obj_val}, Previous: {obj_val_prev}.")

        obj_val_prev = obj_val

        if np.linalg.norm(quadratic_func.gradient(x_new)) < tol:
            print(f"Converged at iteration {t} with tolerance {tol}")
            break

        grad_new = quadratic_func.A[i_t] @ x_new + quadratic_func.b[i_t]
        grad_old = quadratic_func.A[i_t] @ z[i_t] + quadratic_func.b[i_t]
        grad_diff = grad_new - grad_old
        s = x_new - z[i_t]
        
        B[i_t] = sr1_update(B[i_t], grad_diff, s)  # Update the Hessian approx.
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
    Perform the SR1 update on the Hessian approximation.
    
    Parameters:
    - B: Current Hessian approximation.
    - grad_diff: The difference in gradients.
    - s: The step vector used to compute the update.
    
    Returns:
    - Updated Hessian approximation.
    """
    Bs = B @ s
    diff = grad_diff - Bs
    denom = s.T @ diff

    if np.abs(denom) > 1e-8:  # To avoid division by zero or near-zero values
        B += np.outer(diff, diff) / denom
    
    return B