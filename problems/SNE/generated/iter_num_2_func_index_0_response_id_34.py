from quadratic_function import QuadraticFunction
import numpy as np
def backtracking_line_search(quadratic_func, x, direction, alpha=0.4, beta=0.7):
    """
    Backtracking line search to get optimal step size.
    
    Parameters:
    - quadratic_func: Objective function to optimize.
    - x: Current position in the search space.
    - direction: Search direction for optimization.
    - alpha: Armijo condition parameter.
    - beta: Reduction factor for the step size.
    
    Returns:
    - Optimal step size.
    
    Note:
    Here, we utilize SciPy's minimize function with bounds to perform backtracking line search efficiently.
    """

    def obj_func(t):
        return quadratic_func(x + t * direction)
def search_root(quadratic_func: QuadraticFunction, x0: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """
    Implements the LISR-1 optimization algorithm to find the minimum of a quadratic function.

    Parameters:
    - quadratic_func: An instance of QuadraticFunction.
    - x0: The initial point.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.

    Returns:
    - The point that minimizes the target function.
    """

    d = quadratic_func.d
    n = quadratic_func.n
    
    # Initialize inversion of average Hessian matrix A_avg
    A_avg_inv = np.linalg.inv(quadratic_func.A_avg)
    B_inv = A_avg_inv.copy()  # Start with B_inv as the inverse of the average Hessian

    # Initialize the state variables
    x = x0.copy()
    z = [x0.copy() for _ in range(n)]  # Copy of x for each component
    B = [A_avg_inv.copy() for _ in range(n)]  # Initialize all Hessian approximations

    # Auxiliary variables. Initial guess for phi and g.
    phi = np.sum([B[i] @ z[i] for i in range(n)], axis=0) / n - quadratic_func.gradient(x)
    g = np.sum([quadratic_func.A[i] @ z[i] + quadratic_func.b[i] for i in range(n)], axis=0) / n

    prev_obj_val = quadratic_func.objective_function(x0)

    for t in range(max_iter):
        i_t = t % n
        
        # Compute the current iteration's new direction
        direction = B_inv @ (phi - g)

        # Perform backtracking line search to ensure descent
        step_size = backtracking_line_search(quadratic_func, x, direction)
        
        # Update x with the chosen step size
        x_new = x + step_size * direction

        # Check if the new objective value is better (it should after the line search)
        obj_val = quadratic_func.objective_function(x_new)
        
        # Check if we're actually making progress
        if obj_val > prev_obj_val - tol:
            print(f"Warning: Minimal improvement at iteration {t}. Current Obj Val: {obj_val}, Previous: {prev_obj_val}.")
        
        prev_obj_val = obj_val

        # Check for convergence
        if np.linalg.norm(quadratic_func.gradient(x_new)) < tol:
            print(f"Converged at iteration {t} with tolerance {tol}")
            break
        
        # Compute Hessian and step difference
        grad_new = quadratic_func.A[i_t] @ x_new + quadratic_func.b[i_t]
        grad_old = quadratic_func.A[i_t] @ z[i_t] + quadratic_func.b[i_t]
        grad_diff = grad_new - grad_old
        
        s = x_new - z[i_t]
        B[i_t] = sr1_update(B[i_t], grad_diff, s)  # Update the Hessian approx.
        
        # Sherman-Morrison update for the inverse of the total Hessian
        u = grad_diff
        v = B[i_t] @ s
        B_inv = sherman_morrison_update(B_inv, u, v)

        # Update z[i]
        z[i_t] = x_new.copy()

        # Update the auxiliary variables phi and g
        phi += B[i_t] @ z[i_t] - B[i_t] @ s
        g = np.sum([quadratic_func.A[i] @ z[i] + quadratic_func.b[i] for i in range(n)], axis=0) / n

        # Set x to the new value for the next iteration
        x = x_new

    return x
def sherman_morrison_update(B_inv: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Update the inverse of a matrix B efficiently using Sherman-Morrison formula(v2).
    
    Parameters:
    - B_inv: Current inverse of the matrix B.
    - u: The vector u.
    - v: The vector v.
    
    Returns:
    - Updated inverse of B using Sherman-Morrison.
    """    
    Bu = B_inv @ u
    uTBu = u.T @ Bu

    if np.linalg.norm(uTBu) < 1e-12:  # Handling numerically close to zero cases
        return B_inv
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