from numpy.linalg import inv
from quadratic_function import QuadraticFunction
import numpy as np
def search_root(qf: QuadraticFunction, x0: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """
    Enhanced LISR-1 optimization algorithm to find the minimum of a given quadratic function.
    
    Parameters:
    - qf: An instance of QuadraticFunction.
    - x0: The initial point as a numpy array.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.
    
    Returns:
    - The point that minimizes the target function.
    """
    d = qf.d
    n = qf.n
    
    x = x0.copy()
    B_inv = np.eye(d)  # Initialize approximate Hessian inverse to identity
    momentum = np.zeros(d)  # Initialize momentum term
    
    for t in range(max_iter):
        # Compute the aggregate gradient
        grad = sum(qf.gradient(x) for _ in range(n)) / n
        
        # Compute next step direction
        delta_x = -B_inv @ grad
        
        # Adaptive step size with momentum
        step_size = 1.0
        prev_obj = qf.objective_function(x)
        
        # Backtracking line search with momentum
        while step_size > 1e-8:
            x_new = x + step_size * delta_x + momentum
            
            if qf.objective_function(x_new) < prev_obj:
                break  # Found a valid improvement
            
            step_size *= 0.5  # Reduce step size if no improvement
        
        if np.linalg.norm(x_new - x) < tol:
            return x_new  # Converged to solution
def search_root(qf: QuadraticFunction, x0: np.ndarray, tol: float = 1e-6, max_iter: int = 1000, k: int = 5) -> np.ndarray:
    d = qf.d  # Dimension of x
    n = qf.n  # Number of quadratic functions
    
    x = x0.copy()  # Initial guess
    B = [np.eye(d) for _ in range(n)]  # Start with identity matrices for approximation
    
    for t in range(max_iter):
        it = t % n  # Index mod n for cyclic updates
        
        # Aggregate Hessian and gradient
        B_agg = sum(B)
        g_agg = sum([qf.gradient(x) for _ in range(n)])  # Aggregate gradient using current x
        
        B_agg_inv = np.linalg.inv(B_agg)
        delta_x = -np.dot(B_agg_inv, g_agg)

        # Line search or step scaling to ensure we're moving towards improvement
        step_size = 1.0
        while step_size > 1e-8:
            x_new = x + step_size * delta_x
            if qf.objective_function(x_new) < qf.objective_function(x):
                break  # We found an improving step
            step_size *= 0.5  # Reduce step size if no improvement

        # If no improvement is found, likely near a bad local area �� can stop
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged at iteration {t}")
            return x_new

        x = x_new  # Update current position

        # SR1 update formula
        e_i = np.eye(d)[:, t % d]
        u = np.dot(qf.A[it] - B[it], e_i)
        v = np.dot(B[it], u)

        # Avoiding division by near zero for stability
        denom = np.dot(u.T, v)
        if np.abs(denom) > 1e-12:  # Only perform the update if the denominator is stable
            B[it] += np.outer(v, v) / denom

    print("Reached max iteration without full convergence.")
    return x
def search_root(qf: QuadraticFunction, x0: np.ndarray, tol: float = 1e-6, max_iter: int = 1000, k: int = 5) -> np.ndarray:
    """
    An enhanced version of the LISR-1 optimization algorithm to find the minimum
    of a given quadratic function, with improved convergence and stability.
    
    Parameters:
    - qf: An instance of QuadraticFunction.
    - x0: The initial point as a numpy array.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.
    - k: Number of elements in subset for greedy direction calculations (optional).
    
    Returns:
    - The point that minimizes the target function.
    """
    d = qf.d
    n = qf.n
    
    x = x0.copy()
    B = [np.eye(d) for _ in range(n)] 
    B_inv = [np.eye(d) for _ in range(n)] 
    
    for t in range(max_iter):
        it = t % n 
        
        # Aggregate central terms for the next exponent-based update
        A_avg = sum(qf.A) / n
        grad = qf.gradient(x)
        
        # Pdf alpha for shrinkable rate from last learned diff
        B_inv[0] = np.linalg.inv(A_avg)
        
        # Update x using weighted descent based on aggregated results
        delta_x = -B_inv[it] @ grad
        step_size = 1
        
        while True:
            x_new = x + step_size * delta_x
            if qf.objective_function(x_new) < qf.objective_function(x):
                x = x_new
                break  # Valid improvement
            step_size *= 0.5 
            if step_size < 1e-8:  # Prevent infinite reduction
                break

        if np.linalg.norm(delta_x) < tol:
            print(f"Converged at iteration {t}")
            return x