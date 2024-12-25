from numpy.linalg import inv
from quadratic_function import QuadraticFunction
import numpy as np
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

        # If no improvement is found, likely near a bad local area ¡ª can stop
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

        # If no improvement is found, likely near a bad local area ¡ª can stop
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
    Optimizes a quadratic function using the LISR-1 algorithm with enhancements for stability and efficiency.
    
    Parameters:
    - qf: An instance of QuadraticFunction.
    - x0: The initial point.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.
    - k: Number of elements in subset for greedy direction calculations (optional).
    
    Returns:
    - The point that minimizes the target function.
    """
    d = qf.d  # Dimension of x
    n = qf.n  # Number of quadratic functions
    
    x = x0.copy()  # Starting guess for minimization
    B = np.eye(d)  # Initial Hessian approximation - single matrix for simplicity
    curr_iter = 0
    
    for curr_iter in range(max_iter):
        # Compute aggregate gradient
        grad = np.sum([qf.gradient(x) for _ in range(n)], axis=0)
        
        # Compute the direction for optimization
        delta_x = -np.linalg.solve(B, grad)

        # Implement a simple backtracking line search for ensuring a proper descent
        step_size = 1.0
        while step_size > 1e-8:
            x_new = x + step_size * delta_x
            if qf.objective_function(x_new) < qf.objective_function(x):
                break  # Step is accepted, improves the objective
            step_size *= 0.5  # Shrink the step size
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged after {curr_iter} iterations")
            return x_new