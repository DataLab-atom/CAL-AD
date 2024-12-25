from numpy.linalg import inv
from quadratic_function import QuadraticFunction
import numpy as np
def search_root(qf: QuadraticFunction, x0: np.ndarray, tol: float = 1e-6, max_iter: int = 1000, k: int = 5) -> np.ndarray:
    """
    Implements a revised optimization algorithm to minimize a quadratic function with enhancements
    for robustness and efficiency.

    Parameters:
    - qf: An instance of QuadraticFunction, providing the properties and methods required for optimization.
    - x0: The initial point as a NumPy array.
    - tol: Tolerance for convergence.
    - max_iter: Specify the maximum number of iterations before stopping.
    - k: Parameter affecting direction calculations (not used explicitly in this version).

    Returns:
    - The point approximated to minimize the target quadratic function.
    """
    d = qf.d  # Dimension of the optimization variable
    n = qf.n  # Number of quadratic functions

    x = x0.copy()  # Starting point
    B = [np.eye(d) for _ in range(n)]  # Initiating approximative Hessian matrices to identity
    history = []  # Optional: Store history for monitoring convergence
    
    for t in range(max_iter):
        it = t % n  # Choose the quadratic function in a cyclic manner

        # Compute the aggregate gradient and Hessian approximation
        grad_aggregate = np.zeros(d)
        for i in range(n):
            grad_aggregate += qf.gradient(x)  # Accumulate gradients
            
        B_agg = np.mean(B, axis=0)  # Average Hessian approximation
        B_inv = np.linalg.inv(B_agg)  # Inverted aggregate Hessian

        delta_x = -np.dot(B_inv, grad_aggregate)  # Direction for optimization

        # Implement backtracking line search for robust step size
        alpha = 1.0  # Initial full step
        while alfa > 1e-8:
            x_new = x + alpha * delta_x  # New proposed point
            if qf.objective_function(x_new) < qf.objective_function(x):  # Check for improvement
                break
            alpha *= 0.5  # Halve step size if no improvement

        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged at iteration {t+1} with point: {x_new}")
            return x_new
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