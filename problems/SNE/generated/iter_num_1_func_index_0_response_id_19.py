from numpy.linalg import inv
from quadratic_function import QuadraticFunction
import numpy as np
def search_root(qf: QuadraticFunction, x0: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """
    A refined version of the `search_root` function for optimizing
    a activity-block structure while searching for the minimum of 
    a quadratic function specified by the QuadraticFunction object.

    Parameters:
    - qf: A QuadraticFunction instance.
    - x0: Initial guess for the minimizer (np.ndarray).
    - tol: Tolerance for convergence (float).
    - max_iter: Maximum number of iterations (int).

    Returns:
    - The optimal point x that minims the quadratic function.
    """
    d = qf.d  # Dimension of the variable x
    n = qf.n  # Number of quadratic contributions
    
    x = np.copy(x0)  # Initialize optimization
    B = np.eye(d)  # Approximate Hessian / inverse 
       
    for iter in range(max_iter):
        # Compute the gradient at the current point
        grad = sum(qf.gradient(x))

        # Compute the search direction
        direction = -np.linalg.solve(B, grad)

        # Line search for optimal step size
        alpha = 1.0
        while alpha > 1e-8:
            x_new = x + alpha * direction
            if qf.objective_function(x_new) < qf.objective_function(x):
                break
            alpha *= 0.5  # Decrease step size
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged at iteration {iter}")
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