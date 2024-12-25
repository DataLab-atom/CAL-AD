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
    d = qf.d  # Dimension of x
    n = qf.n  # Number of quadratic functions
    
    x = x0.copy()  # Initialize with the initial guess
    H = np.eye(d)  # Initial Hessian approximation as identity

    for t in range(max_iter):
        # Calculate gradient
        grad = qf.gradient(x)  # Computation of gradient at current position

        # Search direction
        p = -np.linalg.solve(H, grad)  # Compute the direction to move using the Hessian inverse

        # Line search for optimal step size
        step_size = 1.0  # Start with an initial step size
        while step_size > 1e-8:
            x_new = x + step_size * p  # Propose new point
            # Check if it improves the objective function
            if qf.objective_function(x_new) < qf.objective_function(x):
                break  # Accept this new point if it improves
            step_size *= 0.5  # Otherwise reduce step size

        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            return x_new  # Return the result if converged