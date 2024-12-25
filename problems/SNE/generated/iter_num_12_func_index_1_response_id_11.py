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
def search_root(quadratic_func: QuadraticFunction, initial_point: np.ndarray, tolerance: float = 1e-6, max_iterations: int = 1000) -> np.ndarray:
    """
    Implements the LISR-1 optimization algorithm to find the minimum of a quadratic function.
    
    Parameters:
    - quadratic_func: An instance of QuadraticFunction.
    - initial_point: The initial point for optimization.
    - tolerance: Tolerance for convergence.
    - max_iterations: Maximum number of iterations.
    
    Returns:
    - The point that minimizes the target function.
    """

    num_dimensions = quadratic_func.dimensions
    num_components = quadratic_func.components
    
    average_hessian_inv = np.linalg.inv(quadratic_func.average_hessian)
    inverse_hessian = average_hessian_inv.copy()
    
    x = initial_point.copy()
    point_history = [initial_point.copy() for _ in range(num_components)]
    hessian_approximations = [average_hessian_inv.copy() for _ in range(num_components)]
    
    phi = np.sum([hessian_approximations[i] @ point_history[i] for i in range(num_components)], axis=0) / num_components - quadratic_func.gradient(x)
    g = np.sum([quadratic_func.hessians[i] @ point_history[i] + quadratic_func.biases[i] for i in range(num_components)], axis=0) / num_components
    
    previous_obj_value = quadratic_func.evaluate_objective_function(initial_point)
    
    for iteration in range(max_iterations):
        index_t = iteration % num_components
        
        direction = inverse_hessian @ (phi - g)
        step_size = backtracking_line_search(quadratic_func, x, direction)
        
        new_point = x + step_size * direction
        obj_value = quadratic_func.evaluate_objective_function(new_point)
        
        if obj_value > previous_obj_value - tolerance:
            print(f"Warning: Limited improvement at iteration {iteration}. Current Obj Val: {obj_value}, Previous: {previous_obj_value}.")
        
        previous_obj_value = obj_value
        
        if np.linalg.norm(quadratic_func.gradient(new_point)) < tolerance:
            print(f"Converged at iteration {iteration} with tolerance {tolerance}")
            break
        
        gradient_new = quadratic_func.hessians[index_t] @ new_point + quadratic_func.biases[index_t]
        gradient_old = quadratic_func.hessians[index_t] @ point_history[index_t] + quadratic_func.biases[index_t]
        gradient_diff = gradient_new - gradient_old
        
        step = new_point - point_history[index_t]
        hessian_approximations[index_t] = sr1_update(hessian_approximations[index_t], gradient_diff, step)
        
        u = gradient_diff
        v = hessian_approximations[index_t] @ step
        inverse_hessian = sherman_morrison_update(inverse_hessian, u, v)
        
        point_history[index_t] = new_point.copy()
        phi += hessian_approximations[index_t] @ point_history[index_t] - hessian_approximations[index_t] @ step
        g = np.sum([quadratic_func.hessians[i] @ point_history[i] + quadratic_func.biases[i] for i in range(num_components)], axis=0) / num_components
        
        x = new_point
    
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