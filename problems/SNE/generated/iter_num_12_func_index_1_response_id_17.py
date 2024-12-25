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
    - initial_point: The initial point to start optimization.
    - tolerance: Tolerance for convergence.
    - max_iterations: Maximum number of iterations.

    Returns:
    - The point that minimizes the target quadratic function.
    """

    dimension = quadratic_func.d
    components = quadratic_func.n
    
    # Initialize the inverse of the average Hessian matrix A_avg
    average_hessian_inv = np.linalg.inv(quadratic_func.A_avg)
    inv_hessian_b = average_hessian_inv.copy()  # Start with B_inv as the inverse of the average Hessian
    
    # Initialize state variables
    current_point = initial_point.copy()
    z_vectors = [initial_point.copy() for _ in range(components)]  # Copy of x for each component
    hessian_approximations = [average_hessian_inv.copy() for _ in range(components)]  # Initialize all Hessian approximations
    
    # Initialize auxiliary variables
    phi = np.sum([hessian_approximations[i] @ z_vectors[i] for i in range(components)], axis=0) / components - quadratic_func.gradient(current_point)
    g = np.sum([quadratic_func.A[i] @ z_vectors[i] + quadratic_func.b[i] for i in range(components)], axis=0) / components
    
    previous_obj_value = quadratic_func.objective_function(initial_point)

    for iteration in range(max_iterations):
        component_index = iteration % components
        
        # Compute the current iteration's new direction
        direction = inv_hessian_b @ (phi - g)

        # Perform backtracking line search to ensure descent
        step_size = backtracking_line_search(quadratic_func, current_point, direction)
        
        # Update current_point with the chosen step size
        new_point = current_point + step_size * direction

        # Check if the new objective value is better, after the line search
        obj_value = quadratic_func.objective_function(new_point)
        
        # Check progress and convergence
        if obj_value > previous_obj_value - tolerance:
            print(f"Warning: Minimal improvement at iteration {iteration}. Current Obj Val: {obj_value}, Previous: {previous_obj_value}.")
        
        previous_obj_value = obj_value

        if np.linalg.norm(quadratic_func.gradient(new_point)) < tolerance:
            print(f"Converged at iteration {iteration} with tolerance {tolerance}")
            break
        
        # Compute Hessian and step difference
        grad_new = quadratic_func.A[component_index] @ new_point + quadratic_func.b[component_index]
        grad_old = quadratic_func.A[component_index] @ z_vectors[component_index] + quadratic_func.b[component_index]
        grad_difference = grad_new - grad_old
        
        step_vector = new_point - z_vectors[component_index]
        hessian_approximations[component_index] = sr1_update(hessian_approximations[component_index], grad_difference, step_vector)  # Update the Hessian approx.
        
        # Sherman-Morrison update for the inverse of the total Hessian
        u_update = grad_difference
        v_update = hessian_approximations[component_index] @ step_vector
        inv_hessian_b = sherman_morrison_update(inv_hessian_b, u_update, v_update)

        # Update z[i]
        z_vectors[component_index] = new_point.copy()

        # Update the auxiliary variables phi and g
        phi += hessian_approximations[component_index] @ z_vectors[component_index] - hessian_approximations[component_index] @ step_vector
        g = np.sum([quadratic_func.A[i] @ z_vectors[i] + quadratic_func.b[i] for i in range(components)], axis=0) / components

        # Set current_point to the new value for the next iteration
        current_point = new_point

    return current_point
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