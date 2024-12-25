import numpy as np
from typing import List, Tuple

def generate_A_b(xi: float, d: int, n: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate A_i and b_i matrices/vectors according to the given parameters, ensuring the objective function value is always greater than zero.
    
    Parameters:
        xi (float): Parameter affecting the condition number.
        d (int): Dimension.
        n (int): Number of samples to generate.
    
    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Lists containing all A_i matrices and all b_i vectors.
    """
    A_list: List[np.ndarray] = []
    b_list: List[np.ndarray] = []
    
    for _ in range(n):
        # Generate diagonal entries for A_i
        diag_entries: np.ndarray = np.zeros(d)
        # First half of diagonal entries
        diag_entries[:d//2] = np.random.uniform(1, 10**(xi / 2), size=d//2)
        # Second half of diagonal entries
        diag_entries[d//2:] = np.random.uniform(10**(-xi / 2), 1, size=d - d//2)
        
        # Construct the diagonal matrix
        A_i: np.ndarray = np.diag(diag_entries)
        A_list.append(A_i)
        
        # Generate b_i
        b_i: np.ndarray = np.random.uniform(0, 10**3, size=d)
        b_list.append(b_i)
    
    return A_list, b_list

def objective_function(x: np.ndarray, A_list: List[np.ndarray], b_list: List[np.ndarray]) -> float:
    """
    Compute the value of the objective function f(x).
    
    f(x) = \frac{1}{n} \sum_{i=1}^n \left( \frac{1}{2} \langle x, A_i x \rangle + \langle b_i, x \rangle \right)

    Parameters:
        x (np.ndarray): Input vector of shape (d,).
        A_list (List[np.ndarray]): List of all A_i matrices, each of shape (d,d).
        b_list (List[np.ndarray]): List of all b_i vectors, each of shape (d,).
    
    Returns:
        float: The value of the objective function f(x).
    """
    n: int = len(A_list)  # Get the number of samples
    d: int = x.shape[0]   # Get the dimension of vector x
    
    # Initialize the objective function value
    f_x: float = 0.0
    
    # Iterate over all samples to compute the objective function value
    for i in range(n):
        A_i: np.ndarray = A_list[i]
        b_i: np.ndarray = b_list[i]
        
        # Compute the quadratic and linear terms
        quadratic_term: float = 0.5 * np.dot(x.T, np.dot(A_i, x))
        linear_term: float = np.dot(b_i, x)
        
        # Accumulate each term to the total objective function value
        f_x += quadratic_term + linear_term
    
    # Average the objective function value over all samples
    f_x /= n
    
    return f_x

