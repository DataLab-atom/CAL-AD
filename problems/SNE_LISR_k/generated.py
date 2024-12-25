from typing import List, Callable
import numpy as np

def objective_function(x: np.ndarray, A_list: List[np.ndarray], b_list: List[np.ndarray]) -> float:
    """Compute the value of the objective function f(x)."""
    n: int = len(A_list)
    f_x: float = 0.0
    for i in range(n):
        f_x += 0.5 * np.dot(x.T, np.dot(A_list[i], x)) + np.dot(b_list[i], x)
    return f_x / n

def srk(G: np.ndarray, A: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Compute the Symmetric Rank-k (SR-k) update."""
    if np.allclose(G @ U, A @ U):
        return G
    else:
        temp = U.T @ (G - A) @ U
        if np.linalg.matrix_rank(temp) < temp.shape[0]:  # Handle potential singularity
            temp = temp + 1e-6 * np.eye(temp.shape[0])
        return G - (G - A) @ U @ np.linalg.inv(temp) @ U.T @ (G - A)

def greedy_matrix(G: np.ndarray, A: np.ndarray, k: int) -> np.ndarray:
    """Select the greedy matrix U based on the largest k diagonal entries."""
    diff = np.diag(G - A)
    indices = np.argsort(diff)[::-1][:k]
    U = np.eye(G.shape[0])[:, indices]
    return U


def search_root(objective_function: callable, x0: np.ndarray, A_list: List[np.ndarray], b_list: List[np.ndarray],
                tol: float = 1e-6, max_iter: int = 1000, k: int = 5) -> np.ndarray:
    '''Implements the LISR-k optimization algorithm.'''
    n = len(A_list)
    d = x0.shape[0]
    z_list = [x0.copy() for _ in range(n)]
    B_list = [np.eye(d) for _ in range(n)]
    B_bar = n * np.eye(d)
    B_bar_inv = np.eye(d) / n
    x = x0.copy()

    for t in range(max_iter):
        i_t = t % n
        grad_sum = sum(np.dot(A, z) + b for A, z, b in zip(A_list, z_list, b_list))
        x = B_bar_inv @ grad_sum

        omega = 1.0  # Initialize omega

        U = greedy_matrix(B_list[i_t], A_list[i_t], k)
        B_new = srk(B_list[i_t], A_list[i_t], U)
        
        V = (B_list[i_t] - A_list[i_t]) @ U

        if np.linalg.matrix_rank(U.T @ V) < U.shape[1]:
            D = U.T @ V - V.T @ B_bar_inv @ V + 1e-6 * np.eye(U.shape[1]) # Ensure invertibility
        else:
            D = U.T @ V - V.T @ B_bar_inv @ V
        
        B_bar_inv = B_bar_inv + B_bar_inv @ V @ np.linalg.inv(D) @ V.T @ B_bar_inv


        B_list[i_t] = B_new
        z_list[i_t] = x.copy()



        if np.linalg.norm(x - z_list[i_t]) < tol:
            break

    return x


if __name__ == "__main__":
    # Test code here
    d = 50
    n = 1000
    A_list = [np.eye(d) * (i + 1) for i in range(n)]
    b_list = [np.ones(d) * i for i in range(n)]
    x0 = np.zeros(d)

    x_opt = search_root(objective_function, x0, A_list, b_list, k=5)
    print(f"Optimal x: {x_opt}")
    print(f"Objective function value at optimal x: {objective_function(x_opt, A_list, b_list)}")


