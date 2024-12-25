import numpy as np
from typing import List
from numpy.linalg import inv, norm, pinv
from quadratic_function import QuadraticFunction

def get_top_k_standard_basis(G: np.ndarray, A: np.ndarray, k: int) -> np.ndarray:
    """
    Selects the top k standard basis vectors corresponding to the largest k diagonal entries of (G - A).

    Parameters:
    - G: Current Hessian estimator matrix.
    - A: Actual Hessian matrix.
    - k: Number of top basis vectors to select.

    Returns:
    - U: A d x k matrix where each column is a standard basis vector.
    """
    diag_diff = np.diag(G - A)
    topk_indices = np.argsort(diag_diff)[-k:]
    d = G.shape[0]
    U = np.zeros((d, k))
    for idx, i in enumerate(topk_indices):
        U[i, idx] = 1.0
    return U

def srk_update(G: np.ndarray, A: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Performs the Symmetric Rank-k (SR-k) update.

    Parameters:
    - G: Current Hessian estimator matrix.
    - A: Actual Hessian matrix.
    - U: Matrix of directions for the SR-k update.

    Returns:
    - G_new: Updated Hessian estimator matrix after SR-k.
    """
    S = G - A
    SU = S @ U
    UTSU = U.T @ SU
    try:
        UTSU_inv = inv(UTSU)
    except np.linalg.LinAlgError:
        UTSU_inv = pinv(UTSU)
    G_new = G - SU @ UTSU_inv @ SU.T
    return G_new

def sherman_morrison_woodbury(A_inv: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Updates the inverse of a matrix using the Sherman-Morrison-Woodbury formula.

    Parameters:
    - A_inv: Current inverse of the matrix A.
    - U: Matrix used in the rank-k update.
    - V: Matrix used in the rank-k update.

    Returns:
    - A_inv_new: Updated inverse matrix after applying Sherman-Morrison-Woodbury.
    """
    W = U.T @ V
    try:
        W_inv = inv(W - U.T @ A_inv @ V)
    except np.linalg.LinAlgError:
        W_inv = pinv(W - U.T @ A_inv @ V)
    A_inv_new = A_inv + A_inv @ V @ W_inv @ U.T @ A_inv
    return A_inv_new

def sr1_update(G: np.ndarray, A: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Performs the Symmetric Rank-1 (SR1) update.

    Parameters:
    - G: Current Hessian estimator matrix.
    - A: Actual Hessian matrix.
    - u: Direction vector for the SR1 update.

    Returns:
    - G_new: Updated Hessian estimator matrix after SR1.
    """
    S = G - A
    Su = S @ u
    denom = u.T @ Su
    if denom == 0:
        return G
    G_new = G - np.outer(Su, Su) / denom
    return G_new

def srk_update_block(G: np.ndarray, A: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Performs the Symmetric Rank-k (SR-k) update for block updates.

    Parameters:
    - G: Current Hessian estimator matrix.
    - A: Actual Hessian matrix.
    - U: Matrix of directions for the SR-k update.

    Returns:
    - G_new: Updated Hessian estimator matrix after SR-k.
    """
    return srk_update(G, A, U)

def sr1_update_greedy(G: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Performs the greedy SR1 update by selecting the direction that maximizes u.T (G - A) u.

    Parameters:
    - G: Current Hessian estimator matrix.
    - A: Actual Hessian matrix.

    Returns:
    - G_new: Updated Hessian estimator matrix after greedy SR1.
    """
    d = G.shape[0]
    diagonal_diff = np.diag(G - A)
    max_idx = np.argmax(diagonal_diff)
    u = np.zeros(d)
    u[max_idx] = 1.0
    return sr1_update(G, A, u)

def search_root(qf: 'QuadraticFunction', x0: np.ndarray, tol: float=1e-06, max_iter: int=1000, k: int=1) -> np.ndarray:
    """
    Implements the LISR-k optimization algorithm to find the minimum of a given quadratic function.

    Parameters:
    - qf: An instance of QuadraticFunction.
    - x0: The initial point (numpy array of shape (d,)).
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.
    - k: The rank parameter for the SR-k update.

    Returns:
    - The point that minimizes the target function.
    """
    n, d = (qf.n, qf.d)
    z = [x0.copy() for _ in range(n)]
    alpha_0 = 1.0
    B = [(1 + alpha_0) ** 2 * qf.A[i] for i in range(n)]
    phi = sum([B[i] @ z[i] for i in range(n)])
    grad = [qf.A[i] @ z[i] + qf.b[i] for i in range(n)]
    g = sum(grad)
    BarB = sum(B)
    BarB_inv = inv(BarB)
    for t in range(max_iter):
        x = BarB_inv @ (phi - g)
        current_grad = qf.gradient(x)
        grad_norm = norm(current_grad)
        if grad_norm < tol:
            return x
        i_t = t % n
        z_old = z[i_t].copy()
        z[i_t] = x.copy()
        g = g - (qf.A[i_t] @ z_old + qf.b[i_t]) + (qf.A[i_t] @ x + qf.b[i_t])
        if t % n == 0 and t != 0:
            omega = 1.1
        else:
            omega = 1.0
        G = B[i_t]
        A = qf.A[i_t]
        U = get_top_k_standard_basis(G, A, k)
        B_new = srk_update_block(G, A, U)
        B_new = omega * B_new
        phi = phi + B_new @ x - B[i_t] @ z_old
        delta_B = B_new - B[i_t]
        if k == 1:
            v = delta_B @ U
            u = U[:, 0].reshape(-1, 1)
            w = U[:, 0].T @ v
            if w - (v.T @ BarB_inv @ v)[0, 0] != 0:
                BarB_inv = BarB_inv + BarB_inv @ v @ v.T @ BarB_inv / (w - (v.T @ BarB_inv @ v)[0, 0])
        else:
            V = delta_B @ U
            BarB_inv = sherman_morrison_woodbury(BarB_inv, U, V)
        B[i_t] = B_new.copy()
        BarB = BarB + delta_B.copy()
    return x