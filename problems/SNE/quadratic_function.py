import numpy as np

class QuadraticFunction:
    def __init__(self, d, n, xi, kappa):
        """
        Initialize the quadratic function with given parameters.
        
        Parameters:
        - d: Dimension of the vector x.
        - n: Number of terms in the sum.
        - xi: Parameter affecting the condition number of the problem.
        - kappa: Condition number of the problem.
        """
        self.d = d
        self.n = n
        self.xi = xi
        self.kappa = kappa
        self.A, self.b = self.generate_matrices()
        self.A_avg = self.compute_average_A()
        self.L = self.compute_L()
        self.mu = self.compute_mu()
        self.M = self.compute_M()
        self.rho = self.kappa  # Use the provided condition number
        self.r0 = self.compute_r0(np.random.rand(d))  # Initial guess for x0

    def generate_matrices(self):
        """
        Generate the matrices A_i and vectors b_i for the quadratic function minimization problem.
        
        Returns:
        - A: List of n diagonal matrices A_i.
        - b: List of n vectors b_i.
        """
        A = []
        b = []
        
        # Generate A_i
        for _ in range(self.n):
            diag_A = np.zeros(self.d)
            # Ensure the condition number is close to kappa
            max_val = 10 ** (self.xi / 2)
            min_val = max_val / self.kappa
            diag_A[:self.d // 2] = np.random.uniform(min_val, max_val, size=self.d // 2)
            diag_A[self.d // 2:] = np.random.uniform(min_val, max_val, size=self.d - self.d // 2)
            np.random.shuffle(diag_A)  # Shuffle to avoid any potential bias
            A.append(np.diag(diag_A))
        
        # Generate b_i
        for _ in range(self.n):
            b.append(np.random.uniform(0, 10 ** 3, size=self.d))
        
        return A, b

    def compute_average_A(self):
        """
        Compute the average of the matrices A_i.
        
        Returns:
        - Average matrix A.
        """
        A_avg = np.zeros((self.d, self.d))
        for i in range(self.n):
            A_avg += self.A[i]
        return A_avg / self.n

    def compute_L(self):
        """
        Compute the Lipschitz constant L of the gradient.
        
        Returns:
        - L: Lipschitz constant of the gradient.
        """
        eigenvalues = np.linalg.eigvals(self.A_avg)
        return np.max(eigenvalues)

    def compute_mu(self):
        """
        Compute the strong convexity parameter μ.
        
        Returns:
        - μ: Strong convexity parameter.
        """
        eigenvalues = np.linalg.eigvals(self.A_avg)
        return np.min(eigenvalues)

    def compute_M(self):
        """
        Compute the Lipschitz constant M of the Hessian.
        
        Returns:
        - M: Lipschitz constant of the Hessian.
        """
        return self.L / (self.mu ** 1.5)

    def compute_r0(self, x0):
        """
        Compute the initial distance r0 to the optimum.
        
        Parameters:
        - x0: Initial guess for x.
        
        Returns:
        - r0: Initial distance to the optimum.
        """
        A_inv = np.linalg.inv(self.A_avg)
        x_star = -A_inv @ np.sum(self.b, axis=0) / self.n
        return np.linalg.norm(x0 - x_star)

    def objective_function(self, x):
        """
        Compute the value of the objective function f(x).
        
        Parameters:
        - x: The vector x in R^d.
        
        Returns:
        - Value of the objective function at x.
        """
        total = 0.0
        for i in range(self.n):
            Ax = self.A[i] @ x
            total += 0.5 * x.T @ Ax + self.b[i].T @ x
        return total / self.n

    def gradient(self, x):
        """
        Compute the gradient of the objective function f(x).
        
        Parameters:
        - x: The vector x in R^d.
        
        Returns:
        - Gradient of the objective function at x.
        """
        grad = np.zeros(self.d)
        for i in range(self.n):
            grad += self.A[i] @ x + self.b[i]
        return grad / self.n
    
if __name__ == "__main__":
    # Example usage
    d = 20
    n = 10
    xi = 4
    kappa = 3.03e2  # Condition number
    quadratic_func = QuadraticFunction(d, n, xi, kappa)

    # Print parameters
    print(f"Lipschitz constant L: {quadratic_func.L}")
    print(f"Strong convexity parameter μ: {quadratic_func.mu}")
    print(f"Lipschitz constant M: {quadratic_func.M}")
    print(f"Condition number ρ: {quadratic_func.rho}")
    print(f"Initial distance r0: {quadratic_func.r0}")

    # Example objective function and gradient evaluation
    x = np.random.rand(d)
    print(f"Objective function value at x: {quadratic_func.objective_function(x)}")
    print(f"Gradient at x: {quadratic_func.gradient(x)}")