import numpy as np

class LinearSystemsSolver:
    def __init__(self, matrix):
        self.A = matrix

    def decomposition(self):
        """
        Allow to find the Triangular Lower matrix used in the decompostion of a matrix:
        A = LL^(-1)
        :return: L
        """
        if np.sum(np.linalg.eigvals(self.A) < 0) > 0:
            raise ValueError(
                "Matrice has to be semi definite. Here is the eigenvalues found: "
                + str(np.linalg.eigvals(self.A))
            )

        L = np.zeros((len(self.A), len(self.A)))

        for j in range(len(self.A)):

            L[j, j] = np.sqrt(
                self.A[j, j] - np.sum([L[j, k] ** 2 for k in range(0, j)])
            )

            for i in range(j + 1, len(self.A)):

                L[i, j] = (
                    self.A[i, j] - np.sum([L[i, j - 1] * L[j, k] for k in range(0, j)])
                ) / L[j, j]

        return L

    def solver(self, vector):

        if len(vector) != len(self.A):
            raise ValueError(
                "Vector output does not have the same length as the matrix"
            )

        lower = self.decomposition()
        upper = lower.copy().T

        y, x = np.zeros(len(vector)), np.zeros(len(vector))

        for i in range(len(y)):

            y[i] = (vector[i] - sum([lower[i, j] * y[j] for j in range(i)])) / lower[i, i]

        for i in np.arange(len(x) - 1, -1, -1):

            x[i] = (y[i] - sum([upper[i, j] * x[j] for j in range(i + 1, len(x))])) / upper[i, i]

        return x
    
    def solver2(self, vector):
        if len(vector) != len(self.A):
            raise ValueError(
                "Vector output does not have the same length as the matrix"
            )

        lower = self.decomposition()
        upper = lower.copy().T
        y = np.linalg.inv(lower) @ b
        x = np.linalg.inv(upper) @ y
        
        return x

if __name__ == '__main__':
    A = np.array([1,2,0,0,2,6,-2,0,0,-2,5,-2,0,0,-2,3]).reshape(4, 4)
    b = np.array([3,8,9,3])
    #res = np.linalg.cholesky(A)
    met = LinearSystemsSolver(A)
    #ch = met.decomposition()
    x = met.solver(b)
    xx = met.solver2(b)
    print(np.linalg.solve(A, b))

