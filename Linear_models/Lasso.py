import numpy as np
from LinearRegression import LinearRegression


class Lasso(LinearRegression):
    """
    This class inherit from the linear regression class. Only the derivative of the cost function change.
    """
    def __init__(self, learning_rate=0.01, n_iter=2000, _lambda=0.1, optimizer='GradientDescent'):
        super().__init__(
            learning_rate=learning_rate,
            n_iter=n_iter,
            optimizer=optimizer
        )
        self._lambda = _lambda

    def _find_derivatives(self, X_mat, err):
        n_row = X_mat.shape[0]
        if self.optimizer == 'SGD':
            idx = np.random.randint(low=0, high=n_row)
            n_row = 1
        elif self.optimizer == 'MBSGD':
            idx = np.random.randint(low=0, high=n_row, size=self.size_mbsgd)
            n_row = self.size_mbsgd
        else:
            idx = range(n_row)

        self.db = 2 * np.sum(err[idx]) / n_row
        self.dw = (2 * np.dot(X_mat[idx, 1:].T, err[idx]) + self._lambda * np.sign(self.weight[1:])) / n_row

    @staticmethod
    def minmax(X):
        """
        Scaling value is important in shrinkage methods. Not shrinking can lead to have the same results as
        a linear regression
        """
        # return (X - np.mean(X)) / np.sqrt(np.var(X))
        return (X - X.min()) / (X.max() - X.min())


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # evaluation
    def mse(y_true, predictions):
        return np.mean((y_true - predictions)**2)

    X, y = datasets.make_regression(n_samples=10000, n_features=3, noise=30, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Test if _lambda = 0
    # beta_lin = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    # lasso = Lasso(n_iter=10000, learning_rate=0.01, _lambda=0)
    # lasso.fit(X_train, y_train)

    # Transform the data
    X_train = Lasso.minmax(X_train)
    y_train = Lasso.minmax(y_train)

    lasso = Lasso(n_iter=10000, learning_rate=0.01, _lambda=1)
    lasso.fit(X_train, y_train)
    pred = lasso.predict(X_test)
    mse_lasso = mse(y_test, pred)
