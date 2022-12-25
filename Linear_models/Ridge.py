import numpy as np
from LassoRegression import LinearRegression


class Ridge(LinearRegression):
    """
    This class inherit from the linear regression class. Only the derivative of the cost function change.
    """
    def __init__(self, learning_rate=0.01, n_iter=2000, _lambda=0.1):
        super().__init__(
            learning_rate=learning_rate,
            n_iter=n_iter
        )
        self._lambda = _lambda

    def _find_derivatives(self, X, err):
        n_row = X.shape[0]
        db = 2 * np.sum(err) / n_row
        dw = 2 * (np.dot(X[:, 1:].T, err) + self._lambda * self.weight[1:]) / n_row
        return dw, db

    @staticmethod
    def minmax(X):
        """
        Scaling value is essensial in shrinkage methods. Not shrinking can lead to have the same results as
        a linear regression
        """
        return (X - np.mean(X)) / np.sqrt(np.var(X))


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # evaluation
    def mse(y_true, predictions):
        return np.mean((y_true - predictions)**2)

    X, y = datasets.make_regression(n_samples=1000, n_features=2, noise=30, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Test if _lambda = 0
    # beta_lin = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    # ridge = Ridge(n_iter=10000, learning_rate=0.01, _lambda=0)
    # ridge.fit(X_train, y_train)

    # Transform the data
    X_train = Ridge.minmax(X_train)
    y_train = Ridge.minmax(y_train)

    ridge = Ridge(n_iter=10000, learning_rate=0.01, _lambda=100)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_test)
    mse_ridge = mse(y_test, pred)

    # Straight Forward method:
    # ones = np.ones((X_train.shape[0], 1))
    # X_train = np.concatenate([ones, X_train], axis=1)
    # np.linalg.inv(X_train.T @ X_train + 100 * np.identity(X_train.shape[1])) @ X_train.T @ y_train