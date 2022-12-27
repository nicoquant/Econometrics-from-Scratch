import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iter=2000, optimizer='GradientDescent', size_mbsgd=128):
        self.weight = None
        self.learning_rate = learning_rate
        self.n_iters = n_iter
        self.optimizer = optimizer
        self.dw = None
        self.db = None
        self.size_mbsgd = size_mbsgd
        if self.optimizer in ['Momentum', 'ADAM']:
            self.Vw = 0
            self.Vb = 0
        if self.optimizer in ['RMSprop', 'ADAM']:
            self.Sw = 0
            self.Sb = 0

    def fit(self, X, y):
        """
        Alternative to b = (X' X)**(-1) (X'y)

        Using an optimisation algo we update the weights at each iteration
        """
        n_row, n_features = X.shape
        self.weight = np.zeros(n_features+1)
        ones_col = np.ones((n_row, 1))
        X = np.concatenate([ones_col, X], axis=1)

        for _ in range(self.n_iters):
            y_pred = X @ self.weight
            # error
            err = y_pred - y
            # derivatives of cost function
            self._find_derivatives(X, err)
            # update weights
            self.update_weight()
            print(self.weight)

    def _find_derivatives(self, X, err):
        """
        Depending on the optimisation technique, we might need to select randomly a sample or a set of sample
        """
        n_row = X.shape[0]
        if self.optimizer == 'SGD':
            idx = np.random.randint(low=0, high=n_row, size=1)
            n_row = 1
        elif self.optimizer == 'MBSGD':
            idx = np.random.randint(low=0, high=n_row, size=self.size_mbsgd)
            n_row = self.size_mbsgd
        else:
            idx = range(n_row)

        self.db = 2 * np.sum(err[idx]) / n_row
        self.dw = 2 * np.dot(X[idx, 1:].T, err[idx]) / n_row

    def predict(self, X):
        ones_col = np.ones((X.shape[0], 1))
        X = np.concatenate([ones_col, X], axis=1)
        return X @ self.weight

    def update_weight(self):
        """
        Depending on the optimisation technique, we update the weight differently
        """
        if self.optimizer == "Momentum":
            self.momentum_optimizer()
            self.weight[1:] = self.weight[1:] - self.learning_rate * self.Vw
            self.weight[0] = self.weight[0] - self.learning_rate * self.Vb
        elif self.optimizer == 'RMSprop':
            self.RMSprop_optimizer()
            self.weight[1:] = self.weight[1:] - self.learning_rate * (self.dw / (np.sqrt(self.Sw) + 10e-8))
            self.weight[0] = self.weight[0] - self.learning_rate * self.db / (np.sqrt(self.Sb) + 10e-8)
        elif self.optimizer == 'ADAM':
            self.adam_optimizer()
            self.weight[1:] = self.weight[1:] - self.learning_rate * (self.Vw / (np.sqrt(self.Sw + 10e-8)))
            self.weight[0] = self.weight[0] - self.learning_rate * self.Vb / (np.sqrt(self.Sb + 10e-8))
        else:
            self.weight[0] -= self.learning_rate * self.db
            self.weight[1:] -= self.learning_rate * self.dw

    def momentum_optimizer(self, beta=0.9):
        self.Vw = beta * self.Vw + (1 - beta) * self.dw
        self.Vb = beta * self.Vb + (1 - beta) * self.db

    def RMSprop_optimizer(self, beta=0.999):
        self.Sw = beta * self.Sw + (1 - beta) * self.dw**2
        self.Sb = beta * self.Sb + (1 - beta) * self.db**2

    def adam_optimizer(self, b1=0.9, b2=0.999):
        self.momentum_optimizer(beta=b1)
        self.RMSprop_optimizer(beta=b2)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # evaluation
    def mse(y_true, predictions):
        return np.mean((y_true - predictions)**2)

    X, y = datasets.make_regression(n_samples=10000, n_features=2, noise=30, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    import time

    start = time.time()
    lin = LinearRegression(n_iter=1000, learning_rate=0.02)
    lin.fit(X_train, y_train)
    print(lin.weight)
    pred = lin.predict(X_test)
    mse_lin = mse(y_test, pred)
    print(time.time() - start)

    start = time.time()
    lin = LinearRegression(n_iter=2500, learning_rate=0.02, optimizer='MBSGD')
    lin.fit(X_train, y_train)
    print(lin.weight)
    pred = lin.predict(X_test)
    mse_lin_3 = mse(y_test, pred)
    print(time.time() - start)

    print(np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train)
