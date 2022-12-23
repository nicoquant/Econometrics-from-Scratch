import numpy as np


class LinearRegression:
    def __init__(self, learning_rate = 0.01, n_iter = 1000):
        self.weight = None
        self.learning_rate = learning_rate
        self.n_iters = n_iter

    def fit(self, X, y):
        """
        Alternative to b = (X' X)**(-1) (X'y)

        Using gradient descent methodology which consist in updating the weights by substrating at each iterations the
        derivative of the cost function w.r to them
        """
        n_row, n_features = X.shape
        self.weight = np.zeros(n_features+1)
        ones_col = np.ones((n_row,1))
        X = np.concatenate([ones_col, X], axis = 1)

        for _ in range(self.n_iters):
            y_pred = X @ self.weight
            # error
            err = y_pred - y
            # derivatives of cost function
            dw = 2 * np.dot(X[:,1:].T, err) / n_row
            db = 2 * np.sum(err) / n_row
            # update weights
            self.weight[0] -= self.learning_rate * db
            self.weight[1:] -= self.learning_rate * dw

    def predict(self, X):
        ones_col = np.ones((X.shape[0], 1))
        X = np.concatenate([ones_col, X], axis=1)
        return X @ self.weight

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    X, y = datasets.make_regression(n_samples = 100, n_features = 2, noise = 20, random_state = 4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    plt.scatter(X[:, 0], y)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    def mse(y_test, pred):
        return np.mean((y_test - pred)**2)

    mse_ = mse(y_test, pred)
    y_pred_line = reg.predict(X)
    plt.scatter(X[:,0], y_pred_line)

