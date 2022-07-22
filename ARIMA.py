import yfinance as yf
from scipy.linalg import toeplitz, inv
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from scipy.ndimage.interpolation import shift
import sys
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize


ibm = yf.Ticker("IBM").history(period="5y").Close.to_numpy()[0:1000]
train = ibm[0 : int(len(ibm) / 2)]
test = ibm[int(len(ibm) / 2) :]


class ARMA:
    def __init__(self, serie, order):

        if len(order) == 2:
            self.ar_param = order[0]
            self.ma_param = order[-1]

        else:
            print(
                "Can you please enter a value for p and q ? No need to enter a value for integrating the process,"
                "you will do it by calling the function stationarity_check"
            )
            self.ar_param = int(input("p: "))
            self.ma_param = int(input("q: "))

        self.serie = serie
        self.mu = np.mean(serie)
        self.eta = np.random.normal(0, 1, size=len(serie))
        self.lag = sum(order)
        self.i = 0

    def correlation(self):
        gamma = np.correlate(self.serie - self.mu, self.serie - self.mu, mode="full")
        return gamma[gamma.size // 2 :]

    def AR_YW_estimate(self):
        """

        :return: Estimate of AR(p) model.
        Based on Chapter 3, Slides 15: Applied Time Series Analysis, Prof. Dr. Ralf Brüggemann, Konstanz
        """

        autocorr = self.correlation() / self.correlation()[0]

        # Taking the lag-th first values
        r_p = autocorr[: self.lag + 1]  # (lag+1)*(lag+1) matrix

        # Bulding the matrix using the lag-1th value: lag*lag matrix, with 1 as diag
        R_p = toeplitz(r_p[:-1])

        return inv(R_p).dot(
            r_p[1:]
        )  # r_p[1:]: since we do not take corr(Xt, Xt) in account

    def AR_least_squared_estimator(self):
        """

        :param lag: autoregressor
        :return: Estimate of AR(p) model.
        Based on Chapter 3, Slides 16: Applied Time Series Analysis, Prof. Dr. Ralf Brüggemann, Konstanz
        """

        # Building of the matrix which contains the lags of the serie and taking out the p-th first values
        X = np.array(
            [shift(self.serie, i, cval=np.NaN) for i in range(1, self.lag + 1)]
        ).T[self.lag :]

        # Take out the p-th first values
        r_p = self.serie[self.lag :]

        return inv(X.T @ X) @ X.T @ r_p

    def loglikelihood_ar(self, param):
        """
        As stated in many papers (à citer), we assume past errors = 0
        :param param: initial values for the minimization
        :return: log likelihood
        """

        param_ar = np.append([1], -param)

        lag_ar = len(param_ar)

        T = len(self.serie)

        X = np.array(
            [shift(self.serie, i, cval=np.NaN) for i in range(1, lag_ar + 1)]
        ).T[lag_ar:]
        eta2 = param_ar @ X.T

        return (
            T / 2 * np.log(np.var(self.serie))
            + T / 2 * np.log(2 * np.pi)
            + np.sum((np.array(eta2) ** 2) / (2 * np.var(self.serie)))
        )

    def ar_estimation(self, initial_guess=None):
        """

        :param initial_guess: initial_guess
        :return: parameters for the ma part
        """
        if initial_guess == None:
            initial_guess = np.array([0] * self.ar_param)
        else:
            try:
                assert len(initial_guess) == self.ar_param

            except AssertionError:
                print(
                    "Please fix your starting parameters. The optimization has been done using a set of 0 values as "
                    "starting parameters."
                )
                initial_guess = np.array([0] * self.ar_param)
        return minimize(self.loglikelihood_ar, initial_guess).x

    def stationarity_check(self, estimation_method="ls"):

        if estimation_method == "ls":
            method = self.AR_least_squared_estimator()
        elif estimation_method == "yw":
            method = self.AR_YW_estimate()
        else:
            method = input(
                "Please choose an estimation method between: 'ls' and 'yw' or 'exit' to exit: "
            )
            if estimation_method == "exit":
                sys.exit()
            return self.stationarity_check(method)

        coeff = [1] + list(method * (-1))
        z = np.roots(list(reversed(coeff)))

        if all(round(x, 2) > 1 for x in list(map(np.abs, z))):
            print("Stationarity check: Pass")

        else:
            print("Warning: Your stability condition is not satisfied.")
            self.integrated_process(estimation_method)

    def stationary_test_adf(self):

        if adfuller(self.serie)[1] > 0.05:
            print(
                "Warning: Your stability condition is not satisfied. Please, make your serie stationary"
            )
        else:
            print("Stationarity check: Pass")

    def integrated_process(self, estimation_method):

        if estimation_method == "ls":
            estimation_method = self.AR_least_squared_estimator()
        elif estimation_method == "yw":
            estimation_method = self.AR_YW_estimate()
        else:
            estimation_method = input(
                "Please choose an estimation method between: 'ls' and 'yw' or 'exit' to exit: "
            )
            if estimation_method == "exit":
                sys.exit()
            return self.integrated_process(estimation_method)

        self.i += self.i + 1
        print("The series has been integrated: I(" + str(self.i) + ")")
        self.serie = np.diff(self.serie)
        self.eta = np.diff(self.eta)
        return estimation_method, self.stationarity_check()

    def loglikelihood_ma(self, param):

        """
        As stated in many papers (à citer), we assume past errors = 0c
        :param param: initial values for the minimization
        :return: log likelihood
        """
        lag = len(param)

        T = len(self.serie)

        eta2 = [0] * lag
        for t in range(1, len(self.serie)):
            eta2.insert(
                0,
                self.serie[t]
                - np.mean(self.serie)
                - param @ np.array(eta2[: len(param)]),
            )

        return (
            T / 2 * np.log(np.var(self.serie))
            + 1 / (2 * np.var(self.serie)) * np.sum(np.array(eta2) ** 2)
            + np.mean(self.serie)
        )

    def ma_estimation(self, initial_guess=None):
        """

        :param initial_guess: initial_guess
        :return: parameters for the ma part
        """
        if initial_guess == None:
            initial_guess = np.array([0] * self.ma_param)
        else:
            try:
                assert len(initial_guess) == self.ma_param

            except AssertionError:
                print(
                    "Please fix your starting parameters. The optimization has been done using a set of 0 values as "
                    "starting parameters."
                )
                initial_guess = np.array([0] * self.ma_param)
        return minimize(self.loglikelihood_ma, initial_guess).x

    def loglikelihood_arma(self, param):
        """
        As stated in many papers (à citer), we assume past errors = 0
        :param param: initial values for the minimization
        :return: log likelihood
        """

        if self.ar_param is None:
            print("You have to define p:")
            sys.exit()

        param_ar = np.append([1], -param[: self.ar_param])
        param_ma = param[self.ar_param :]

        T = len(self.serie)
        mu = np.mean(self.serie)
        eta2 = [0] * len(param_ma)
        for t in range(len(param_ar), T):
            eta2.insert(
                0,
                param_ar @ np.flip(self.serie[t - len(param_ar) : t])
                - mu
                - param_ma @ np.array(eta2[: len(param_ma)]),
            )

        return (
            T / 2 * (np.log10(np.var(self.serie)))
            + np.log10(2 * np.pi)
            + np.sum((np.array(eta2) ** 2) / (2 * np.var(self.serie)))
        )

    def arma_estimation(self, initial_guess=None):
        """

        :param initial_guess: initial_guess
        :return: parameters for the ma part
        """
        if initial_guess == None:
            initial_guess = np.array([0] * (self.ar_param + self.ma_param))
        else:
            try:
                assert len(initial_guess) == self.ar_param + self.ma_param

            except AssertionError:
                print(
                    "Please fix your starting parameters. The optimization has been done using a set of 0 values as "
                    "starting parameters."
                )
                initial_guess = np.array([0] * (self.ar_param + self.ma_param))
        return minimize(self.loglikelihood_arma, initial_guess).x

    def get_serie(self):
        return self.serie


if __name__ == "__main__":

    est = ARMA(train, order=(1, 2))
    est.stationarity_check()
    serie = est.get_serie()

    yule_walker = est.AR_YW_estimate()
    least_squared = est.AR_least_squared_estimator()
    loglikelihood_est_ar = est.ar_estimation()

    ma_est = est.ma_estimation()
    arma_est = est.arma_estimation()

    model = ARIMA(np.diff(train), order=(1, 0, 2))
    model_fit = model.fit()

    print("")
    print("")
    print(
        "Loglikelihood obtained by this minimization: "
        + str(model.loglike([np.mean(serie)] + list(arma_est) + [np.var(serie)]))
    )
    print("Parameters obtained by this estimation: " + str(arma_est))

    print("")
    print("")

    print(
        "Loglikelihood obtained by statsmodel minimization: "
        + str(model.loglike(model_fit.params))
    )
    print(
        "Parameters obtained by statsmodel estimation: " + str(model_fit.params[1:-1])
    )
