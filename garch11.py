import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from arch import arch_model

class Garch_11:

    def __init__(self, residu):

        self.res = residu*100


    def garch(self, params):

        omega = params[0]#/100
        alpha = params[1]
        beta = params[2]

        sigma2 = [omega / (1-alpha-beta)]
        for t in range(1, len(self.res)):

            sigma2.append(omega + alpha * self.res[t - 1]**2 + beta * sigma2[t - 1])


        return sigma2

    def loglikelihood(self, params):
        sigma2 = self.garch(params)

        return -np.sum(-np.log(sigma2[1:]) - (self.res[1:] ** 2) / sigma2[1:])


    def garch_estimation(self, params):
        cons = ({'type': 'ineq', 'fun': lambda x: 1 - x[1] - x[2]})
        params[0] = params[0] / 100 # rescaling the omega
        return minimize(self.loglikelihood, params, constraints = cons, bounds= ((0.0, 1000), (0.0,1000), (0.0,1000))).x

ibm = yf.download('AAPL', start = '2018-01-01', interval = '1d').Close.to_numpy()

st = np.array(np.log(ibm) - np.log(ibm).mean())
serie = np.diff(st)

est = Garch_11(serie)
esti = est.garch_estimation([0.02, 0.1, 0.7])


# model de référence
am = arch_model(serie)
res = am.fit()
t = res.params






