import numpy as np
import yfinance as yf
from scipy.optimize import minimize


class Garch_11:

    def __init__(self, residu):

        self.res = residu * 100

    def garch(self, params):

        omega = params[0]
        alpha = params[1]
        beta = params[2]

        sigma2 = [omega / (1-alpha-beta)]
        z = [np.power(self.res[0] / sigma2[0],2)]
        for t in range(1, len(self.res)):

            #sigma2.append(omega + alpha * np.power(self.res[t - 1], 2) + beta * sigma2[t - 1])
            sigma2.append(omega + alpha * sigma2[t - 1] * z[t - 1] + beta * sigma2[t - 1])
            z.append(np.power(self.res[t]/sigma2[t],2))

        return sigma2

    def loglikelihood(self, params):
        sigma2 = self.garch(params)

        return -np.sum(-np.log(sigma2) - (self.res ** 2) / sigma2)


    def garch_estimation(self, params):
        cons = ({'type': 'ineq', 'fun': lambda x: -x[1] - x[2] + 1})
        return minimize(self.loglikelihood, params, constraints = cons, bounds= ((0.001, None), (0,0.999), (0,0.999))).x

ibm = yf.download('BTC', start = '2018-01-01', interval = '1d').Close.to_numpy()

st = np.array(np.log(ibm) - np.log(ibm).mean())

serie = np.diff(st)
est = Garch_11(serie)

esti = est.garch_estimation([0.1, 0.1, 0.75])









