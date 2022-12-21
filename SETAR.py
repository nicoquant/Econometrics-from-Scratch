import math
import numpy as np
from numpy.linalg import inv
from statsmodels.stats.diagnostic import acorr_ljungbox as lbtest
from statsmodels.stats.diagnostic import het_arch as archtest
from statsmodels.stats.stattools import jarque_bera as jbtest


class Setar:
    def __init__(self, params):
        self.params = params
        self.squ_err_opt = (
            np.inf
        )  # used as initial value to compute the optimal sum of squared resiudals
        self.beta_opt = None
        self.err = None
        self.X_opt = None
        self.seuil_opt = None
        self.pvals = None
        self.lb_test = None
        self.arch_test = None
        self.jb_test = None

    def process_simulation(self, lamb, T):
        """
        Simuation of a setar(1,1) process
        """
        Y = np.zeros((T, 1))
        e = np.random.normal(0, 1, T)
        for t in range(1, T):
            if Y[t - 1] <= lamb:
                Y[t] = self.params[0] + self.params[1] * Y[t - 1] + e[t]
            else:
                Y[t] = self.params[2] + self.params[3] * Y[t - 1] + e[t]
        return Y

    @staticmethod
    def beta(X, y):
        """
        OLS estimation to estimate parameters
        """
        return inv(X.T @ X) @ X.T @ y

    def fit(self, Y):
        """
        Fitting the data:
            1) select 70% of the data
            2) loop on every possible lambda to find the one which minimize the squared errors
        """
        T = len(Y)
        y = Y[: T - 1]
        x = Y[1:T]
        seuil_range = sorted(y)[int(0.15 * (T - 1)): int(0.85 * (T - 1))]
        seuil_range = np.concatenate(seuil_range, axis=0).reshape(len(seuil_range), 1)

        # array for linearity test
        self.wald_array_dist = np.zeros((len(seuil_range), 1))
        R = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
        r = np.zeros((2, 1))
        for i in range(0, len(seuil_range)):

            seuil = seuil_range[i]
            indicatrice = x <= seuil

            X = (
                np.array(
                    [
                        indicatrice,
                        indicatrice * x,
                        1 - indicatrice,
                        (1 - indicatrice) * x,
                    ]
                )
                .reshape(4, T - 1)
                .T
            )
            beta = self.beta(X, y)
            error = y - X @ beta
            squ_err = error.T @ error / (T - 1)


            if squ_err < self.squ_err_opt:

                self.beta_opt = beta
                self.err = error
                self.squ_err_opt = squ_err
                self.X_opt = X
                self.seuil_opt = seuil
                self.wald = inv(squ_err) @ (R @ beta - r).T @ (R @ inv(X.T @ X) @ R.T) @ (R @ beta - r)


    def specification_test(self):
        '''
        Compute t-stats to observe the significance of the parameters
        Compute Liyung Box, arch and jb test on residuals
        '''
        # Significance of params

        var_beta = self.squ_err_opt * inv(self.X_opt.T @ self.X_opt)
        sigbeta = np.sqrt(np.diag(var_beta)).reshape(4, 1)
        t_stats = self.beta_opt / sigbeta
        self.pvals = [2 * (1 - math.erf(abs(tstat))) for tstat in t_stats]
        # pval of a coeff = 2(1 - cdf(x))
        # if pval < 0.05 ==> Significative

        self.lb_test = lbtest(self.err, 5)
        self.arch_test = archtest(self.err ** 2, 5)
        self.jb_test = jbtest(self.err)

        # self.linearity_test_ = self.linearity_test()

    def linearity_test(self, T=1000):

        one = np.ones((T - 1, 1))
        Xline = np.array([[one], [X]])
        bet_lin = inv(Xline.T @ Xline) @ Xline.T * Y
        e_lin = Y - Xline @ bet_lin
        sigma_lin = np.sqrt(e_lin.T @ e_lin / len(e_lin))
        nb_rep = 1000
        Wsim = np.zeros((nb_rep, 1))

        for i in range(0, nb_rep):
            xsim = np.zeros((T, 1))

            for t in range(1, T):
                xsim[t] = bet_lin[0] + bet_lin[1] * xsim[t - 1] + sigma_lin * np.random.normal(0, 1, 1)
            Wsim[i] = self.wald_setar_test_lin(xsim)

        VC = np.quantile(Wsim, 0.95)
        tstat = bet_lin / sigma_lin
        pval = np.sum(self.wald < Wsim) / nb_rep

        return tstat, VC, pval

    def wald_setar_test_lin(self,x):
        pass


parameters = (-0.5, 0.5, 0.5, 0.7)
mod = Setar(parameters)
true_y = mod.process_simulation(0, 500)
mod.fit(true_y)
mod.specification_test()
