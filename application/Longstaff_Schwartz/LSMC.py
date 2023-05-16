import numpy as np
from application.options.payoff import european_payoff
from application.simulation.sim_gbm import GBM
from application.Longstaff_Schwartz.utils.fit_predict import *


class LSMC():
    def __init__(self, simulator, K, r, payoff_func, option_type):
        """
        Longstaff-Schwartz Monte Carlo method for pricing an American option.

        :param t:               Time steps
        :param X:               Simulated paths
        :param K:               Strike
        :param r:               Risk free rate
        :param payoff_func:     Payoff function to be called
        :param type:            Type of option
        :param fit_func         Function for fitting the (expected) value of value of continuation
        :param pred_func        Function for predicting the (expected) value of continuation
        """
        self.t = simulator.t
        self.dt = simulator.dt
        self.M = simulator.M
        self.N = simulator.N
        self.X = simulator.X
        self.W = simulator.W

        self.sigma = simulator.sigma
        self.K = K
        self.r = r

        self.payoff_func = payoff_func
        self.option_type = option_type

        self.df = np.exp(-r * self.dt)

        self.fit_func = None
        self.pred_func = None

        # Initialization
        self.payoff = payoff_func(self.X, self.K, option_type)
        self.cashflow = np.zeros_like(self.payoff)
        self.cashflow[self.M, ] = self.payoff[self.M, :]
        self.stopping_rule = np.zeros_like(self.X).astype(dtype=bool)
        self.stopping_rule[self.M, :] = self.payoff[self.M, :] > 0

        # LSMC results (price and optimal stopping)
        self.price = None
        self.opt_stopping_rule = None               # Matrix from Longstaff-Schwartz's paper with first optimal stopping
        self.pathwise_opt_stopping_idx = None       # Vector with index of the optimal stopping time for each path
        self.pathwise_opt_stopping_time = None      # Vector with the optimal stopping time for each path

        # Adjoint Differential Greeks (assumes Black-Scholes model)
        self.bs_price_ad = None
        self.bs_delta_ad = None
        self.bs_vega_ad = None

    def run_backwards(self, fit_func, pred_func, regress_only_itm=True, *args, **kwargs):
        """
        Iterates backwards in time using the LSMC algorithm.

        :param fit_func         Function for fitting the (expected) value of value of continuation
        :param pred_func        Function for predicting the (expected) value of continuation
        :param regress_only_itm: Use only In-The-Money paths in fit- and predict-statement
                                 According to Longstaff & Schwart's article this improves efficiency and computation time.
        :param args:            Arguments passed to fit and predict function
        :param kwargs:          Keyword arguments passed to fit and predict function
        """
        self.fit_func = fit_func
        self.pred_func = pred_func

        for j in range(self.M - 1, 0, -1):
            itm = self.payoff[j, :] > 0 if regress_only_itm else np.ones_like(self.payoff[j, :]).astype(dtype=bool)

            # Fit prediction model
            fit = fit_func(self.X[j, itm], self.cashflow[j + 1, itm] * self.df[j], *args, **kwargs)

            # Predict value of continuation
            pred = list(pred_func(self.X[j, itm], fit))
            EV_cont = np.zeros((self.N,))
            EV_cont[itm] = pred

            # Determine whether is it optimal to exercise or continue. Update values accordingly
            self.stopping_rule[j, :] = self.payoff[j, :] > EV_cont
            self.cashflow[j, :] = np.where(self.stopping_rule[j, :],
                                           self.payoff[j, :],                     # Value of exercising
                                           self.cashflow[j + 1, :] * self.df[j])  # Value of continuation

        # Calculate outputs
        self.price = np.mean(self.cashflow[1, :] * self.df[0])
        self.opt_stopping_rule = np.cumsum(np.cumsum(self.stopping_rule[1:], axis=0), axis=0) == 1

        self.pathwise_opt_stopping_idx = np.argmax(self.stopping_rule, axis=0).astype(float)
        self.pathwise_opt_stopping_idx[self.pathwise_opt_stopping_idx == 0] = np.nan  # Not exercising at time 0

        self.pathwise_opt_stopping_time = [idx if np.isnan(idx)
                                           else self.t[int(idx)]
                                           for idx in self.pathwise_opt_stopping_idx]

    def pathwise_bs_greeks_ad(self):
        """
        Calculate delta by algorithm 2 in
            'Fast Estimates of Greeks from American Options - A Case Study in Adjoint Algorithm Differentiation'
            by Deussen, Mosenkis, Naumann (2018).
        And slides
            'Fast Delta-Estimates for American Options by Adjoint Algorithmic Differentiation'
            by Deussen (Autodiff, 2015)
        """
        bs_price_ad = 0.0
        bs_delta_ad = 0.0
        bs_vega_ad = 0.0

        # Calculate "contribution" from each path
        for p in range(self.N):
            idx = self.pathwise_opt_stopping_idx[p]
            if np.isnan(idx):
                continue
            idx = int(idx)
            tau = self.t[idx]
            X_tau = self.X[idx, p]
            W_tau = self.W[idx, p]

            bs_price_ad += self.payoff_func(X_tau, self.K, self.option_type) * np.exp(-self.r * tau)
            # Calculate greeks (assume Black Scholes model)
            # TODO: This can be generalized to ANY model by using Adjoint Differentiation (AD). Also more greeks can be added
            #bs_delta_ad += - X_tau * np.exp(-self.r * t[idx]) / self.X[0, p]
            bs_delta_ad += -np.exp((self.r - 0.5 * self.sigma**2) * tau + self.sigma * W_tau)
            bs_vega_ad += -X_tau * (-self.sigma * tau)

        # Results are the average across each path
        self.bs_price_ad = bs_price_ad / self.N
        self.bs_delta_ad = bs_delta_ad / self.N
        self.bs_vega_ad = bs_vega_ad / self.N


if __name__ == '__main__':
    """
    # Example from the Longstaff-Schwartz article
    K = 1.1
    r = 0.06
    t = np.linspace(start=0, stop=3, num=4)
    option_type = 'PUT'
    deg = 2

    X = np.array((
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        [1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88],
        [1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22],
        [1.34, 1.54, 1.03, 0.92, 1.52, 0.90, 1.01, 1.34]
    ))

    LSMC_example = LSMC(t=t, X=X, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
    LSMC_example.run_backwards(fit_func=fit_poly, pred_func=pred_poly, regress_only_itm=True, deg=deg)

    print("Price Longstaff-Schwarz: ", LSMC_example.price)
    print("Pathwise optimal stopping time: \n", LSMC_example.pathwise_opt_stopping_time)
    print("Pathwise optimal stopping index: \n", LSMC_example.pathwise_opt_stopping_idx)
    print(LSMC_example.opt_stopping_rule)
    """

    # Simulating with GBM
    x0 = 40
    K = 40
    r = 0.06
    t0 = 0.0
    T = 1.0
    N = 100000
    M = 50
    sigma = 0.2
    seed = 1234
    use_av = True
    option_type = 'PUT'

    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)
    simulator = GBM(t=t, x0=x0, N=N, mu=r, sigma=sigma, use_av=use_av, seed=seed)
    simulator.sim_exact()

    for deg in range(5, 6):
        LSMC_gbm = LSMC(simulator, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
        LSMC_gbm.run_backwards(fit_func=fit_poly, pred_func=pred_poly, regress_only_itm=True, deg=deg)
        LSMC_gbm.pathwise_bs_greeks_ad()

        print('deg = {}: Price = {:.4f}, Delta = {:.4f}, Vega = {:.4f}'.format(
            deg, LSMC_gbm.price, LSMC_gbm.bs_delta_ad, LSMC_gbm.bs_vega_ad))
