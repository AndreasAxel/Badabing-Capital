import numpy as np
from application.Longstaff_Schwartz.LSMC import LSMC
from application.Longstaff_Schwartz.utils.fit_predict import fit_poly, pred_poly
from application.simulation.sim_gbm import GBM
from application.options.payoff import european_payoff


def ISD(N, x0, alpha, seed=None):
    """
    Creates N initially state dispersed variables (ISD).
    With reference to Latourneau & Stentoft (2023)

    :param N:       no. paths
    :param x0:      init. value
    :param alpha:   band width dispersion
    :param seed:    seed

    :return:        size N vector of state dispersed vars around x0
    """
    vecUnif = np.random.default_rng(seed=seed).uniform(low=0, high=1, size=N)

    # Epanechnikov kernel
    kernel = 2 * np.sin( np.arcsin(2 * vecUnif -1) / 3)

    # Initial state dispersion
    X = float(x0) + alpha * kernel
    return X


if __name__ == '__main__':
    t0 = 0.0
    T = 1.0
    x0 = 40.0
    N = 100000
    M = 50
    r = 0.06
    sigma = 0.2
    K = 40.0
    seed = None
    deg_lsmc = 9
    deg_stentoft = 9
    option_type = 'PUT'

    alpha = 5.0

    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)

    # Simulate paths (with Initial State Dispersion)
    x_isd = ISD(N=N, x0=x0, alpha=alpha)
    X = GBM(t=t, x0=x_isd, N=N, mu=r, sigma=sigma, seed=seed, use_av=True)
    X.sim_exact()

    # LSMC
    lsmc = LSMC(simulator=X, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
    lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

    # Get cashflow at stopping-time
    cf = lsmc.payoff[1:]
    cf_tau = np.zeros(N)
    cf_tau[~np.isnan(lsmc.pathwise_opt_stopping_idx)] = cf[lsmc.opt_stopping_rule]

    # Calculate discount factor
    df = [np.exp(-r*tau) if ~np.isnan(tau) else 0.0 for tau in lsmc.pathwise_opt_stopping_time]
    # or "total" discounting?
    # df = np.exp(-r*T)

    # Calculate discounted cashflow (present value)
    cf_pv = cf_tau * df

    # Compute price and greeks from estimates                           
    s0 = x0 # Spot value

    coef_price = fit_poly(x=x_isd - x0, y=cf_pv, deg=deg_stentoft)  # coefficients `b`
    price = pred_poly(x=s0 - x0, fit=coef_price)

    coef_delta = np.polyder(coef_price, 1)
    delta = pred_poly(x=s0 - x0, fit=coef_delta)

    coef_gamma = np.polyder(coef_price, 2)
    gamma = pred_poly(x=s0 - x0, fit=coef_gamma)

    print('price = {}, delta = {}, gamma = {}'.format(price, delta, gamma))