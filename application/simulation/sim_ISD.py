import numpy as np
from sklearn.neighbors import KernelDensity


def sim_ISD(N, x0:float, alpha, seed=None, *args, **kwargs):
    """
    Creates N initially state dispersed variables (ISD).
    With reference to Latourneau & Stentoft (2023)

    :param N:       no. paths
    :param x0:      init. value
    :param alpha:   band width dispersion
    :param seed:    seed
    :param args:
    :param kwargs:
    :return:        size N vector of state dispersed vars around x0
    """

    # epanechnikov kernel
    vecUnif = np.random.default_rng(seed=seed).uniform(low=0, high=1, size=N)
    kernel = 2 * np.sin( np.arcsin(2 * vecUnif -1) / 3)

    # Initial state dispersion
    X = float(x0) + alpha * kernel
    return X


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from application.simulation.sim_gbm import sim_gbm
    from application.Longstaff_Schwartz.LSMC import LSMC
    from application.Longstaff_Schwartz.utils.fit_predict import fit_poly, pred_poly
    from application.options.payoff import european_payoff
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
    x_isd = sim_ISD(N=N, x0=x0, alpha=alpha)
    X = sim_gbm(t=t, x0=x_isd, N=N, mu=r, sigma=sigma, seed=seed)

    # LSMC
    lsmc = LSMC(X=X, t=t, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
    lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

    # Get cashflow at stopping-time
    cf = lsmc.payoff[1:]
    cf_tau = np.zeros(N)
    cf_tau[~np.isnan(lsmc.pathwise_opt_stopping_idx)] = cf[lsmc.opt_stopping_rule]

    # Calculate discount factor
    df = [np.exp(-r*tau) if ~np.isnan(tau) else 0.0 for tau in lsmc.pathwise_opt_stopping_time]

    # Calculate discounted cashflow (present value)
    cf_pv = cf_tau * df

    coef_price = fit_poly(x=x_isd - x0, y=cf_pv, deg=deg_stentoft)  # coefficients `b`
    price = pred_poly(x=x0 - x0, fit=coef_price)

    coef_delta = np.polyder(coef_price, 1)
    delta = pred_poly(x=x0 - x0, fit=coef_delta)


