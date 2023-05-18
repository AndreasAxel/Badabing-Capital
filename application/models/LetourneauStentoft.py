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


def disperseFit(t0, T, x0, x_isd, N, M, r, sigma,K, seed,
                deg_lsmc, deg_stentoft, option_type,

                alpha = 25, *args, **kwargs):

    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)
    X = GBM(t=t, x0=x_isd, N=N, mu=r, sigma=sigma, seed=seed, use_av=True)
    X.sim_exact()
    # LSMC method for cashflows & stopping times
    lsmc = LSMC(simulator=X, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
    lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

    cf = lsmc.payoff
    cf = np.sum((cf * lsmc.opt_stopping_rule), axis=0)
    # Calculate discount factor
    df = [np.exp(-r * tau) if ~np.isnan(tau) else 0.0 for tau in lsmc.pathwise_opt_stopping_time]
    cf_pv = cf * df

    coef_price = fit_poly(x=x_isd - x0, y=cf_pv, deg=deg_stentoft)  # coefficients `b`
    coef_delta = np.polyder(coef_price, 1)
    coef_gamma = np.polyder(coef_price, 2)

    return x0, coef_price, coef_delta, coef_gamma

def Letourneau( spot, x0, priceFit, deltaFit, gammaFit, *args, **kwargs):
    """
    :param alpha:
    :param N:
    :param deg_stentoft:
    :param args:
    :param kwargs:
    :return:
    """

    price = pred_poly(x=spot - x0, fit=priceFit)
    delta = pred_poly(x=spot - x0, fit=deltaFit)
    gamma = pred_poly(x=spot - x0, fit=gammaFit)

    return (price, delta, gamma)



if __name__ == '__main__':
    plot = True

    t0 = 0.0
    T = 1.0
    x0 = 40.0
    N = 100000
    M = 50
    r = 0.06
    sigma = 0.2
    K = 40.0
    seed = 1234
    deg_lsmc = 9
    deg_stentoft = 9
    option_type = 'PUT'
    alpha = 25

    x_isd = ISD(N=N, x0=x0, alpha=alpha, seed=seed)
    fitted = disperseFit(t0=t0,
                         T=T,
                         x0=x0,
                         N=N,
                         M=M,
                         r=r,
                         sigma=sigma,
                         K=K,
                         seed=seed,
                         deg_lsmc=deg_lsmc,
                         deg_stentoft=deg_stentoft,
                         option_type=option_type,
                         alpha=alpha,
                         x_isd=x_isd)

    import matplotlib.pyplot as plt
    plt.scatter(x=delta, y=binDelta, marker='o')
    plt.show()

    """
    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)

    ## Calc. American Option price with Letourneau & Stentoft
    # Simulate paths (with Initial State Dispersion)
    x_isd = ISD(N=N, x0=x0, alpha=alpha)
    X = GBM(t=t, x0=x_isd, N=N, mu=r, sigma=sigma, seed=seed, use_av=True)
    X.sim_exact()
    # LSMC method for cashflows & stopping times
    lsmc = LSMC(simulator=X, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
    lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

    cf = lsmc.payoff
    cf = np.sum((cf * lsmc.opt_stopping_rule), axis=0)
    # Calculate discount factor
    df = [np.exp(-r*tau) if ~np.isnan(tau) else 0.0 for tau in lsmc.pathwise_opt_stopping_time]
    cf_pv = cf * df


    # Compute price and greeks from estimates                           
    s0 = x0 # Spot value
    coef_price = fit_poly(x=x_isd - x0, y=cf_pv, deg=deg_stentoft)  # coefficients `b`
    price = pred_poly(x=s0 - x0, fit=coef_price)

    coef_delta = np.polyder(coef_price, 1)
    delta = pred_poly(x=s0 - x0, fit=coef_delta)

    coef_gamma = np.polyder(coef_price, 2)
    gamma = pred_poly(x=s0 - x0, fit=coef_gamma)

    print('price = {}, delta = {}, gamma = {}'.format(price, delta, gamma))
    """
    def forPlotting(alpha, N, M, x0, t0, r, K, sigma, seed, deg_lsmc):
        t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)
        x_isd = ISD(N=N, x0=x0, alpha=alpha)
        X = GBM(t=t, x0=x_isd, N=N, mu=r, sigma=sigma, seed=seed, use_av=True)
        X.sim_exact()
        lsmc = LSMC(simulator=X, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
        lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

        # get cash flows
        cf = lsmc.payoff
        # vector of stopped cash flows
        cf = np.sum((cf * lsmc.opt_stopping_rule), axis=0)
        # discounting pathwise w/ stopping time
        df = [np.exp(-r * tau) if ~np.isnan(tau) else 0.0 for tau in lsmc.pathwise_opt_stopping_time]
        cf = cf * df

        x_isd = x_isd[~np.isnan(lsmc.pathwise_opt_stopping_idx)]
        cf = cf[~np.isnan(lsmc.pathwise_opt_stopping_idx)]

        return x_isd, cf


    if plot:
        import matplotlib.pyplot as plt
        alpha1, alpha2 = 0.5, 25
        x,y = forPlotting(alpha1, N, M, x0, t0, r, K, sigma, seed, deg_lsmc)
        x2, y2 = forPlotting(alpha2, N, M, x0, t0, r, K, sigma, seed, deg_lsmc)

        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle('Payoff dispersion, N={}'.format(N))
        ax1.scatter(x,y, marker='o', alpha=0.5, s=2)
        ax1.set_xlabel('Discounted payoff for alpha = {}'.format(alpha1))
        ax2.scatter(x2, y2, marker='o', alpha=0.5, s=2)
        ax2.set_xlabel('Discounted payoff for alpha = {}'.format(alpha2))
        plt.show()