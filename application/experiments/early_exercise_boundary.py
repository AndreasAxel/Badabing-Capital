from application.binomial_model.binomial_model import binomial_tree
from application.options.payoff import european_payoff
from application.simulation.sim_gbm import GBM
from application.Longstaff_Schwartz.LSMC import LSMC
from application.utils.LSMC_fit_predict import *


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    M_binom = 2500 # for Binomial
    M_LSMC = 50 # for LSMC
    N = 100000 # stocks in LSMC
    t0 = 0.0
    T = 0.25
    r = 0.06
    sigma = 0.2
    S0 = 40
    K = 40
    option_type = 'PUT'
    eur_amr = 'AMR'
    use_av = True
    deg = 9
    seed = 1234

    # get binomial model boundary
    u = np.exp(sigma * np.sqrt(T / M_binom))
    d = np.exp(-sigma * np.sqrt(T / M_binom))

    price, delta, eeb = binomial_tree(K, T, S0, r, M_binom, u, d, european_payoff, option_type=option_type, eur_amr=eur_amr)


    # get LSMC boundary
    t = np.linspace(start=t0, stop=T, num=M_LSMC + 1, endpoint=True)
    x0 = np.linspace(20, 60, N, endpoint=True)
    simulator = GBM(t=t, x0=x0, N=N, mu=r, sigma=sigma, use_av=use_av, seed=seed)
    simulator.sim_exact()
    LSMC_gbm = LSMC(simulator, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
    LSMC_gbm.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg)

    # plot boundaries
    plt.title('Early Exercise Boundary')
    plt.plot(np.linspace(0, T, M_binom+1, True), eeb)
    plt.hlines(K, xmin=0.0, xmax=T, colors='black', linestyles='dashed')
    plt.plot(t, LSMC_gbm.early_exercise_boundary)
    plt.show()
