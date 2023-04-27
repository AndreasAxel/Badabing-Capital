import numpy as np
import matplotlib.pyplot as plt
from application.simulation.sim_gbm import sim_gbm
from application.Longstaff_Schwartz.LSMC import lsmc
from application.Longstaff_Schwartz.utils.fit_predict import *
from application.options.payoff import european_payoff


if __name__ == '__main__':
    x0 = 36
    K = 40
    r = 0.06
    t0 = 0.0
    T = 1.0
    M = 50
    sigma = 0.2
    seed = None
    deg = 3
    type = 'PUT'
    use_av = True
    base = 4

    repeat = 10
    N_sim = [base ** n for n in range(2, 10)]

    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)

    prices = np.zeros((repeat, len(N_sim)))

    for i in range(repeat):
        for j, N in enumerate(N_sim):
            X = sim_gbm(t=t, x0=x0, N=N, mu=r, sigma=sigma, use_av=use_av, seed=seed)
            p = lsmc(t=t, X=X, K=K, r=r, payoff_func=european_payoff,
                     type=type, fit_func=fit_laguerre_poly,
                     pred_func=pred_laguerre_poly,
                     deg=deg)
            prices[i, j] = p

    plt.semilogx(N_sim, np.std(prices, axis=0), base=base)
    plt.title('LSMC convergence in number of simulations (N)')
    plt.xlabel('Number of simulations (N)')
    plt.ylabel('Standard deviation of price')
    plt.show()
