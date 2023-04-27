import numpy as np
import matplotlib.pyplot as plt
from application.simulation.sim_gbm import sim_gbm
from application.Longstaff_Schwartz.LSMC import lsmc
from application.Longstaff_Schwartz.utils.fit_predict import *
from application.options.payoff import european_payoff
import warnings
warnings.filterwarnings("ignore")


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

    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)

    for i in range(repeat):
        N_sim = [base ** n for n in range(2, 10)]
        prices = []
        for N in N_sim:
            X = sim_gbm(t=t, x0=x0, N=N, mu=r, sigma=sigma, use_av=use_av, seed=seed)
            p = lsmc(t=t, X=X, K=K, r=r, payoff_func=european_payoff,
                     type=type, fit_func=fit_laguerre_poly,
                     pred_func=pred_laguerre_poly,
                     deg=deg)
            prices.append(p)

        plt.semilogx(N_sim, prices, base=base, color='black')
    plt.show()
