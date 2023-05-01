import numpy as np
import pandas as pd
from LSMC import lsmc
from application.options.payoff import european_payoff
from application.simulation.sim_gbm import sim_gbm
from application.Longstaff_Schwartz.utils.fit_predict import *
import matplotlib.pyplot as plt


def bumpNreval(params: 'dict', deg=5, maxSimul=10000, minSimul=1000, stepSimul=19, bumpFactor=1.01):
    """
    :param params:      dictionary of params: x0, K, r, sigma, t0, T, M, seed, type
    :param deg:         no. of basis functions in Longstaff-Schwartz regression step
    :param maxSimul:    max no. of GBM paths in Longstaff-Schwartz
    :param minSimul:    min no. of GBM paths in Longstaff-Schwartz
    :param stepSimul:   no. points to evaluate
    :param bumpFactor:  bump underlying by this factor

    :return:            grid w/ columns: #simul, base price, bump price, bump factor, delta
    """

    # range of simulations
    sims = np.linspace(minSimul, maxSimul, stepSimul).astype('int')

    grid = np.array([(n, np.nan, np.nan, np.nan, np.nan) for n in sims])

    x0 = params['x0']
    K = params['K']
    r = params['r']
    t0 = params['t0']
    T = params['T']
    M = params['M']
    sigma = params['sigma']
    seed = params['seed']
    type = params['type']

    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)

    # simulating base and bumped underlyings (GBM)
    i = 0
    for N in sims:
        baseX = sim_gbm(t=t, x0=x0, N=N, mu=r, sigma=sigma, seed=seed)
        bumpX = sim_gbm(t=t, x0=x0 * bumpFactor, N=N, mu=r, sigma=sigma, seed=seed)

        base = lsmc(t=t, X=baseX, K=K, r=r, payoff_func=european_payoff,
                    type=type, fit_func=fit_laguerre_poly,
                    pred_func=pred_laguerre_poly,
                    deg=deg)
        bump = lsmc(t=t, X=bumpX, K=K, r=r, payoff_func=european_payoff,
                    type=type, fit_func=fit_laguerre_poly,
                    pred_func=pred_laguerre_poly,
                    deg=deg)
        delta = (base - bump) / (x0 * (bumpFactor - 1))

        # vec (base, bumped, bumpFactor)
        grid[i, 1:] = (base, bump, bumpFactor, delta)
        i += 1

    return grid


if __name__ == '__main__':
    reps = 100

    params = {
        'x0': 36,
        'K': 40,
        'r': 0.06,
        't0': 0.0,
        'T': 1.0,
        'M': 50,
        'sigma': 0.2,
        'seed': None,
        'type': 'PUT'
    }

    stddevMatrix = bumpNreval(params=params)
    stddevMatrix = stddevMatrix[:, [0, 4]]

    for r in range(reps):
        print(r)
        next = bumpNreval(params=params)
        stddevMatrix = np.c_[stddevMatrix, next[:, -1]]

    print(stddevMatrix)

    stddevs = np.std(stddevMatrix[:, 1:], axis=1)

    #plotting
    x = stddevMatrix[:, 0]
    y = stddevs
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    abline_values = [slope * i + intercept for i in np.log(x)]
    print('loglog slope', slope)

    plt.figure()
    plt.title('LSMC ∆ stability (bump-n-reval)')
    plt.plot(x, y, 'o', color='blue')
    plt.plot(stddevMatrix[:, 0], stddevs,
             linestyle='-', color='blue')
    plt.ylabel('Standard deviation of ∆')
    plt.xlabel('# simulations')
    plt.text(1000, min(stddevs), r'K={}, r={}, sigma={}, #rep={}'.format(params['K'], params['r'],
                                                               params['sigma'], reps))
    plt.show()

    plt.figure()
    plt.title('LSMC ∆ stability (bump-n-reval)')
    plt.plot(np.log(x), np.log(y), 'o',
              color='blue')
    plt.plot(np.log(x), abline_values, '-', color='grey')
    plt.plot(np.log(x), np.log(y),
             linestyle='-', color='blue')
    plt.ylabel('Standard deviation of ∆')
    plt.xlabel('# simulations')
    plt.text(np.log(1000), np.log(min(stddevs)), r'K={}, r={}, sigma={}, #rep={}, slope={}'.format(params['K'], params['r'],
                                                               params['sigma'], reps, np.round(slope,3)))
    plt.show()

