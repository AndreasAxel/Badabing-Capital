import numpy as np
import matplotlib.pyplot as plt
from application.binomial_model.binomial_model import binomial_tree_bs
from application.options.payoff import european_payoff


if __name__ == '__main__':
    K = 40
    T = 1.0
    S0 = 40
    r = 0.06
    sigma = 0.2

    M_list = [2**i for i in range(4, 14)]

    price = []
    delta = []

    for M in M_list:
        p, d, _ = binomial_tree_bs(K, T, S0, r, sigma, M, european_payoff, option_type='PUT', eur_amr='EUR')
        price.append(p)
        delta.append(d)

    price = np.array(price)
    delta = np.array(delta)

    fig, ax = plt.subplots(2, sharex='all')
    ax[0].scatter(M_list, np.abs(price - price[-1]), color='black')
    ax[0].set_ylabel('Price')
    ax[0].loglog(base=2)
    ax[1].scatter(M_list, np.abs(delta - delta[-1]), color='blue')
    ax[1].set_ylabel('Delta')
    ax[1].set_xlabel('Number of steps (M)')
    ax[1].loglog(base=2)
    plt.suptitle('Convergence - Binomial Option Price Model - Absolute Error')
    plt.show()
