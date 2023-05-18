import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from application.binomial_model.binomial_model import binomial_tree_bs
from application.options.payoff import european_payoff


if __name__ == '__main__':
    spot = np.linspace(start=10, stop=70, num=101, endpoint=True)
    K = 40
    r = 0.06
    sigma = 0.2
    T = 1.0
    M = 5000
    option_type = 'PUT'
    eur_amr = 'AMR'

    price = []
    delta = []

    for s in tqdm(spot):
        res = binomial_tree_bs(K=K, T=T, S0=s, r=r, sigma=sigma, M=M, payoff_func=european_payoff,
                               option_type=option_type, eur_amr=eur_amr)
        price.append(res[0])
        delta.append(res[1])

    fig, ax = plt.subplots(2, sharex='all')
    plt.title('Binomial model for {} {} option'.format(eur_amr, option_type))
    ax[0].plot(spot, price, color='blue')
    ax[0].set_ylabel('Price')

    ax[1].plot(spot, delta, color='red')
    ax[1].set_ylabel('Delta')
    ax[1].set_xlabel('Spot')

    plt.show()
