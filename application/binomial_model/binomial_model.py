import numpy as np
from application.options.payoff import european_payoff


def binomial_tree(K, T, S0, r, M, u, d, payoff_func, option_type='PUT', eur_amr='EUR'):
    """
    Binomial model for vanilla European and American options.

    :param K:               Strike
    :param T:               Expiration time
    :param S0:              Spot
    :param r:               Risk free rate
    :param M:               Number of discretization steps
    :param u:               Factor for up move
    :param d:               Factor for down move
    :param payoff_func:     Payoff function to be called
    :param option_type:     'PUT' / 'CALL'
    :param eur_amr:         'EUR' / 'AMR'
    :return:                Option price and delta (both at time 0)
    """
    # Auxiliary variables
    dt = T / M
    a = np.exp(r * dt)
    q = (a - d) / (u - d)
    df = np.exp(-r * dt)

    # Initialise stock prices at expiry, and option payoff
    S = S0 * d ** (np.arange(M, -1, -1)) * u ** (np.arange(0, M + 1, 1))
    V = payoff_func(S, K, option_type=option_type)
    delta = np.nan

    # Backward recursion through the tree
    for i in np.arange(M - 1, -1, -1):
        S = S0 * d ** np.arange(i, -1, -1) * u ** np.arange(0, i + 1, 1)
        V[:i + 1] = df * (q * V[1:i + 2] + (1 - q) * V[0:i + 1])
        V = V[:-1]
        if eur_amr == 'AMR':
            V = np.maximum(V, payoff_func(S, K, option_type=option_type))

        if i == 1:
            delta = (V[0] - V[1]) / (S[0] - S[1])

    return V[0], delta


def binomial_tree_bs(K, T, S0, r, sigma, M, payoff_func, option_type='PUT', eur_amr='EUR'):
    u = np.exp(sigma * np.sqrt(T / M))
    d = np.exp(-sigma * np.sqrt(T / M))
    return binomial_tree(K, T, S0, r, M, u, d, payoff_func, option_type, eur_amr)


if __name__ == '__main__':
    M = 5000
    T = 1.0
    r = 0.06
    sigma = 0.2
    S0 = 40
    K = 40
    option_type = 'PUT'

    u = np.exp(sigma * np.sqrt(T / M))
    d = np.exp(-sigma * np.sqrt(T / M))

    print(binomial_tree(K, T, S0, r, M, u, d, european_payoff, option_type='PUT', eur_amr='AMR'))
