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
    :return:                Option price and delta (both at time 0), and early exercise boundary
    """
    # Auxiliary variables
    dt = T / M
    a = np.exp(r * dt)
    q = (a - d) / (u - d)
    df = np.exp(-r * dt)

    # Initialise stock prices and option payoff at expiry; delta and early exercise boundary
    S = S0 * d ** (np.arange(M, -1, -1)) * u ** (np.arange(0, M + 1, 1))
    V = payoff_func(S, K, option_type=option_type)
    delta = np.nan
    B = np.full(shape=(M+1,), fill_value=np.nan)
    B[M] = K

    # Backward recursion through the tree
    for i in np.arange(M - 1, -1, -1):
        S = S0 * d ** np.arange(i, -1, -1) * u ** np.arange(0, i + 1, 1)
        V[:i + 1] = df * (q * V[1:i + 2] + (1 - q) * V[0:i + 1])
        V = V[:-1]
        if eur_amr == 'AMR':
            payoff = payoff_func(S, K, option_type=option_type)
            ex = V < payoff
            if np.sum(ex) > 0:
                B[i] = np.max(S[ex])
            V = np.maximum(V, payoff)

        if i == 1:
            delta = (V[0] - V[1]) / (S[0] - S[1])

    return V[0], delta, B


def binomial_tree_bs(K, T, S0, r, sigma, M, payoff_func, option_type='PUT', eur_amr='EUR'):
    u = np.exp(sigma * np.sqrt(T / M))
    d = 1/u
    return binomial_tree(K, T, S0, r, M, u, d, payoff_func, option_type, eur_amr)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    M = 5000
    T = 1.0
    r = 0.06
    sigma = 0.2
    S0 = 40
    K = 40
    option_type = 'PUT'
    eur_amr = 'AMR'

    u = np.exp(sigma * np.sqrt(T / M))
    d = np.exp(-sigma * np.sqrt(T / M))

    price, delta, eeb = binomial_tree(K, T, S0, r, M, u, d, european_payoff, option_type=option_type, eur_amr=eur_amr)

    print(price, delta)

    plt.plot(np.linspace(0, T, M+1, True), eeb)
    plt.title('Early Exercise Boundary')
    plt.show()


