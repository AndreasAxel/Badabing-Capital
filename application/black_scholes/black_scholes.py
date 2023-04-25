import numpy as np
from scipy.stats import norm
from application.options.payoff import european_payoff


def bs_d(S, K, r, sigma, t, T):
    if t == T:
        d1 = np.nan
        d2 = np.nan
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
        d2 = d1 - (sigma * np.sqrt(T - t))

    return d1, d2


def bs_price(S, K, r, sigma, t, T, type='CALL'):
    d1, d2 = bs_d(S=S, K=K, r=r, sigma=sigma, t=t, T=T)

    if t == T:
        return european_payoff(S, K, type)

    if type == 'CALL':
        return norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r*(T-t))
    if type == 'PUT':
        return norm.cdf(-d2) * K * np.exp(-r*(T-t)) - norm.cdf(-d1) * S
    return np.nan


if __name__ == '__main__':
    S = 36
    K = 40
    r = 0.06
    sigma = 0.2
    t = 0
    T = 1
    type = 'PUT'

    print(bs_price(S=S, K=K, r=r, sigma=sigma, t=t, T=T, type=type))
