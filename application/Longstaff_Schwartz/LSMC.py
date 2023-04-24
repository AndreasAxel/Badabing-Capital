import numpy as np
from application.options.payoff import european_payoff
from application.simulation.sim_gbm import sim_gbm


def lsmc(t, X, K, r, payoff_func, type, deg):
    """
    Longstaff-Schwartz Monte Carlo method for pricing an American option.

    :param t:               Time steps
    :param X:               Simulated paths
    :param K:               Strike
    :param r:               Risk free rate
    :param payoff_func:     Payoff function to be called
    :param type:            Type of option
    :param deg:             Degree of the polynomial
    :return:                Price of the american option
    """

    assert np.ndim(t) == 1, "Time steps must be a 1-dimensional array."
    assert np.ndim(X) == 2, "The passed simulations `X` mush be a 2-dimensional numpy-array."
    assert len(X) == len(t), "The length of the passed simulations `X` and time steps `t` must be of same length."

    M = len(t) - 1
    N = np.shape(X)[1]

    dt = np.diff(t)

    # Initiate
    discount_factor = np.exp(-r * dt)
    payoff = payoff_func(X, K, type)

    # formatting stopping rule and cashflow-matrix
    stopping_rule = np.full((M, N), False)
    cashflow = np.full((M, N), np.nan)

    stopping_rule[M-1:, ] = payoff[M, :] > 0
    cashflow[M-1, :] = payoff[M, :]

    for j in range(M-1, 0, -1):
        itm = payoff[j, :] > 0

        # Perform regression
        coeff = np.polyfit(x=X[j, itm], y=cashflow[j, itm] * discount_factor[-j], deg=deg)

        # Determine stopping rule by
        # comparing the value of exercising with the expected value of continuation
        EV_cont = np.polyval(p=coeff, x=X[j, itm])
        exercise = payoff[j, itm]

        stopping_rule[j-1, itm] = exercise > EV_cont

        # Update cash-flow matrix
        cashflow[j-1, :] = stopping_rule[j-1, :] * payoff[j, :]

    # Format stopping rule and cashflow-matrix
    stopping_rule = np.cumsum(np.cumsum(stopping_rule, axis=0), axis=0) == 1
    cashflow = stopping_rule * cashflow

    # Calculate price
    price = np.mean(cashflow.T @ np.cumprod(discount_factor))

    return price


if __name__ == '__main__':
    # Example from the Longstaff-Schwartz article
    K = 1.10
    r = 0.06
    t = np.linspace(start=0, stop=3, num=4)
    type = 'PUT'
    deg = 2


    X = np.array((
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        [1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88],
        [1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.22],
        [1.34, 1.54, 1.03, 0.92, 1.52, 0.90, 1.01, 1.34]
    ))
    print("Price Longstaff-Schwarz: ", lsmc(t=t, X=X, K=K, r=r, payoff_func=european_payoff, type=type, deg=deg))


    # Simulating with GBM
    x0 = 1.0
    t0 = 0.0
    T = 1.0
    N = 10000
    M = 252
    sigma = 0.2
    seed = 1234

    t = np.linspace(start=t0, stop=T, num=M+1, endpoint=True)
    X = sim_gbm(t=t, x0=x0, N=N, mu=r, sigma=sigma, seed=seed)

    print("Price with GBM simulation: ", lsmc(t=t, X=X, K=K, r=r, payoff_func=european_payoff, type=type, deg=deg))

