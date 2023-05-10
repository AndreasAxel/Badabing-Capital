import numpy as np
from application.options.payoff import european_payoff
from application.simulation.sim_gbm import sim_gbm
from application.Longstaff_Schwartz.utils.fit_predict import *


def lsmc(X, t, K, r, payoff_func, type, fit_func, pred_func, return_pathwise_opt_stopping=False, *args, **kwargs):
    """
    Longstaff-Schwartz Monte Carlo method for pricing an American option.

    :param t:               Time steps
    :param X:               Simulated paths
    :param K:               Strike
    :param r:               Risk free rate
    :param payoff_func:     Payoff function to be called
    :param type:            Type of option
    :param fit_func         Function for fitting the (expected) value of value of continuation
    :param pred_func        Function for predicting the (expected) value of continuation
    :param return_pathwise_opt_stopping
                            Boolean determining if the pathwise optimal stopping time and index should be returned
    :return:                Price of the american option,
                            time of the pathwise optimal stopping time
                            index of the pathwise optimal stopping times
    """

    assert np.ndim(t) == 1, "Time steps must be a 1-dimensional array."
    assert np.ndim(X) == 2, "The passed simulations `X` mush be a 2-dimensional numpy-array."
    assert len(X) == len(t), "The length of the passed simulations `X` and time steps `t` must be of same length."

    M = len(t) - 1
    N = X.shape[1]
    dt = np.diff(t)
    df = np.exp(-r * dt)

    # Initialize objects
    payoff = payoff_func(X, K, type)
    V = np.zeros_like(payoff)
    V[M, :] = payoff[M, :]
    stopping_rule = np.zeros_like(X).astype(dtype=bool)
    stopping_rule[M, :] = payoff[M, :] > 0

    for j in range(M - 1, 0, -1):
        # Determine In-The-Money paths
        # According to Longstaff & Schwart's article this improves efficiency and computation time.
        itm = payoff[j, :] > 0

        # Fit prediction model
        fit = fit_func(X[j, itm], V[j + 1, itm] * df[j], *args, **kwargs)

        # Predict value of continuation
        pred = list(pred_func(X[j, itm], fit))
        EV_cont = np.zeros((N,))
        EV_cont[itm] = pred

        # Determine whether is it optimal to exercise or continue. Update values accordingly
        stopping_rule[j, :] = payoff[j, :] > EV_cont
        V[j, :] = np.where(stopping_rule[j, :],
                           payoff[j, :], V[j + 1, :] * df[j])  # Exercise / Continuation

    if not return_pathwise_opt_stopping:
        return np.mean(V[1, :] * df[0])

    else:
        # Matrix from the paper with stopping rule
        # opt_stopping_rule_matrix = np.cumsum(np.cumsum(stopping_rule[1:], axis=0), axis=0) == 1

        # Determine index and time of pathwise optimal stopping strategy
        pathwise_opt_stopping_idx = np.argmax(stopping_rule, axis=0).astype(float)
        pathwise_opt_stopping_idx[pathwise_opt_stopping_idx == 0] = np.nan  # Not exercising at time 0

        pathwise_opt_stopping_time = [idx if np.isnan(idx) else t[int(idx)] for idx in pathwise_opt_stopping_idx]

        return np.mean(V[1, :] * df[0]), pathwise_opt_stopping_time, pathwise_opt_stopping_idx


def lsmc_pathwise_delta(X, t, K, r, pathwise_opt_stopping_idx, payoff_func, type='PUT'):
    """
    Calculate delta by algorithm 2 in
        'Fast Estimates of Greeks from American Options - A Case Study in Adjoint Algorithm Differentiation'
        by Deussen, Mosenkis, Naumann (2018).
    And slides
        'Fast Delta-Estimates for American Options by Adjoint Algorithmic Differentiation'
        by Deussen (Autodiff, 2015)

    :param t:               Time steps
    :param X:               Simulated paths
    :param K:               Strike
    :param r:               Risk free rate
    :param pathwise_opt_stopping_idx:
    :param payoff_func:     Payoff function to be called
    :param type:            Type of option
    :return:                Option price, Option delta
    """
    V_out = 0.0
    g_out = 0.0

    N = np.shape(X)[1]

    for p in range(N):
        idx = pathwise_opt_stopping_idx[p]
        if np.isnan(idx):
            continue
        idx = int(idx)
        S = X[idx, p]
        V_out += payoff_func(S, K, type) * np.exp(-r * t[idx])
        g_out += - S * np.exp(-r * t[idx]) / X[0, 0]  # Dirty delta calculation (this can be replaced by AD)

    V_out = V_out / N
    g_out = g_out / N

    return V_out, g_out


if __name__ == '__main__':
    # Example from the Longstaff-Schwartz article
    K = 1.1
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
    price, pathwise_opt_stopping_time, pathwise_opt_stopping_idx = lsmc(t=t, X=X, K=K, r=r, payoff_func=european_payoff,
                                                                        type=type, fit_func=fit_poly,
                                                                        pred_func=pred_poly,
                                                                        return_pathwise_opt_stopping=True, deg=deg)

    print("Price Longstaff-Schwarz: ", price)
    print("Pathwise optimal stopping time: \n", pathwise_opt_stopping_time)
    print("Pathwise optimal stopping index: \n", pathwise_opt_stopping_idx)

    # Simulating with GBM
    x0 = 36
    K = 40
    r = 0.06
    t0 = 0.0
    T = 1.0
    N = 100000
    M = 50
    sigma = 0.2
    seed = 1234

    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)
    X = sim_gbm(t=t, x0=x0, N=N, mu=r, sigma=sigma, seed=seed)

    for deg in range(10):
        price, pathwise_opt_stopping_time, pathwise_opt_stopping_idx = lsmc(t=t, X=X, K=K, r=r, payoff_func=european_payoff,
                                                                            type=type, fit_func=fit_poly,
                                                                            pred_func=pred_poly,
                                                                            return_pathwise_opt_stopping=True,
                                                                            deg=deg)

        price_ad, delta_ad = lsmc_pathwise_delta(X=X, t=t, K=K, r=r, pathwise_opt_stopping_idx=pathwise_opt_stopping_idx,
                                                 payoff_func=european_payoff,
                                                 type=type
                                                 )
        print('deg = {}: Price = {:.4f}, Delta ={:.4f}'.format(deg, price, delta_ad))


    # Use Neural Network
    # print("Price with Sequentical Neural Network =", lsmc(t=t, X=X, K=K, r=r, payoff_func=european_payoff, type=type,
    #                                                      fit_func=NN_fit, pred_func=NN_pred, num_epochs=2,
    #                                                      batch_size=32*4))
