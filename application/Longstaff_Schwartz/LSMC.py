import numpy as np
from application.options.payoff import european_payoff
from application.simulation.sim_gbm import sim_gbm
from application.Longstaff_Schwartz.utils.fit_predict import *
import warnings
warnings.filterwarnings("ignore")


def lsmc(X, t, K, r, payoff_func, type, fit_func, pred_func, *args, **kwargs):
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
    :return:                Price of the american option
    """

    assert np.ndim(t) == 1, "Time steps must be a 1-dimensional array."
    assert np.ndim(X) == 2, "The passed simulations `X` mush be a 2-dimensional numpy-array."
    assert len(X) == len(t), "The length of the passed simulations `X` and time steps `t` must be of same length."

    M = len(t) - 1
    dt = np.diff(t)
    df = np.exp(-r * dt)

    # Initialize objects
    payoff = payoff_func(X, K, type)
    V = np.zeros_like(payoff)
    V[M, :] = payoff[M, :]
    stopping_rule = np.zeros_like(X).astype(dtype=bool)
    stopping_rule[M, :] = payoff[M, :] > 0

    for j in range(M - 1, 0, -1):
        # Fit prediction model
        fit = fit_func(X[j, :], V[j + 1, :] * df[j], *args, **kwargs)

        # Predict value of continuation
        EV_cont = pred_func(X[j, :], fit)

        # Determine whether is it optimal to exercise or continue. Update values accordingly
        stopping_rule[j, :] = payoff[j, :] > np.maximum(EV_cont, 0.0)
        V[j, :] = np.where(stopping_rule[j, :],
                           payoff[j, :], V[j + 1, :] * df[j])  # Exercise / Continuation

    # Format stopping rule
    stopping_rule = np.cumsum(np.cumsum(stopping_rule[1:], axis=0), axis=0) == 1

    return np.mean(V[1, :] * df[0])


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
    print("Price Longstaff-Schwarz: ", lsmc(t=t, X=X, K=K, r=r, payoff_func=european_payoff, type=type,
                                            fit_func=fit_poly, pred_func=pred_poly, deg=deg))

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

    t = np.linspace(start=t0, stop=T, num=M+1, endpoint=True)
    X = sim_gbm(t=t, x0=x0, N=N, mu=r, sigma=sigma, seed=seed)

    for deg in range(10):
        print('deg =', deg, ": Price with GBM simulation =", lsmc(t=t, X=X, K=K, r=r, payoff_func=european_payoff,
                                                                  type=type, fit_func=fit_laguerre_poly,
                                                                  pred_func=pred_laguerre_poly,
                                                                  deg=deg))

    # Use Neural Network
    print("Price with Sequentical Neural Network =", lsmc(t=t, X=X, K=K, r=r, payoff_func=european_payoff, type=type,
                                                          fit_func=NN_fit, pred_func=NN_pred, num_epochs=2,
                                                          batch_size=32*4))
