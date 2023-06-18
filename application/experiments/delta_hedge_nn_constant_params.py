import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from application.binomial_model.binomial_model import binomial_tree_bs
from application.simulation.sim_gbm import GBM
from application.Longstaff_Schwartz.LSMC import LSMC
from application.utils.LSMC_fit_predict import fit_poly, pred_poly
from application.options.payoff import european_payoff
from application.models.neural_approximator import Neural_approximator


def simulate_pathwise_data(t, N, r, sigma, K, option_type, vol_mult=1.0):
    rng = np.random.default_rng()
    Z = rng.normal(loc=0.0, scale=1.0, size=N)

    sigma = sigma * vol_mult

    spot = K * np.exp((r - 0.5 * sigma ** 2) * t[-1] + np.sqrt(t[-1]) * sigma * Z)

    simulator = GBM(t=t, x0=spot, N=N, mu=r, sigma=sigma, use_av=True, seed=None)
    simulator.sim_exact()
    lsmc = LSMC(simulator=simulator, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
    lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=9)

    tau = np.array([t if ~np.isnan(t) else 1.0 for t in lsmc.pathwise_opt_stopping_time])
    df = np.exp(-r * tau)

    one = ~np.isnan(lsmc.pathwise_opt_stopping_time)
    S_tau = np.sum(lsmc.X * lsmc.opt_stopping_rule, axis=0)
    payoff = df * np.sum(lsmc.payoff * lsmc.opt_stopping_rule, axis=0)
    delta = - df * S_tau / spot * one

    return spot.reshape(-1, 1), payoff.reshape(-1, 1), delta.reshape(-1, 1)


def nn_fit_predict(x, t, N_train, r, sigma, K, option_type, hidden_layers, hidden_units, lambda_, epochs=100):
    # generate pathwise training data from LSMC
    x_train, y_train, z_train = simulate_pathwise_data(t, N_train, r, sigma, K, option_type)

    # Initialize regressor
    regressor = Neural_approximator(x_raw=x_train, y_raw=y_train, dydx_raw=z_train)

    # Prepare network, True for differentials
    regressor.prepare(N_train, differential=True, weight_seed=None, hidden_layers=hidden_layers, hidden_units=hidden_units, lam=lambda_)

    # Train network
    regressor.train('differential', epochs=epochs)

    # Predict on spot
    predictions, deltas = regressor.predict_values_and_derivs(x)

    return predictions[:, 0], deltas[:, 0]


if __name__ == '__main__':
    # Fixed parameters
    t0 = 0.0
    T = 0.25
    x0 = 40.0
    K = 40.0
    M = 50
    N = 10000
    r = 0.06
    sigma = 0.2

    seed = 1234
    deg_lsmc = 9
    option_type = 'PUT'
    eur_amr = 'AMR'

    delta_lb = -1.0  # Set this to -np.inf if unbounded
    delta_ub = 0.0  # Set this to np.inf if unbounded

    # Variables to vary
    lambda_ = np.array([0.0, 1.0])  # Regularization (MSE of price only, MSE of price and delta)
    hidden_layers = {1: 35, 4: 5, 9: 3} # hidden layers (keys) with neurons (vals)
                                        # designed to have params constant

    N_train = np.array([128 * 2 ** i for i in range(6)])

    # Auxiliary variables
    param = np.array([(hl, lam) for hl in hidden_layers.keys() for lam in lambda_])
    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)
    dt = T / M

    # Simulate stock paths
    gbm = GBM(t=t, x0=x0, N=N, mu=r, sigma=sigma, use_av=True, seed=seed)
    gbm.sim_exact()
    S = gbm.X

    # Setup Early Exercise Boundary
    binom = binomial_tree_bs(K=K, T=T, S0=x0, r=r, sigma=sigma,
                             M=5000, payoff_func=european_payoff, option_type=option_type, eur_amr=eur_amr)
    binom_eeb = binom[2]
    binom_eeb[np.isnan(binom_eeb)] = np.nanmin(binom_eeb)
    eeb = binom_eeb[[int(5000 / T * s) for s in t]]


    def hedge_expirement(hl, lam):
        # Array for storing hedge errors
        hedge_err_std = np.full((len(N_train),), np.nan)

        # choose neurons from dict
        nl = hidden_layers[hl]

        for k, n in enumerate(N_train):

            # ----------------------------------------- #
            # Initialize experiment                     #
            # ----------------------------------------- #

            alive = S[0, :] > eeb[0]
            exercise = np.full_like(S, False, dtype=bool)

            # Differential Neural Network
            a = np.full_like(S, np.nan)
            b = np.full_like(S, np.nan)
            V = np.full_like(S, np.nan)

            # Find initial hedge
            a[0, :] = np.minimum(delta_ub, np.maximum(
                delta_lb,
                nn_fit_predict(x=np.array(S[0, :]).reshape(-1, 1), t=t, N_train=n, r=r, sigma=sigma, K=K,
                               option_type=option_type, hidden_layers=hl, hidden_units=nl, lambda_=lam)[1]
            ))
            b[0, :] = binom[0] - a[0, :] * S[0, :]
            V[0, :] = b[0, :] + a[0, :] * S[0, :]

            # ----------------------------------------- #
            # Dynamics hedging experiment               #
            # ----------------------------------------- #

            for j, s in enumerate(t[1:], start=1):
                V[j, :] = a[j - 1, :] * S[j, :] + b[j - 1, :] * np.exp(dt * r)
                a[j, :] = np.minimum(delta_ub, np.maximum(
                    delta_lb, nn_fit_predict(x=S[j, :].reshape(-1, 1), t=t[j:], N_train=n, r=r, sigma=sigma, K=K,
                                             option_type=option_type, hidden_layers=hl, hidden_units=nl, lambda_=lam)[1]
                ))
                a[j, :] = np.array([a if alive[i] else 0.0 for i, a in enumerate(a[j, :])])
                b[j, :] = V[j, :] - a[j, :] * S[j, :]

                # Option holder makes exercise decision
                exercise[j] = np.minimum((S[j] < eeb[j]), alive)
                alive = np.minimum(alive, ~exercise[j])

            # Extract stopping times (paths not exercised is set to expire at maturity)
            tau_idx = np.argmax(exercise, 0)
            tau_idx = np.array([j if j > 0 else M for j in tau_idx])

            # Discount factor for PnL
            df = np.array([np.exp(-r * t[j]) for j in tau_idx])

            # ----------------------------------------- #
            # Calculate PnL (Hedge Error)               #
            # ----------------------------------------- #

            # Payoff on the sold put option
            p = np.array([np.max([K - S[tau_idx[i], i], 0.0]) for i in range(N)])

            # Value of hedge portfolio
            v = np.array([V[tau_idx[i], i] for i in range(N)])

            # PnL (hedge error)
            pnl = df * (v - p)

            # Update results
            hedge_err_std[k] = np.std(pnl)

        return {'HIDDEN_LAYERS': hl, 'LAMBDA': lam, 'STD_HEDGE_ERR': hedge_err_std}


    res = []
    for (hl, lam) in tqdm(param):
        res.append(hedge_expirement(int(hl), lam))

    df = pd.concat([pd.DataFrame(x) for x in res]).reset_index().rename(columns={'index': 'N_TRAIN'})
    df['N_TRAIN'] = N_train[df['N_TRAIN']]

    # ----------------------------------------- #
    # Plot results                              #
    # ----------------------------------------- #

    color_lambda = {0.0: 'orange', 1.0: 'blue'}

    for i, (hl, lam) in enumerate(param):
        print(i, (hl, lam))
        plt.plot(N_train, df[(df['HIDDEN_LAYERS'] == hl) & (df['LAMBDA'] == lam)]['STD_HEDGE_ERR'],
                 color=color_lambda[lam], marker='o')
        plt.text(np.min(N_train) * 0.9,
                 df[(df['HIDDEN_LAYERS'] == hl) & (df['LAMBDA'] == lam)]['STD_HEDGE_ERR'].head(1) * 1.025,
                 'L={}xN={}'.format(int(hl), int(hidden_layers[hl])), color=color_lambda[lam])

    plt.ylim(df['STD_HEDGE_ERR'].min() * 0.8, df['STD_HEDGE_ERR'].max() * 1.2)
    plt.ylabel('STD( hedge error )')
    plt.xlabel('Training Samples')
    plt.semilogx(base=2)
    plt.xticks(ticks=N_train, labels=N_train)
    plt.text(np.max(N_train) * 0.5, df['STD_HEDGE_ERR'].max() * 1.15, 'N = {:,}'.format(N))
    plt.text(np.min(N_train), df['STD_HEDGE_ERR'].max() * 1.15, 'Regularized NN (λ = 1.0)', color='blue')
    plt.text(np.min(N_train), df['STD_HEDGE_ERR'].max() * 1.10, 'Classical NN      (λ = 0.0)', color='orange')
    plt.show()
