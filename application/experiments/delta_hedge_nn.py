import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from application.binomial_model.binomial_model import binomial_tree_bs
from application.models.LetourneauStentoft import ISD
from application.models.regressionModels import DifferentialRegression
from application.simulation.sim_gbm import GBM
from application.Longstaff_Schwartz.LSMC import LSMC
from application.utils.LSMC_fit_predict import fit_poly, pred_poly
from application.options.payoff import european_payoff
from application.models.neural_approximator import *
from application.Longstaff_Schwartz.dataset_generator import gen_LSMC_pathwise_data


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


def nn_fit_predict(x, t, N_train, r, sigma, K, option_type, epochs=100):
    # generate pathwise training data from LSMC
    x_train, y_train, z_train = simulate_pathwise_data(t, N_train, r, sigma, K, option_type, vol_mult=vol_mult)
    # Initialize regressor
    regressor = Neural_approximator(x_raw=x_train, y_raw=y_train, dydx_raw=z_train)
    # Prepare network, True for differentials
    regressor.prepare(N_train, differential=True, weight_seed=None)
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
    M = 50
    r = 0.06
    sigma = 0.2
    K = 40.0
    seed = 1234
    deg_lsmc = 9
    deg_stentoft = 9
    option_type = 'PUT'
    eur_amr = 'AMR'
    alpha = 0.5
    vol_mult = 1.0
    N = 10000
    rep = 1

    delta_lb = -1.0  # Set this to -np.inf if unbounded
    delta_ub = 0.0   # Set this to np.inf if unbounded

    # Variables to vary
    N_train = 4096
    print(N_train)

    # Auxiliary variables
    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)
    dt = T / M

    # Simulate stock paths
    gbm = GBM(t=t, x0=x0, N=N, mu=r, sigma=sigma, use_av=False, seed=seed)
    gbm.sim_exact()
    S = gbm.X

    # Setup Early Exercise Boundary
    binom = binomial_tree_bs(K=K, T=T, S0=x0, r=r, sigma=sigma,
                             M=5000, payoff_func=european_payoff, option_type=option_type, eur_amr=eur_amr)
    binom_eeb = binom[2]
    binom_eeb[np.isnan(binom_eeb)] = np.nanmin(binom_eeb)
    eeb = binom_eeb[[int(5000 / T * s) for s in t]]

    plt.plot(np.linspace(start=0.0, stop=T, num=5000 + 1, endpoint=True), binom_eeb, color='red', label='EEB (M=5000)')
    plt.plot(t, eeb, color='black', label='EEB (M={})'.format(M))
    plt.plot(t, S[:, :100], color='blue', alpha=0.1)
    plt.title('Stock paths & Early Exercise Boundary')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('')
    plt.show()

    # Initialize experiment

    # Actual realization of stock prices
    a = np.full_like(S, np.nan)  # Position in the stock
    b = np.full_like(S, np.nan)  # Position in the risk-free asset
    V = np.full_like(S, np.nan)  # Value-process of our hedge portfolio (excluding what we sold)

    # Find initial hedge
    a[0, :] = np.minimum(delta_ub, np.maximum(
        delta_lb,
        nn_fit_predict(x=np.array(x0).reshape(-1, 1), t=t, N_train=N_train, r=r, sigma=sigma, K=K, option_type=option_type)[1]
    ))
    b[0, :] = binom[0] - a[0, :] * S[0, :]
    V[0, :] = b[0, :] + a[0, :] * S[0, :]

    alive = S[0, :] > eeb[0]
    exercise = np.full_like(S, False, dtype=bool)
    pnl = np.full(N, np.nan)

    for j, s in tqdm(enumerate(t[1:], start=1)):

        # Update positions
        V[j, :] = a[j - 1, :] * S[j, :] + b[j - 1, :] * np.exp(dt * r)
        a[j, :] = np.minimum(delta_ub, np.maximum(
            delta_lb, nn_fit_predict(x=S[j, :].reshape(-1 ,1), t=t[j:], N_train=N_train, r=r, sigma=sigma, K=K, option_type=option_type)[1]
        ))
        a[j, :] = np.array([a if alive[i] else 0.0 for i, a in enumerate(a[j, :])])
        b[j, :] = V[j, :] - a[j, :] * S[j, :]

        exercise[j] = np.minimum((S[j] < eeb[j]), alive)
        alive = np.minimum(alive, ~exercise[j])

    # Extract stopping times (paths not exercised is set to expire at maturity)
    tau_idx = np.argmax(exercise, 0)
    tau_idx = np.array([j if j > 0 else M for j in tau_idx])

    x = np.array([S[tau_idx[i], i] for i in range(N)])
    v = np.array([V[tau_idx[i], i] for i in range(N)])
    p = np.array([np.max([K-S[tau_idx[i], i], 0.0]) for i in range(N)])

    plt.scatter(x, v, color='blue', label='Hedge (V)', alpha=500/N)
    plt.scatter(x, p, color='red', s=2, label='Put (p)', alpha=500/N)
    plt.legend()
    plt.xlabel(r'$S(\tau)$')
    plt.show()

    # Discount factor for PnL
    df = np.array([np.exp(-r*t[j]) for j in tau_idx])

    # Calculate present value of PnL for each path
    pnl = df * (v - p)

    print('mean={:.4f}, std={:.4f}, rmse={:.4f}'.format(np.mean(pnl), np.std(pnl), np.sqrt(np.mean(pnl**2))))

    plt.hist(pnl, bins=100, density=True, label='mean = {:.4f}, std. dev. = {:.4f}'.format(np.mean(pnl), np.std(pnl)))
    plt.axvline(x=np.mean(pnl), ls='--', color='red')
    #plt.title('Density of Hedge Error (PnL)')
    plt.legend(markerscale=0.0, prop={'size': 8}, loc='lower center')
    plt.show()
