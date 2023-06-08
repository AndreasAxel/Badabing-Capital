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


def simulate_pathwise_data(t, N, r, sigma, K, eeb, option_type, vol_mult=1.0):
    rng = np.random.default_rng()
    Z = rng.normal(loc=0.0, scale=1.0, size=N)
    sigma = sigma * vol_mult

    spot = K * np.exp((r - 0.5 * sigma ** 2) * t[-1] + np.sqrt(t[-1]) * sigma * Z)

    gbm = GBM(t=t, x0=spot, N=N, mu=r, sigma=sigma, use_av=True, seed=None)
    gbm.sim_exact()
    S = gbm.X

    M = len(t) - 1

    tau_idx = np.argmax(np.cumsum(S < eeb.reshape(-1, 1), axis=0) > 0, axis=0)
    tau_idx = np.array([j if S[j, i] < eeb[j] else M for i, j in enumerate(tau_idx)])
    tau = t[tau_idx]

    df = np.exp(-r * tau)

    S_tau = np.array([S[j, i] for i, j in enumerate(tau_idx)])

    payoff = df * european_payoff(x=S_tau, K=K, option_type=option_type)
    delta = - df * np.where(S_tau > K, S_tau / spot, 0.0)  # TODO Check this

    # plt.scatter(spot, payoff, alpha=0.05)
    # plt.show()
    # plt.scatter(spot, delta, alpha=0.05)
    # plt.show()

    return spot.reshape(-1, 1), payoff.reshape(-1, 1), delta.reshape(-1, 1)


def diff_reg_fit_predict(x, t, N, r, sigma, K, eeb, option_type, deg=9, alpha=0.5, vol_mult=1.0):
    # Generate pathwise samples
    x_train, y_train, z_train = simulate_pathwise_data(t, N, r, sigma, K, eeb, option_type, vol_mult=vol_mult)

    diff_reg = DifferentialRegression(degree=deg, alpha=alpha)
    diff_reg.fit(x_train, y_train, z_train)
    price, delta = diff_reg.predict(x.reshape(-1, 1), predict_derivs=True)

    return price.reshape(len(x), ), delta.reshape(len(x), )


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

    rep = 1

    # Variables to vary
    N = 10000

    # Auxiliary variables
    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)
    dt = T / M

    # Simulate stock paths
    gbm = GBM(t=t, x0=x0, N=N, mu=r, sigma=sigma, use_av=False, seed=5)
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

    a[0, :] = diff_reg_fit_predict(x=np.array([x0]), t=t, N=N, r=r, sigma=sigma, K=K, eeb=eeb, option_type=option_type)[1]
    b[0, :] = binom[0] - a[0, :] * S[0, :]
    V[0, :] = b[0, :] + a[0, :] * S[0, :]

    alive = S[0, :] > eeb[0]
    exercise = np.full_like(S, False, dtype=bool)

    pnl = np.full(N, np.nan)

    for j, s in tqdm(enumerate(t[1:], start=1)):

        # Perform simulations necessary for determining delta
        x_isd = ISD(N=N, x0=K, alpha=alpha)
        gbm_isd = GBM(t=t[j:], x0=x_isd, N=N, mu=r, sigma=sigma, use_av=True)
        gbm_isd.sim_exact()

        lsmc = LSMC(simulator=gbm_isd, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
        lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

        # Update positions
        V[j, :] = a[j - 1, :] * S[j, :] + b[j - 1, :] * np.exp(dt * r)
        a[j, :] = diff_reg_fit_predict(x=S[j, :], t=t, N=N, r=r, sigma=sigma, K=K, eeb=eeb, option_type=option_type)[1]
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
    plt.scatter(x, p, color='red', label='Put (p)', alpha=500/N)
    plt.legend()
    plt.xlabel('S(tau)')
    plt.show()

    # Discount factor for PnL
    df = np.array([np.exp(-r*t[j]) for j in tau_idx])

    # Calculate present value of PnL for each path
    pnl = df * (v) - p

    print('mean={:.4f}, std={:.4f}, rmse={:.4f}'.format(np.mean(pnl), np.std(pnl), np.sqrt(np.mean(pnl**2))))

    plt.hist(pnl, bins=100, density=True)
    plt.title('Density of Hedge Error (PnL)')
    plt.show()