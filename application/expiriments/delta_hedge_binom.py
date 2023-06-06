import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from application.options.payoff import european_payoff
from application.simulation.sim_gbm import GBM
from application.binomial_model.binomial_model import binomial_tree_bs


def binom_delta_helper(s, x, K, T, r, sigma, M_hedge, alive):
    if s != T:
        binom = binomial_tree_bs(K=K, T=T - s, S0=x, r=r, sigma=sigma, M=M_hedge,
                                 payoff_func=european_payoff, eur_amr='AMR')
        delta = binom[1] if alive else 0.0
    else:
        delta = (-1.0 if x <= K else 0.0) if alive else 0.0

    return delta



if __name__ == '__main__':
    M_hedge = 1000  # Number of discretizations used to determine Delta and Early Exercise Boundary
    M = 250         # Number of time steps in simulations
    N = 8 * 12      # Number of paths to simulate
    T = 0.25        # Time to expiry
    r = 0.06        # Risk free rate
    sigma = 0.2     # Volatility
    x0 = 40         # Spot
    K = 40          # Strike

    option_type = 'PUT'
    eur_amr = 'AMR'
    seed = 1234
    
    t = np.linspace(start=0.0, stop=T, num=M+1, endpoint=True)
    dt = 1/M
    
    # Setup Early Exercise Boundary
    binom = binomial_tree_bs(K=K, T=T, S0=x0, r=r, sigma=sigma,
                             M=M_hedge, payoff_func=european_payoff, option_type=option_type, eur_amr=eur_amr)
    binom_eeb = binom[2]
    binom_eeb[np.isnan(binom_eeb)] = np.nanmin(binom_eeb)
    eeb = binom_eeb[[int(M_hedge / T * s) for s in t]]

    # Simulate stock paths
    gbm = GBM(t=t, x0=x0, N=N, mu=r, sigma=sigma, use_av=True, seed=seed)
    gbm.sim_exact()
    S = gbm.X
    
    # Initialize experiment
    a = np.full_like(S, np.nan)
    b = np.full_like(S, np.nan)
    V = np.full_like(S, np.nan)
    
    a[0] = binom[1]
    b[0] = binom[0] - a[0] * S[0]
    V[0] = b[0] + a[0] * S[0]
    
    price = np.full_like(S, np.nan)
    price[0] = binom[0]
    
    alive = np.full(N, True)
    exercise = np.full_like(S, False, dtype=bool)

    for j, s in tqdm(enumerate(t[1:], start=1)):

        exercise[j] = np.minimum((S[j] < eeb[j]), alive)
        alive = np.minimum(alive, ~exercise[j])
    
        V[j] = a[j - 1] * S[j] + b[j - 1] * np.exp(r * dt)

        # Calculate delta used to hedge for each path (parallel)
        a[j, :] = Parallel(n_jobs=-1)(delayed(binom_delta_helper)(s, S[j, i], K, T, r, sigma, M_hedge, alive[i]) for i in range(N))

        """
        for i in range(N):
            if s != T:
                binom = binomial_tree_bs(K=K, T=T - s, S0=S[j, i], r=r, sigma=sigma, M=M_hedge,
                                         payoff_func=european_payoff, eur_amr='AMR')
                price[j, i] = binom[0]
                a[j, i] = binom[1] if alive[i] else 0.0
            else:
                price[j, i] = european_payoff(x=S[j, i], K=K, option_type=option_type)
                a[j, i] = (-1.0 if S[j, i] <= K else 0.0) if alive[i] else 0.0
        """
        b[j] = V[j] - a[j] * S[j]

    # Extract stopping times (paths not unexercised is set to expire at maturity - the sold option is worth 0.0 then)
    tau_idx = np.argmax(exercise, 0)
    tau_idx = np.array([j if j > 0 else M for j in tau_idx])

    # Discount factor for PnL
    df = np.array([np.exp(-r*t[j]) for j in tau_idx])

    # Calculate present value of PnL for each path
    pnl = df * (V[tau_idx, :] - european_payoff(x=S[tau_idx, :], K=K, option_type=option_type))

    print('mean={:.4f}, std={:.4f}, rmse={:.4f}'.format(np.mean(pnl), np.std(pnl), np.sqrt(np.mean(pnl**2))))

    plt.scatter(S[tau_idx, :], V[tau_idx, :], color='blue', alpha=0.1)
    plt.scatter(S[tau_idx, :], european_payoff(x=S[tau_idx, :], K=K, option_type=option_type), color='red', alpha=0.1)
    plt.show()

    idx = 0
    print(tau_idx[idx])
    print(V[tau_idx[idx], idx], np.maximum(K - S[tau_idx[idx], idx], 0.0))
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex='all')
    ax[0].plot(t, S[:, idx])
    ax[0].plot(t, eeb, color='black')
    ax[1].plot(t, a[:, idx])
    ax[2].plot(t, b[:, idx])
    ax[3].plot(t, V[:, idx])

    ax[0].set_ylabel('S')
    ax[1].set_ylabel('a')
    ax[2].set_ylabel('b')
    ax[3].set_ylabel('V')

    ax[1].set_ylim(-1.1, 0.1)
    plt.show()


    
    
