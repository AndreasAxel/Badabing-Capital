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
    M_hedge = 500   # Number of discretizations used to determine Delta and Early Exercise Boundary
    M = 200         # Number of time steps in simulations
    N = 8           # Number of paths to simulate
    T = 0.25        # Time to expiry
    r = 0.01        # Risk free rate
    sigma = 0.3     # Volatility
    x0 = 40         # Spot
    K = 40          # Strike

    option_type = 'PUT'
    eur_amr = 'AMR'
    seed = 1
    
    t = np.linspace(start=0.0, stop=T, num=M+1, endpoint=True)
    dt = 1/M

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

    plt.plot(np.linspace(start=0.0, stop=T, num=5000+1, endpoint=True), binom_eeb, color='red', label='EEB (M=5000)')
    plt.plot(t, eeb, color='black', label='EEB (M={})'.format(M))
    plt.plot(t, S, color='blue', alpha=0.5)
    plt.title('Stock paths & Early Exercise Boundary')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('')
    plt.show()
    
    # Initialize experiment
    a = np.full_like(S, np.nan)
    b = np.full_like(S, np.nan)
    V = np.full_like(S, np.nan)

    a[0] = binom[1]
    b[0] = binom[0] - a[0] * S[0]
    V[0] = b[0] + a[0] * S[0]

    print('P(0, S_0) = {}'.format(binom[0]))
    print('a_0 = {}'.format(a[0, 0]))
    print('b_0 = {}'.format(b[0, 0]))
    print('V_0 = {}'.format(V[0, 0]))

    price = np.full_like(S, np.nan)
    price[0] = binom[0]
    
    alive = np.full(N, True)
    exercise = np.full_like(S, False, dtype=bool)

    for j, s in tqdm(enumerate(t[1:], start=1)):
    
        V[j] = a[j - 1] * S[j] + b[j - 1] * np.exp(r * dt)

        # Calculate delta used to hedge for each path (parallel)

        a[j, :] = Parallel(n_jobs=-2)(delayed(binom_delta_helper)(
            s=s, x=S[j, i], K=K, T=T, r=r, sigma=sigma, M_hedge=M_hedge, alive=alive[i]) for i in range(N))

        b[j] = V[j] - a[j] * S[j]

        exercise[j] = np.minimum((S[j] < eeb[j]), alive)
        alive = np.minimum(alive, ~exercise[j])

    # Extract stopping times (paths not exercised is set to expire at maturity)
    tau_idx = np.argmax(exercise, 0)
    tau_idx = np.array([j if j > 0 else M for j in tau_idx])

    x = np.array([S[tau_idx[i], i] for i in range(N)])
    v = np.array([V[tau_idx[i], i] for i in range(N)])
    p = np.array([np.max([K-S[tau_idx[i], i], 0.0]) for i in range(N)])

    plt.scatter(x, v, color='blue', label='Hedge (V)')
    plt.scatter(x, p, color='red', label='Put (p)')
    plt.legend()
    plt.xlabel('S(tau)')
    plt.show()

    # Discount factor for PnL
    df = np.array([np.exp(-r*t[j]) for j in tau_idx])

    # Calculate present value of PnL for each path
    pnl = df * (v - p)

    print('mean={:.4f}, std={:.4f}, rmse={:.4f}'.format(np.mean(pnl), np.std(pnl), np.sqrt(np.mean(pnl**2))))

