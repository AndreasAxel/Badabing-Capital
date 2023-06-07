import numpy as np
import matplotlib.pyplot as plt


def payoff(x, K, option_type='PUT'):
    if option_type == 'PUT':
        return np.maximum(K-x, 0.0)
    elif option_type == 'CALL':
        return np.maximum(x-K, 0.0)
    else:
        raise NotImplementedError


def binomial_tree(K, T, S0, r, M, u, d, payoff_func, option_type='PUT', eur_amr='AMR'):
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
    V = payoff_func(S, K, option_type)
    delta = np.nan
    B = np.full(shape=(M+1,), fill_value=np.nan)
    B[M] = K

    # Backward recursion through the tree
    for i in np.arange(M - 1, -1, -1):
        S = S0 * d ** np.arange(i, -1, -1) * u ** np.arange(0, i + 1, 1)
        V[:i + 1] = df * (q * V[1:i + 2] + (1 - q) * V[0:i + 1])
        V = V[:-1]
        if eur_amr == 'AMR':
            payoff = payoff_func(S, K, option_type)
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

    M_hedge = 100  # Number of discretizations used to determine Delta and Early Exercise Boundary
    M = 1000          # Number of time steps in simulations
    N = 8           # Number of paths to simulate
    T = 0.25        # Time to expiry
    r = 0.06        # Risk free rate
    sigma = 0.2     # Volatility
    x0 = 40         # Spot
    K = 40          # Strike

    option_type = 'PUT'
    eur_amr = 'AMR'
    seed = 2

    t = np.linspace(start=0.0, stop=T, num=M + 1, endpoint=True)
    dt = 1 / M

    # Simulate stock paths
    rng = np.random.default_rng(seed=seed)
    Z = rng.standard_normal(size=(M, N))
    dW = Z * np.sqrt(np.diff(t).reshape(-1, 1))
    W = np.cumsum(np.vstack([np.zeros(shape=(1, N)), dW]), axis=0)
    S = x0 * np.exp((r - 0.5 * sigma ** 2) * t.reshape(-1, 1) + sigma * W)

    # Setup Early Exercise Boundary
    binom = binomial_tree_bs(K=K, T=T, S0=x0, r=r, sigma=sigma,
                             M=5000, payoff_func=payoff, option_type=option_type, eur_amr=eur_amr)
    binom_eeb = binom[2]
    binom_eeb[np.isnan(binom_eeb)] = np.nanmin(binom_eeb)
    eeb = binom_eeb[[int(5000 / T * s) for s in t]]

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

    plt.plot(np.linspace(start=0.0, stop=T, num=5000+1, endpoint=True), binom_eeb, color='red', label='EEB (M=5000)')
    plt.plot(t, eeb, color='black', label='EEB (M={})'.format(M))
    plt.plot(t, S, color='blue', alpha=0.5)
    plt.title('Stock paths & Early Exercise Boundary')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('')
    plt.show()

    price = np.full_like(S, np.nan)
    price[0] = binom[0]

    alive = np.full(N, True)
    exercise = np.full_like(S, False, dtype=bool)

    # Hedge dynamically
    for j, s in enumerate(t[1:], start=1):

        # Step 1 - Update value of hedge portfolio
        V[j] = a[j - 1] * S[j] + b[j - 1] * np.exp(r * dt)

        # Step 2 - Calculate delta used to hedge for each path
        for i in range(N):
            if s != T:
                binom = binomial_tree_bs(K=K, T=T - s, S0=S[j, i], r=r, sigma=sigma, M=M_hedge,
                                         payoff_func=payoff, eur_amr='AMR')
                price[j, i] = binom[0]
                a[j, i] = binom[1] if alive[i] else 0.0
            else:
                price[j, i] = payoff(x=S[j, i], K=K, option_type=option_type)
                a[j, i] = (-1.0 if S[j, i] <= K else 0.0) if alive[i] else 0.0

        # Step 3 - Update amount invested in the risk free asset
        b[j] = V[j] - a[j] * S[j]

        # Step 4 - Check if option is exercised
        exercise[j] = np.minimum((S[j] < eeb[j]), alive)
        alive = np.minimum(alive, ~exercise[j])

    # Extract stopping times (paths not exercised is set to expiry)
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
    df = np.array([np.exp(-r * t[j]) for j in tau_idx])

    # Calculate present value of PnL for each path
    pnl = df * (v - p)

    print('mean={:.4f}, std={:.4f}, rmse={:.4f}'.format(np.mean(pnl), np.std(pnl), np.sqrt(np.mean(pnl ** 2))))

    err = V - price
    for i in range(N):
        if M - tau_idx[i] < 1:
            color = 'blue'
        else:
            color = 'red'
        plt.plot(t[:tau_idx[i]], err[:tau_idx[i], i], color=color)
    plt.xlabel('t')
    plt.ylabel('V - price')
    plt.title('Hedge Error')
    plt.show()
