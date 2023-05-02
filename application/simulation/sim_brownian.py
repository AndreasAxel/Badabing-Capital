import numpy as np
import matplotlib.pyplot as plt


def sim_brownian_motion(t, N, use_av=False, seed=None):
    """
    Simulate standard brownian motion (Wiener process).
    :param t:           Time points
    :param N:           Number of simulations
    :param use_av:      Use antithetic variates
    :param seed:        Seed for replication
    :return:            Brownian motion
    """
    assert np.ndim(t) == 1, 'Time steps must be a 1-dimensional array.'
    M = len(t) - 1

    dt = np.diff(t)

    rng = np.random.default_rng(seed=seed)

    if use_av:
        Z = rng.standard_normal(size=(M, N//2))
        Z = np.hstack([Z, -Z])
    else:
        Z = rng.standard_normal(size=(M, N))

    W = np.zeros(shape=(M + 1, N))
    W[0] = np.zeros(shape=(1, N))

    for j in range(M):
        W[j+1] = W[j] + np.sqrt(dt[j]) * Z[j]

    return W


def sim_brownian_bridge(t, W_t, W_T, N, use_av=False, seed=None):
    """
    Simulate brownian bridge.
    :param t:           Time points
    :param W_t_         Initial value of brownian motion
    :param W_T          Terminal value of brownian motion
    :param N:           Number of simulations
    :param use_av:      Use antithetic variates
    :param seed:        Seed for replication
    :return:            Brownian bridge
    """
    assert np.ndim(t) == 1, 'Time steps must be a 1-dimensional array.'

    M = len(t) - 1
    dt = np.diff(t)

    rng = np.random.default_rng(seed=seed)

    if use_av:
        Z = rng.standard_normal(size=(M, N // 2))
        Z = np.hstack([Z, Z])
    else:
        Z = rng.standard_normal(size=(M, N))

    B = np.zeros(shape=(M + 1, N))
    B[0] = W_t

    for j in range(M):
        mu = W_T * (M-1) / (M-1+1)
        sigma = np.sqrt(dt[j] * (M-j) / (M-j+1))
        B[j+1] = B[j] + mu * np.sqrt(sigma) * Z[j]
    return B


def bridge_brownian_motion(t_W, t_B, W, B):
    """
    Combine a brownian motion and its bridge to get a more granular simulation
    :param t_W: Time points of the brownian motion
    :param t_B: Time points of the brownian bridge
    :param W:   Values of the brownian motion
    :param B:   Values of the brownian bridge
    :return:    The filled timeline and filled brownian motion (both sorted by time)
    """
    assert len(t_W) == len(W), 'Length of brownian motion and its timeline must be equal.'
    assert len(t_B) == len(B), 'Length of brownian bridge and its timeline must be equal.'
    t = np.concatenate([t_W, t_B[1:-1]])
    order = np.argsort(t)
    W_ = np.vstack([W, B])[order, :]
    t = t[order]
    return t, W_


if __name__ == '__main__':
    t0 = 0.0
    T = 1.0
    M1 = 3
    M2 = 20
    N = 2
    use_av = True
    seed = 1234
    break_index = 2

    t1 = np.linspace(start=t0, stop=T, num=M1 + 1, endpoint=True)
    t2 = np.linspace(start=t1[break_index], stop=t1[break_index+1], num=M2 + 1, endpoint=True)

    W = sim_brownian_motion(t=t1, N=N, use_av=use_av, seed=seed)
    B = sim_brownian_bridge(t=t2, W_t=W[break_index], W_T=W[break_index+1], N=N, use_av=use_av, seed=seed)

    t_, W_ = bridge_brownian_motion(t_W=t1, t_B=t2, W=W, B=B)

    plt.plot(t_, W_)
    plt.title('Brownian motion filled with brownian bridge')
    plt.xlabel('t')
    plt.ylabel('W(t)')
    plt.show()
