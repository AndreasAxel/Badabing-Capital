import numpy as np


def sim_gbm(t, x0, N, mu, sigma, seed=None):
    """
    Function for simulating Geometric Brownian Motion over specified time steps and initial values of the processes.
    The simulation uses the discretized analytical solution,
        X(t+1) = X(t) * exp{(mu-0.5*sigma^2)*dt + sigma * W(t)}
    :param t:       time steps (including start and endpoints)
    :param x0:      Initial values of the processes
    :param N:       Number of simulations
    :param mu:      Drift coefficient
    :param sigma:   Diffusion coefficient (volatility)
    :param seed:    Seed used for replication of results
    :return:        numpy.array with dim (M+1, N).
                    Rows are time steps (including the start and end-point)
                    Columns are the number of simulations
    """

    assert np.ndim(t) == 1, 'Time steps must be a 1-dimensional array.'
    M = len(t) - 1

    dt = np.diff(t)

    rng = np.random.default_rng(seed=seed)
    Z = rng.standard_normal(size=(M, N))

    x = np.zeros(shape=(M + 1, N))
    x[0] = x0 * np.ones(shape=(1, N))

    for j in range(len(t[1:])):
        x[j+1] = x[j] * np.exp((mu-0.5*sigma**2)*dt[j] + sigma*np.sqrt(dt[j])*Z[j])

    return x


if __name__ == '__main__':
    # Parameters
    t0 = 0.0
    T = 3.0
    x0 = [50, 100, 50, 100, 50]
    M = 10
    N = 5
    mu = 0.07
    sigma = 0.2
    seed = 1

    # Equidistant time steps
    t = np.linspace(start=t0, stop=T, num=M+1, endpoint=True)

    # Perform simulation
    x = sim_gbm(t=t, x0=x0, N=N, mu=mu, sigma=sigma, seed=seed)
    print(x)
