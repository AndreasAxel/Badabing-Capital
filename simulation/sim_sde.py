import numpy as np


def sim_sde_euler(t, x0, N, a, b, seed=None):
    """
    Function for simulating Euler discretization of an Ito-process of the form
            dX(t) = a(X(t), t) * dt + b(X(t), t) * dW(t)

    :param t:       Time steps (including start and endpoints)
    :param x0:      Initial values of the processes
    :param N:       Number of simulations
    :param a:       Drift function of state and time
    :param b:       Diffusion function of state and time
    :param seed:    Seed for replication
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

    for j, s in enumerate(t[1:]):
        x[j + 1] = x[j] + a(x[j], s) * dt[j] + b(x[j], s) * np.sqrt(dt[j]) * Z[j]

    return x


def sim_sde_milstein(t, x0, N, a, b, seed=None):
    """
    Function for performing Milstein simulation of SDE using Runge-Kutta

    :param t:       Time steps (including start and endpoints)
    :param x0:      Initial values of the processes
    :param N:       Number of simulations
    :param a:       Drift function of state and time
    :param b:       Diffusion function of state and time
    :param seed:    Seed for replication
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

    for j, s in enumerate(t[1:]):
        x_hat = x[j] + a(x[j], s) * dt[j] + b(x[j], s) * np.sqrt(dt[j])

        x[j+1] = x[j] + a(x[j], s) * dt[j] + b(x[j], s) * np.sqrt(dt[j]) * Z[j] + \
                 1 / (2 * np.sqrt(dt[j])) * ((np.sqrt(dt[j]) * Z[j]) ** 2 - dt[j]) * (b(x_hat, s) - b(x[j], s))
    return x


if __name__ == '__main__':
    def a(x, t):
        return 0.07 * x

    def b(x, t):
        return 0.2 * x

    t0 = 0.0
    T = 3.0
    x0 = 100
    M = 100
    N = 5
    mu = 0.07
    sigma = 0.2
    seed = 1

    # Equidistant time steps
    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)

    # Perform simulation
    x = sim_sde_euler(t=t, x0=x0, N=N, a=a, b=b, seed=seed)

    y = sim_sde_milstein(t=t, x0=x0, N=N, a=a, b=b, seed=seed)
