import numpy as np
from sklearn.neighbors import KernelDensity

def sim_ISD(N, x0:float, alpha, seed=None, *args, **kwargs):
    """
    Creates N initially state dispersed variables (ISD).
    With reference to Latourneau & Stentoft (2023)

    :param N:       no. paths
    :param x0:      init. value
    :param alpha:   band width dispersion
    :param seed:    seed
    :param args:
    :param kwargs:
    :return:        size N vector of state dispersed vars around x0
    """

    # epanechnikov kernel
    vecUnif = np.random.default_rng(seed=seed).uniform(low=0, high=1, size=N)
    kernel = 2 * np.sin( np.arcsin(2 * vecUnif -1) / 3)

    # Initial state dispersion
    X = float(x0) + alpha * kernel
    return X

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.hist(sim_ISD(N=1000, x0=100.0, alpha=5), bins=50)
    plt.show()
