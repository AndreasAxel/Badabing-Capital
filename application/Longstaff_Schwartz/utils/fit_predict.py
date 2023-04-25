import numpy as np


def fit_poly(x, y, deg, *args, **kwargs):
    return np.polyfit(x, y, deg)


def pred_poly(x, fit, *args, **kwargs):
    return np.polyval(fit, x)


def fit_laguerre_poly(x, y, deg, *args, **kwargs):
    return np.polynomial.laguerre.lagfit(x, y, deg)


def pred_laguerre_poly(x, fit, *args, **kwargs):
    return np.polynomial.laguerre.lagval(x, fit)

