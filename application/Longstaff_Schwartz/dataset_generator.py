import numpy as np
from application.simulation.sim_gbm import sim_gbm
from application.Longstaff_Schwartz.LSMC import lsmc
from application.Longstaff_Schwartz.utils.fit_predict import *
from application.options.payoff import european_payoff
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")


def generate_training_data(t, N, vec_moneyness, vec_r, vec_sigma, vec_T, type, payoff_func, fit_func, pred_func, deg,
                           export_filepath):

    grid = np.array([
        (m, r, sigma, T, np.nan)
        for m in vec_moneyness
        for r in vec_r
        for sigma in vec_sigma
        for T in vec_T
    ]).astype(float)

    def lsmc_price(i):
        t_arg = int(np.argwhere(np.abs(t - float(grid[i, 3])) < 1E-6)[0])
        X = sim_gbm(t=t[:t_arg], x0=float(grid[i, 0]), N=N, mu=float(grid[i, 1]), sigma=float(grid[i, 2]))

        price = lsmc(X=X / K, t=t[:t_arg], K=1, r=float(grid[i, 1]), payoff_func=payoff_func, type=str(type),
                     fit_func=fit_func, pred_func=pred_func, deg=deg)
        return float(price)

    grid[:, 4] = Parallel(n_jobs=-1)(delayed(lsmc_price)(i) for i in tqdm(range(len(grid))))
    np.savetxt(export_filepath,
               grid,
               delimiter=",",
               header="MONEYNESS, r, sigma, T, PRICE")


if __name__ == '__main__':
    # Filename and export-path for data
    export_filepath = os.getcwd().split('application\\')[0] + '\\data\\training_data_PUT.csv'

    # Grid, simulation and polynomial degree
    t0 = 0
    M = 100
    N = 10000
    deg = 3

    # Vectors used in grid
    K = 100
    vec_S = np.array([50 + n for n in range(0, 101)])
    vec_moneyness = vec_S / K
    vec_T = np.array([0.1 * n for n in range(1, 11)])
    vec_r = np.array([0.05 * n for n in range(11)])
    vec_sigma = np.array([0.05 * n for n in range(1, 7)])

    # Option type
    type = 'PUT'

    t = np.linspace(start=t0, stop=np.max(vec_T), num=M + 1, endpoint=True)

    generate_training_data(t, N, vec_moneyness, vec_r, vec_sigma, vec_T, type, payoff_func=european_payoff,
                           fit_func=fit_laguerre_poly, pred_func=pred_laguerre_poly, deg=deg,
                           export_filepath=export_filepath)
