import numpy as np
import pandas as pd
from tqdm import tqdm
from application.options.payoff import european_payoff
from application.binomial_model.binomial_model import binomial_tree_bs
from application.simulation.sim_gbm import GBM
from application.Longstaff_Schwartz.LSMC import LSMC
from application.utils.LSMC_fit_predict import pred_poly, fit_poly
from application.utils.path_utils import get_data_path


if __name__ == '__main__':
    export_path = get_data_path('LSMC_convergence_test.csv')

    # Constant model parameters
    K = 40
    r = 0.06
    t0 = 0.0
    T = 1.0
    M = 50
    sigma = 0.2
    seed = None

    option_type = 'PUT'
    use_av = True

    # Parameters to vary
    N = [1000, 5000, 10000, 50000, 100000]
    x0 = [40]
    deg = list(range(1, 10))

    # Number of repetitions
    repeat = 100

    # Equidistant time steps
    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)

    # Binomial model as "true" prices
    prices_binom = np.array([
        binomial_tree_bs(K=K, T=T, S0=spot, r=r, sigma=sigma, M=10000,
                         payoff_func=european_payoff, option_type=option_type, eur_amr='AMR')[0]
        for spot in x0
    ], dtype=np.float64)

    # Objects for storing results
    df = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(repeat), N, x0, deg], names=['REP', 'N', 'SPOT', 'DEG']),
        columns=['PRICE', 'ERROR', 'ABS_ERROR'], dtype=np.float64
    )

    # Perform simulations, calculate prices and errors using LSMC
    for rep in tqdm(range(repeat)):
        for i, n in enumerate(N):
            for j, x in enumerate(x0):
                gbm = GBM(t=t, x0=x, N=n, mu=r, sigma=sigma, use_av=use_av, seed=seed)
                gbm.sim_exact()
                lsmc = LSMC(simulator=gbm, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
                for k, d in enumerate(deg):
                    lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=d)
                    df.loc[rep, n, x, d]['PRICE', 'ERROR', 'ABS_ERROR'] = (
                        lsmc.price, lsmc.price-prices_binom[j], np.abs(lsmc.price-prices_binom[j])
                    )

    # Summary statistics
    df_summary = pd.DataFrame(
        index=df.groupby(['N', 'SPOT', 'DEG']).count().index,
        columns=['LSMC_MEAN_PRICE', 'LSMC_STD_PRICE', 'BINOM_PRICE', 'RMSE', 'STD_ERROR'],
        dtype=np.float64
    )

    for i, x in enumerate(x0):
        df_summary.loc[np.array(df_summary.reset_index()['SPOT'] == x), 'BINOM_PRICE'] = prices_binom[i]
    df_summary['LSMC_MEAN_PRICE'] = df.groupby(['N', 'SPOT', 'DEG'])['PRICE'].mean()
    df_summary['LSMC_STD_PRICE'] = df.groupby(['N', 'SPOT', 'DEG'])['PRICE'].std()
    df_summary['RMSE'] = df.groupby(['N', 'SPOT', 'DEG'])['ERROR'].apply(lambda x: np.sqrt(np.mean(x**2)))
    df_summary['STD_ERROR'] = df.groupby(['N', 'SPOT', 'DEG'])['ERROR'].std()

    # Export summary results
    print(df_summary.droplevel(1).to_latex(float_format='%.4f'))
    df_summary.to_csv(export_path, float_format='%.4f')
