import numpy as np
from application.simulation.sim_gbm import GBM
from application.Longstaff_Schwartz.LSMC import LSMC
from application.Longstaff_Schwartz.utils.fit_predict import *
from application.options.payoff import european_payoff
from joblib import Parallel, delayed
from tqdm import tqdm
from application.utils.path_utils import get_data_path


def gen_LSMC_data(t, vec_spot, r, sigma, K, N, export_filepath):

    def calc(s):
        simulator = GBM(t=t, x0=s, N=N, mu=r, sigma=sigma, use_av=True, seed=None)
        simulator.sim_exact()
        lsmc = LSMC(simulator=simulator, K=K, r=r, payoff_func=european_payoff, option_type='PUT')
        lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, regress_only_itm=False, deg=5)
        lsmc.pathwise_bs_greeks_ad()
        return [s, lsmc.bs_price_ad, lsmc.bs_delta_ad, lsmc.bs_vega_ad]

    out = Parallel(n_jobs=-1)(delayed(calc)(s) for s in tqdm(vec_spot))
    out = np.vstack(out)

    np.savetxt(export_filepath,
               out,
               delimiter=",",
               header="SPOT, PRICE, DELTA, VEGA")

    return out


def gen_LSMC_pathwise_data(t, spot, r, sigma, K, N, export_filepath):
    simulator = GBM(t=t, x0=spot, N=N, mu=r, sigma=sigma, use_av=True, seed=None)
    simulator.sim_exact()
    lsmc = LSMC(simulator=simulator, K=K, r=r, payoff_func=european_payoff, option_type='PUT')
    lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, regress_only_itm=False, deg=5)

    S_tau = np.sum(lsmc.X[1:] * lsmc.opt_stopping_rule, axis=0)
    delta = - S_tau / spot

    out = np.vstack([
        spot * np.ones(shape=(N,)),
        S_tau,
        lsmc.pathwise_opt_stopping_time,
        lsmc.pathwise_opt_stopping_idx,
        delta
    ]).T

    np.savetxt(export_filepath,
               out,
               delimiter=",",
               header="SPOT, S_TAU, TAU, TAU_IDX, DELTA")
    return out


if __name__ == '__main__':
    # Parameters
    t0 = 0.0
    T = 1.0
    x0 = 40
    K = 40
    M = 52
    N = 100000
    r = 0.06
    sigma = 0.2
    size = 8192
    num_std = 5


    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)
    vec_spot = np.random.default_rng().uniform(low=x0*(1-num_std*sigma), high=x0*(1+num_std*sigma), size=size)

    #export_filepath = get_data_path('LSMC_put_with_AD.csv')
    #print(gen_LSMC_data(t=t, vec_spot=vec_spot, r=r, sigma=sigma, K=K, N=N, export_filepath=export_filepath))

    export_filepath = get_data_path('LSMC_pathwise.csv')
    print(gen_LSMC_pathwise_data(t=t, spot=x0, r=r, sigma=sigma, K=K, N=N, export_filepath=export_filepath))
