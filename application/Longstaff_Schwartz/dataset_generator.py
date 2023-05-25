from joblib import Parallel, delayed
from tqdm import tqdm
from application.utils.path_utils import get_data_path
from application.models.LetourneauStentoft import *
from application.binomial_model.binomial_model import *
from application.utils.LSMC_fit_predict import *


def gen_LSMC_pathwise_data(t, spot, r, sigma, K, N, export_filepath):
    simulator = GBM(t=t, x0=spot, N=N, mu=r, sigma=sigma, use_av=True, seed=None)
    simulator.sim_exact()
    lsmc = LSMC(simulator=simulator, K=K, r=r, payoff_func=european_payoff, option_type='PUT')
    lsmc.run_backwards(fit_func=fit_laguerre_poly, pred_func=pred_laguerre_poly, deg=5)

    tau = np.array([t if ~np.isnan(t) else 1.0 for t in lsmc.pathwise_opt_stopping_time])
    df = np.exp(-r * tau)

    one = ~np.isnan(lsmc.pathwise_opt_stopping_time)
    S_tau = np.sum(lsmc.X * lsmc.opt_stopping_rule, axis=0)
    payoff = df * np.sum(lsmc.payoff * lsmc.opt_stopping_rule, axis=0)
    delta = - df * S_tau / spot * one

    out = np.vstack([
        spot * np.ones(shape=(N,)),
        payoff,
        delta
    ]).T

    np.savetxt(export_filepath,
               out,
               delimiter=",",
               header="SPOT, PAYOFF, DELTA")
    return out


def gen_Letourneau_data(spot, fitted, N, export_filepath):
    x0 = fitted[0],
    priceFit = fitted[1]
    deltaFit = fitted[2]
    gammaFit = fitted[3]

    payoff, delta, gamma = Letourneau(spot, x0, priceFit, deltaFit, gammaFit)

    out = np.vstack([
        spot * np.ones(shape=(N,)),
        payoff,
        delta,
        gamma
    ]).T

    np.savetxt(export_filepath,
               out,
               delimiter=",",
               header="SPOT, PRICE, DELTA, GAMMA")
    return out


def gen_binomial(vec_spot, K, T, r, sigma, M, export_filepath):

    def calc(s):
        p, d, _ = binomial_tree_bs(K=K, T=T, S0=s, r=r, sigma=sigma, M=M, payoff_func=european_payoff,
                                   option_type='PUT', eur_amr='AMR')
        return [s, p, d]

    out = Parallel(n_jobs=-1)(delayed(calc)(s) for s in tqdm(vec_spot))
    out = np.vstack(out)

    np.savetxt(export_filepath,
               out,
               delimiter=",",
               header="SPOT, PRICE, DELTA")
    return out


if __name__ == '__main__':
    # Parameters
    t0 = 0.0
    T = 1.0
    x0 = 40
    K = 40
    M = 252
    N = 100000
    r = 0.06
    sigma = 0.2
    seed = 1234

    alpha = 25

    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)
    x_linear = np.linspace(20, 60, N, True)
    x_isd = ISD(N=N, x0=x0, alpha=alpha, seed=seed)

    # ------------------------------------------------- #
    # LSMC (pathwise)                                   #
    # ------------------------------------------------- #

    export_filepath = get_data_path('LSMC_pathwise.csv')
    data_lsmc = gen_LSMC_pathwise_data(t=t, spot=x_linear, r=r, sigma=sigma, K=K, N=N, export_filepath=export_filepath)

    # Plot simulated pathwise deltas
    #import matplotlib.pyplot as plt
    #spot = data_lsmc[:, 0]
    #pathwise_delta = data_lsmc[:, 2]
    #plt.scatter(spot, pathwise_delta, alpha=0.05)
    #plt.show()

    # ------------------------------------------------- #
    # Binomial tree                                     #
    # ------------------------------------------------- #

    #export_filepath = get_data_path('binomial.csv')
    #gen_binomial(vec_spot=x_linear, K=K, T=T, r=r, sigma=sigma, M=2500, export_filepath=export_filepath)

    # ------------------------------------------------- #
    # Letourneau & Stentoft                             #
    # ------------------------------------------------- #

    letourneauExport = False

    if letourneauExport:
        deg_lsmc = 9
        deg_stentoft = 9
        option_type = 'PUT'
        fitted = disperseFit(t0=t0,
                             T=T,
                             x0=x0,
                             N=N,
                             M=M,
                             r=r,
                             sigma=sigma,
                             K=K,
                             seed=None,
                             deg_lsmc=deg_lsmc,
                             deg_stentoft=deg_stentoft,
                             option_type=option_type,
                             x_isd=x_isd)
        export_filepath = get_data_path('letourneauStentoft_data.csv')
        gen_Letourneau_data(spot=x_isd, fitted=fitted, N=N, export_filepath=export_filepath)

    """
    import matplotlib.pyplot as plt
    X = np.loadtxt(export_filepath, delimiter=',')
    spot = X[:, 0]
    price = X[:, 1]
    delta = X[:, 2]
    vega = X[:, 3]

    fig, ax = plt.subplots(nrows=3, sharex='all')
    ax[0].scatter(spot, price, alpha=0.5, color='black')
    ax[1].scatter(spot, delta, alpha=0.5, color='blue')
    ax[2].scatter(spot, vega, alpha=0.5, color='green')
    ax[0].set_ylabel('Price')
    ax[1].set_ylabel('Delta')
    ax[2].set_ylabel('Vega')
    plt.show()

    #export_filepath = get_data_path('LSMC_pathwise_v2.csv')
    #print(gen_LSMC_pathwise_data(t=t, spot=x0, r=r, sigma=sigma, K=K, N=N, export_filepath=export_filepath))
    """
