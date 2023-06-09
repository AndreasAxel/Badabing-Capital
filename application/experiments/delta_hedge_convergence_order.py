import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from application.binomial_model.binomial_model import binomial_tree_bs
from application.models.LetourneauStentoft import ISD
from application.models.regressionModels import DifferentialRegression
from application.simulation.sim_gbm import GBM
from application.Longstaff_Schwartz.LSMC import LSMC
from application.utils.LSMC_fit_predict import fit_poly, pred_poly
from application.options.payoff import european_payoff
from application.models.neural_approximator import Neural_approximator


def simulate_pathwise_data(t, N, r, sigma, K, option_type):
    rng = np.random.default_rng()
    Z = rng.normal(loc=0.0, scale=1.0, size=N)

    spot = K * np.exp((r - 0.5 * sigma ** 2) * t[-1] + np.sqrt(t[-1]) * sigma * Z)

    simulator = GBM(t=t, x0=spot, N=N, mu=r, sigma=sigma, use_av=True, seed=None)
    simulator.sim_exact()
    lsmc = LSMC(simulator=simulator, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
    lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=9)

    tau = np.array([t if ~np.isnan(t) else 1.0 for t in lsmc.pathwise_opt_stopping_time])
    df = np.exp(-r * tau)

    one = ~np.isnan(lsmc.pathwise_opt_stopping_time)
    S_tau = np.sum(lsmc.X * lsmc.opt_stopping_rule, axis=0)
    payoff = df * np.sum(lsmc.payoff * lsmc.opt_stopping_rule, axis=0)
    delta = - df * S_tau / spot * one

    return spot.reshape(-1, 1), payoff.reshape(-1, 1), delta.reshape(-1, 1)


def letourneau_fit_predict(lsmc, x, x0, deg_stentoft=9):
    # Extract (pathwise) payoffs
    cf = lsmc.payoff
    cf = np.sum((cf * lsmc.opt_stopping_rule), axis=0)

    # Calculate discount factor
    df = [np.exp(-lsmc.r * tau) if ~np.isnan(tau) else 0.0 for tau in lsmc.pathwise_opt_stopping_time]

    # Calculate (pathwise) discounted cashflows
    cf_pv = cf * df

    # Fit polynomials for price and delta
    coef_price = fit_poly(x=lsmc.X[0, :] - x0, y=cf_pv, deg=deg_stentoft)
    coef_delta = np.polyder(coef_price, 1)

    # Predict price and delta
    price = pred_poly(x=x - x0, fit=coef_price)
    delta = pred_poly(x=x - x0, fit=coef_delta)

    return price, delta


def diff_reg_fit_predict(x, t, N_train, r, sigma, K, option_type, deg=9, alpha=0.5):
    # Generate pathwise samples
    x_train, y_train, z_train = simulate_pathwise_data(t, N_train, r, sigma, K, option_type)

    diff_reg = DifferentialRegression(degree=deg, alpha=alpha)
    diff_reg.fit(x_train, y_train, z_train)
    price, delta = diff_reg.predict(x.reshape(-1, 1), predict_derivs=True)

    return price.reshape(len(x), ), delta.reshape(len(x), )


def nn_fit_predict(x, t, N_train, r, sigma, K, option_type, epochs=100):
    # generate pathwise training data from LSMC
    x_train, y_train, z_train = simulate_pathwise_data(t, N_train, r, sigma, K, option_type)
    # Initialize regressor
    regressor = Neural_approximator(x_raw=x_train, y_raw=y_train, dydx_raw=z_train)
    # Prepare network, True for differentials
    regressor.prepare(N_train, differential=True, weight_seed=None)
    # Train network
    regressor.train('differential', epochs=epochs)
    # Predict on spot
    predictions, deltas = regressor.predict_values_and_derivs(x)

    return predictions[:, 0], deltas[:, 0]


if __name__ == '__main__':
    # Fixed parameters
    t0 = 0.0
    T = 0.25
    x0 = 40.0
    K = 40.0
    M = 50
    N = 10000
    r = 0.06
    sigma = 0.2

    seed = 1234
    deg_lsmc = 9
    deg_stentoft = 9
    option_type = 'PUT'
    eur_amr = 'AMR'
    alpha_isd = 25.0
    alpha = 0.5         # Regularization

    delta_lb = -1.0  # Set this to -np.inf if unbounded
    delta_ub = 0.0   # Set this to np.inf if unbounded

    # Variables to vary
    N_train = np.array([128*2**i for i in range(10)])

    # Array for storing hedge errors
    hedge_err_std = np.full((len(N_train), 3), np.nan)

    # Auxiliary variables
    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)
    dt = T / M

    # Simulate stock paths
    gbm = GBM(t=t, x0=x0, N=N, mu=r, sigma=sigma, use_av=True, seed=seed)
    gbm.sim_exact()
    S = gbm.X

    # Setup Early Exercise Boundary
    binom = binomial_tree_bs(K=K, T=T, S0=x0, r=r, sigma=sigma,
                             M=5000, payoff_func=european_payoff, option_type=option_type, eur_amr=eur_amr)
    binom_eeb = binom[2]
    binom_eeb[np.isnan(binom_eeb)] = np.nanmin(binom_eeb)
    eeb = binom_eeb[[int(5000 / T * s) for s in t]]

    for k, n in tqdm(enumerate(N_train)):
        # ----------------------------------------- #
        # Initialize experiment                     #
        # ----------------------------------------- #

        alive = S[0, :] > eeb[0]
        exercise = np.full_like(S, False, dtype=bool)

        # Letourneau & Stentoft
        x_isd = ISD(N=n, x0=x0, alpha=alpha_isd, seed=seed)
        gbm_isd = GBM(t=t, x0=x_isd, N=n, mu=r, sigma=sigma, use_av=True, seed=seed)
        gbm_isd.sim_exact()
        lsmc = LSMC(simulator=gbm_isd, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
        lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

        a_ls = np.full_like(S, np.nan)
        b_ls = np.full_like(S, np.nan)
        V_ls = np.full_like(S, np.nan)

        a_ls[0, :] = np.minimum(delta_ub, np.maximum(
            delta_lb,letourneau_fit_predict(lsmc=lsmc, x=S[0, :], x0=K, deg_stentoft=deg_stentoft)[1]
        ))
        b_ls[0, :] = binom[0] - a_ls[0, :] * S[0, :]
        V_ls[0, :] = b_ls[0, :] + a_ls[0, :] * S[0, :]

        # Differential regression
        a_diff_reg = np.full_like(S, np.nan)
        b_diff_reg = np.full_like(S, np.nan)
        V_diff_reg = np.full_like(S, np.nan)

        a_diff_reg[0, :] = np.minimum(delta_ub, np.maximum(
            delta_lb, diff_reg_fit_predict(x=np.array([x0]), t=t, N_train=n, r=r, sigma=sigma, K=K, option_type=option_type)[1]
        ))
        b_diff_reg[0, :] = binom[0] - a_diff_reg[0, :] * S[0, :]
        V_diff_reg[0, :] = b_diff_reg[0, :] + a_diff_reg[0, :] * S[0, :]

        # Differential Neural Network
        a_nn = np.full_like(S, np.nan)
        b_nn = np.full_like(S, np.nan)
        V_nn = np.full_like(S, np.nan)

        # Find initial hedge
        a_nn[0, :] = np.minimum(delta_ub, np.maximum(
            delta_lb,
            nn_fit_predict(x=np.array(x0).reshape(-1, 1), t=t, N_train=n, r=r, sigma=sigma, K=K,
                           option_type=option_type)[1]
        ))
        b_nn[0, :] = binom[0] - a_nn[0, :] * S[0, :]
        V_nn[0, :] = b_nn[0, :] + a_nn[0, :] * S[0, :]

        # ----------------------------------------- #
        # Dynamics hedging experiment               #
        # ----------------------------------------- #

        for j, s in enumerate(t[1:], start=1):

            # Letourneau & Stentoft
            x_isd = ISD(N=n, x0=K, alpha=alpha_isd)
            gbm_isd = GBM(t=t[j:], x0=x_isd, N=n, mu=r, sigma=sigma, use_av=True)
            gbm_isd.sim_exact()

            lsmc = LSMC(simulator=gbm_isd, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
            lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

            V_ls[j, :] = a_ls[j - 1, :] * S[j, :] + b_ls[j - 1, :] * np.exp(dt * r)
            a_ls[j, :] = letourneau_fit_predict(lsmc=lsmc, x=S[j, :], x0=K, deg_stentoft=deg_stentoft)[1]
            a_ls[j, :] = np.array([a if alive[i] else 0.0 for i, a in enumerate(a_ls[j, :])])
            b_ls[j, :] = V_ls[j, :] - a_ls[j, :] * S[j, :]

            # Differential regression
            V_diff_reg[j, :] = a_diff_reg[j - 1, :] * S[j, :] + b_diff_reg[j - 1, :] * np.exp(dt * r)
            a_diff_reg[j, :] = np.minimum(delta_ub, np.maximum(
                delta_lb, diff_reg_fit_predict(x=S[j, :], t=t[j:], N_train=n, r=r, sigma=sigma, K=K, option_type=option_type)[1]
            ))
            a_diff_reg[j, :] = np.array([a if alive[i] else 0.0 for i, a in enumerate(a_diff_reg[j, :])])
            b_diff_reg[j, :] = V_diff_reg[j, :] - a_diff_reg[j, :] * S[j, :]

            # Differential Neural Network
            V_nn[j, :] = a_nn[j - 1, :] * S[j, :] + b_nn[j - 1, :] * np.exp(dt * r)
            a_nn[j, :] = np.minimum(delta_ub, np.maximum(
                delta_lb, nn_fit_predict(x=S[j, :].reshape(-1 ,1), t=t[j:], N_train=n, r=r, sigma=sigma, K=K, option_type=option_type)[1]
            ))
            a_nn[j, :] = np.array([a if alive[i] else 0.0 for i, a in enumerate(a_nn[j, :])])
            b_nn[j, :] = V_nn[j, :] - a_nn[j, :] * S[j, :]

            # Option holder makes exercise decision
            exercise[j] = np.minimum((S[j] < eeb[j]), alive)
            alive = np.minimum(alive, ~exercise[j])

        # Extract stopping times (paths not exercised is set to expire at maturity)
        tau_idx = np.argmax(exercise, 0)
        tau_idx = np.array([j if j > 0 else M for j in tau_idx])

        # Discount factor for PnL
        df = np.array([np.exp(-r * t[j]) for j in tau_idx])

        # ----------------------------------------- #
        # Calculate PnL (Hedge Error)               #
        # ----------------------------------------- #

        # Spot at stopping times and payoff on the sold put option
        x = np.array([S[tau_idx[i], i] for i in range(N)])
        p = np.array([np.max([K - S[tau_idx[i], i], 0.0]) for i in range(N)])

        # Value of hedge portfolios
        v_ls = np.array([V_ls[tau_idx[i], i] for i in range(N)])
        v_diff_reg = np.array([V_diff_reg[tau_idx[i], i] for i in range(N)])
        v_nn = np.array([V_nn[tau_idx[i], i] for i in range(N)])

        # PnL (hedge error)
        pnl_ls = df * (v_ls - p)
        pnl_diff_reg = df * (v_diff_reg - p)
        pnl_nn = df * (v_nn - p)

        # Update results
        hedge_err_std[k, 0] = np.std(pnl_ls)
        hedge_err_std[k, 1] = np.std(pnl_diff_reg)
        hedge_err_std[k, 2] = np.std(pnl_nn)

    # ----------------------------------------- #
    # Plot results                              #
    # ----------------------------------------- #

    color = ['blue', 'red', 'black']
    label = ['LS Naive Method', 'Diff. Reg.', 'Diff. NN']

    for i in range(3):
        plt.plot(N_train, hedge_err_std[:, i], color=color[i], label=label[i], marker='o')
    plt.semilogx(base=2)
    plt.xticks(ticks=N_train, labels=N_train)
    plt.legend()
    plt.ylim(-0.1, np.max(hedge_err_std)*1.2)
    plt.ylabel('STD( hedge error )')
    plt.xlabel('Training Samples')
    plt.show()

