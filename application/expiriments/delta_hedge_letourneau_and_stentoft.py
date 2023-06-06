import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from application.binomial_model.binomial_model import binomial_tree, binomial_tree_bs
from application.models.LetourneauStentoft import ISD
from application.simulation.sim_gbm import GBM
from application.Longstaff_Schwartz.LSMC import LSMC
from application.utils.LSMC_fit_predict import fit_poly, pred_poly
from application.options.payoff import european_payoff


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


if __name__ == '__main__':
    # Fixed parameters
    t0 = 0.0
    T = 0.25
    x0 = 40.0
    M = 50
    r = 0.06
    sigma = 0.2
    K = 40.0
    seed = 1234
    deg_lsmc = 9
    deg_stentoft = 9
    option_type = 'PUT'
    alpha = 25.0

    rep = 1

    # Variables to vary
    N = 100000

    # Auxiliary variables
    t = np.linspace(start=t0, stop=T, num=M+1, endpoint=True)
    dt = T/M

    # Calculate binomial price
    M_binom = 5000
    eur_amr = 'AMR'

    u = np.exp(sigma * np.sqrt(T / M_binom))
    d = np.exp(-sigma * np.sqrt(T / M_binom))

    binom_price, binom_delta, binom_eeb = binomial_tree(K, T, x0, r, M_binom, u, d, european_payoff,
                                                        option_type=option_type, eur_amr=eur_amr)
    binom_eeb[np.isnan(binom_eeb)] = np.nanmin(binom_eeb)

    eeb = binom_eeb[[int(M_binom / T * s) for s in t]]

    plt.plot(np.linspace(0, T, M_binom+1, True), binom_eeb, color='red')
    plt.plot(t, eeb, color='blue')
    plt.hlines(K, xmin=0.0, xmax=T, colors='black', linestyles='dashed')
    plt.title('Early Exercise Boundary')
    plt.show()

    # Initialize experiment

    # Actual realization of stock prices
    gbm = GBM(t=t, x0=x0, N=N, mu=r, sigma=sigma, use_av=True, seed=seed)
    gbm.sim_exact()
    S = gbm.X

    a = np.full_like(S, np.nan)  # Position in the stock
    b = np.full_like(S, np.nan)  # Position in the risk-free asset
    V = np.full_like(S, np.nan)  # Value-process of our hedge portfolio (excluding what we sold)

    # Find initial hedge
    # TODO SHOULD N be something different from the number of paths?
    x_isd = ISD(N=N, x0=x0, alpha=alpha, seed=seed)
    gbm_isd = GBM(t=t, x0=x_isd, N=N, mu=r, sigma=sigma, use_av=True, seed=seed)
    gbm_isd.sim_exact()
    lsmc = LSMC(simulator=gbm_isd, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
    lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

    a[0, :] = letourneau_fit_predict(lsmc=lsmc, x=S[0, :], x0=K, deg_stentoft=deg_stentoft)[1]
    b[0, :] = binom_price - a[0, :] * S[0, :]
    V[0, :] = a[0, :] * S[0, :] + b[0, :]

    alive = S[0, :] > eeb[0]
    exercise = np.full_like(S, False, dtype=bool)

    pnl = np.full(N, np.nan)

    # TODO OBS NOTE x0=K

    for j, s in tqdm(enumerate(t[1:], start=1)):

        # Find paths to exercise and update alive status
        exercise[j, alive] = S[j, alive] < eeb[j]
        alive = np.minimum(alive, ~exercise[j, :])

        # Perform simulations necessary for determining delta
        x_isd = ISD(N=N, x0=K, alpha=alpha)
        gbm_isd = GBM(t=t[j:], x0=x_isd, N=N, mu=r, sigma=sigma, use_av=True)
        gbm_isd.sim_exact()
        lsmc = LSMC(simulator=gbm_isd, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
        lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

        # Update positions
        V[j, :] = a[j - 1, :] * S[j, :] + b[j - 1, :] * np.exp(dt * r)
        #a[j, :] = np.maximum(np.minimum(
        #    letourneau_fit_predict(lsmc=lsmc, x=S[j, :], x0=K, deg_stentoft=deg_stentoft)[1],
        #    0.0), -1.0)
        a[j, :] = letourneau_fit_predict(lsmc=lsmc, x=S[j, :], x0=K, deg_stentoft=deg_stentoft)[1]
        b[j, :] = V[j, :] - a[j, :] * S[j, :]

        # Calculate (present value) PnL of exercised positions
        pnl[exercise[j, :]] = np.exp(-r*s) * (
                V[j, exercise[j, :]] - european_payoff(S[j, exercise[j, :]], K=K, option_type=option_type)
        )


    # PnL of hedges against unexercised, expired options is just the value of the hedge position
    pnl[np.isnan(pnl)] = np.exp(-r*T) * V[-1, np.isnan(pnl)]

    print("mean = {:.6f}, std = {:.4f}".format(np.nanmean(pnl), np.nanstd(pnl)))

    not_exercised = np.sum(exercise, axis=0) == 0
    plt.scatter(S[exercise], V[exercise], alpha=0.05, color='blue')
    plt.scatter(S[-1, not_exercised], V[-1, not_exercised], alpha=0.05, color='red')
    plt.show()

    not_exercised = np.sum(exercise, axis=0) == 0
    plt.scatter(S[exercise], pnl[np.sum(exercise, axis=0) > 0], alpha=0.05, color='blue')
    plt.scatter(S[-1, not_exercised], pnl[not_exercised], alpha=0.05, color='red')
    plt.show()

    # Compare deltas
    N = 100000
    t = np.linspace(start=t0, stop=T, num=M+1, endpoint=True)
    gbm = GBM(t=t, x0=x0, N=1, mu=r, sigma=sigma, use_av=False, seed=5)
    gbm.sim_exact()
    S = gbm.X[:, 0]

    x_isd = ISD(N=N, x0=x0, alpha=alpha)
    gbm_isd = GBM(t=t, x0=x_isd, N=N, mu=r, sigma=sigma, use_av=True)
    gbm_isd.sim_exact()
    lsmc = LSMC(simulator=gbm_isd, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
    lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

    a = np.full_like(t, np.nan)
    b = np.full_like(t, np.nan)
    V = np.full_like(t, np.nan)

    a_binom = np.full_like(t, np.nan)
    b_binom = np.full_like(t, np.nan)
    V_binom = np.full_like(t, np.nan)

    a[0] = letourneau_fit_predict(lsmc=lsmc, x=S[0], x0=x0, deg_stentoft=deg_stentoft)[1]
    b[0] = binom_price - a[0] * S[0]
    V[0] = b[0] + a[0] * S[0]

    binom = binomial_tree_bs(K=K, T=T, S0=S[0], r=r, sigma=sigma, M=5000, payoff_func=european_payoff, eur_amr='AMR')
    a_binom[0] = binom[1]
    b_binom[0] = binom_price - a_binom[0] * S[0]
    V_binom[0] = b[0] + a[0] * S[0]

    price_binom = np.full_like(t, np.nan)
    price_binom[0] = binom[0]

    alive = True
    exercise = np.full_like(t, False, dtype=bool)

    for j, s in tqdm(enumerate(t[1:], start=1)):
        exercise[j] = (S[j] < eeb[j]) if alive else False
        alive = np.minimum(alive, ~exercise[j])

        # LS hedges
        x_isd = ISD(N=N, x0=K, alpha=alpha)
        gbm_isd = GBM(t=t[j:], x0=x_isd, N=N, mu=r, sigma=sigma, use_av=True)
        gbm_isd.sim_exact()
        lsmc = LSMC(simulator=gbm_isd, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
        lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

        V[j] = a[j-1] * S[j] + b[j-1] * np.exp(-r*dt)
        # TODO use x0=K or x0=S[j]
        a[j] = letourneau_fit_predict(lsmc=lsmc, x=S[j], x0=S[j], deg_stentoft=deg_stentoft)[1] if alive else np.nan
        b[j] = V[j] - a[j] * S[j]

        # Binomial models hedges
        V_binom[j] = a_binom[j-1] * S[j] + b_binom[j-1] * np.exp(-r*dt)
        if s != T:
            binom = binomial_tree_bs(K=K, T=T-s, S0=S[j], r=r, sigma=sigma, M=5000,
                                     payoff_func=european_payoff, eur_amr='AMR')
            price_binom[j] = binom[0]
            a_binom[j] = binom[1] if alive else np.nan
        else:
            price_binom[j] = european_payoff(x=S[j], K=K, option_type=option_type)
            a_binom[j] = (-1.0 if S[j] <= K else 0.0) if alive else 0.0
        b_binom[j] = V_binom[j] - a_binom[j] * S[j]

    tau = t[exercise]
    pnl = np.exp(-r*tau) * (V[exercise] - european_payoff(x=S[exercise], K=K, option_type=option_type))
    pnl_binom = np.exp(-r*tau) * (V_binom[exercise] - european_payoff(x=S[exercise], K=K, option_type=option_type))

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex='all')
    ax[0].plot(t, S, color='black')
    ax[0].plot(t, eeb, color='yellow')
    ax[0].scatter(tau, S[exercise], color='black', marker='x')
    ax[0].vlines(tau, np.min(S), S[exercise], color='black', linestyles='dashed')

    ax[1].plot(t, a, color='blue')
    ax[1].scatter(tau-dt, a[~np.isnan(a)][-1], color='blue')
    ax[1].plot(t, a_binom, color='red')
    ax[1].scatter(tau-dt, a_binom[np.sum(~np.isnan(a))-1], color='red')

    ax[2].plot(t, V - price_binom, color='blue')
    ax[2].scatter(tau, (V - price_binom)[exercise], color='blue')
    ax[2].plot(t, V_binom - price_binom, color='red')
    ax[2].scatter(tau, (V_binom - price_binom)[exercise], color='red')

    ax[0].set_ylabel('S(t)')
    ax[1].set_ylabel('α')
    ax[2].set_ylabel('Hedge Error')
    ax[2].set_xlabel('t')
    plt.show()

















