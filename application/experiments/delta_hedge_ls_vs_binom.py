import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from application.binomial_model.binomial_model import binomial_tree_bs
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
    eur_amr = 'AMR'
    alpha = 25.0

    N = 100000
    M_hedge = 500

    # Auxiliary variables
    t = np.linspace(start=t0, stop=T, num=M+1, endpoint=True)
    dt = T/M

    # Simulate stock paths
    gbm = GBM(t=t, x0=x0, N=1, mu=r, sigma=sigma, use_av=False, seed=5)
    gbm.sim_exact()
    S = gbm.X[:, 0]

    # Setup Early Exercise Boundary
    binom = binomial_tree_bs(K=K, T=T, S0=x0, r=r, sigma=sigma,
                             M=5000, payoff_func=european_payoff, option_type=option_type, eur_amr=eur_amr)
    binom_eeb = binom[2]
    binom_eeb[np.isnan(binom_eeb)] = np.nanmin(binom_eeb)
    eeb = binom_eeb[[int(5000 / T * s) for s in t]]

    plt.plot(np.linspace(start=0.0, stop=T, num=5000+1, endpoint=True), binom_eeb, color='red', label='EEB (M=5000)')
    plt.plot(t, eeb, color='black', label='EEB (M={})'.format(M))
    plt.plot(t, S, color='blue', alpha=0.5)
    plt.title('Stock paths & Early Exercise Boundary')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('')
    plt.show()

    # Initialize experiment
    a = np.full_like(t, np.nan)
    b = np.full_like(t, np.nan)
    V = np.full_like(t, np.nan)

    a_binom = np.full_like(t, np.nan)
    b_binom = np.full_like(t, np.nan)
    V_binom = np.full_like(t, np.nan)

    x_isd = ISD(N=N, x0=x0, alpha=alpha)
    gbm_isd = GBM(t=t, x0=x_isd, N=N, mu=r, sigma=sigma, use_av=True)
    gbm_isd.sim_exact()
    lsmc = LSMC(simulator=gbm_isd, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
    lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

    a[0] = letourneau_fit_predict(lsmc=lsmc, x=S[0], x0=x0, deg_stentoft=deg_stentoft)[1]
    b[0] = binom[0] - a[0] * S[0]
    V[0] = b[0] + a[0] * S[0]

    binom = binomial_tree_bs(K=K, T=T, S0=S[0], r=r, sigma=sigma, M=M_hedge, payoff_func=european_payoff, eur_amr='AMR')
    a_binom[0] = binom[1]
    b_binom[0] = binom[0] - a_binom[0] * S[0]
    V_binom[0] = b_binom[0] + a_binom[0] * S[0]

    price_binom = np.full_like(t, np.nan)
    price_binom[0] = binom[0]

    alive = True
    exercise = np.full_like(t, False, dtype=bool)

    for j, s in tqdm(enumerate(t[1:], start=1)):
        # LS hedges
        x_isd = ISD(N=N, x0=K, alpha=alpha)
        gbm_isd = GBM(t=t[j:], x0=x_isd, N=N, mu=r, sigma=sigma, use_av=True)
        gbm_isd.sim_exact()
        lsmc = LSMC(simulator=gbm_isd, K=K, r=r, payoff_func=european_payoff, option_type=option_type)
        lsmc.run_backwards(fit_func=fit_poly, pred_func=pred_poly, deg=deg_lsmc)

        V[j] = a[j-1] * S[j] + b[j-1] * np.exp(-r*dt)
        a[j] = letourneau_fit_predict(lsmc=lsmc, x=S[j], x0=K, deg_stentoft=deg_stentoft)[1] if alive else 0.0
        b[j] = V[j] - a[j] * S[j]

        # Binom hedge
        if s != T:
            binom = binomial_tree_bs(K=K, T=T - s, S0=S[j], r=r, sigma=sigma, M=M_hedge,
                                     payoff_func=european_payoff, eur_amr='AMR')
            price_binom[j] = binom[0]
            a_binom[j] = binom[1] if alive else 0.0
        else:
            price_binom[j] = european_payoff(x=S[j], K=K, option_type=option_type)
            a_binom[j] = (-1.0 if S[j] <= K else 0.0) if alive else 0.0

        V_binom[j] = a_binom[j-1] * S[j] + b_binom[j-1] * np.exp(-r*dt)
        b_binom[j] = V_binom[j] - a_binom[j] * S[j]

        # Exercise decision
        exercise[j] = np.minimum((S[j] < eeb[j]), alive)
        alive = np.minimum(alive, ~exercise[j])

    # Extract stopping times (paths not exercised is set to expiry)
    tau_idx = np.argmax(exercise, 0)
    tau_idx = tau_idx if tau_idx > 0 else M
    tau = t[tau_idx]

    x = S[tau_idx]
    v = V[tau_idx]
    v_binom = V_binom[tau_idx]
    p = european_payoff(x=x, K=K, option_type=option_type)

    plt.scatter(x, v, color='blue', label='Hedge LS (V)')
    plt.scatter(x, v_binom, color='red', label='Hedge binom (V_binom)')
    plt.scatter(x, p, color='black', label='Put (p)')
    plt.legend()
    plt.xlabel('S(tau)')
    plt.show()

    # Discount factor for PnL
    df = np.exp(-r * tau)

    # Calculate present value of PnL for each path
    pnl = df * (v - p)
    pnl_binom = df * (v_binom - p)

    print('mean={:.4f}, std={:.4f}, rmse={:.4f}'.format(np.mean(pnl), np.std(pnl), np.sqrt(np.mean(pnl**2))))
    print('mean={:.4f}, std={:.4f}, rmse={:.4f}'.format(np.mean(pnl_binom), np.std(pnl_binom), np.sqrt(np.mean(pnl_binom ** 2))))

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

