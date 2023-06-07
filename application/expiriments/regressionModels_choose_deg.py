import numpy as np
from tqdm import tqdm
from application.utils.path_utils import get_data_path
from sklearn.utils import resample
from application.models.regressionModels import DifferentialRegression, make_ridge_cv
import pandas as pd




if __name__ == '__main__':
    # ------------------------------------------------- #
    # Parameter settings                                #
    # ------------------------------------------------- #
    deg = (3,5,7,9)
    N = [256, 512, 1024, 2048, 4096]
    alphas = [0.0, 0.5, 1.0]
    # Number of repetitions
    repeat = 100
    # Size of test data
    sizeTest = 5000



    # ------------------------------------------------- #
    # Load data                                         #
    # ------------------------------------------------- #

    # Load generated, pathwise data
    pathwisePath = get_data_path("LSMC_pathwise_ISD.csv")
    dataPathwise = np.genfromtxt(pathwisePath, delimiter=",", skip_header=0)
    # Load Binomial data for reference
    binomialPath = get_data_path("binomial_unif.csv")
    dataBinomial = np.genfromtxt(binomialPath, delimiter=",", skip_header=0)

    # ------------------------------------------------- #
    # Test data construction                            #
    # ------------------------------------------------- #

    if sizeTest < len(dataBinomial):
        dataBinomial = resample(dataBinomial, n_samples=sizeTest)

    x_test = dataBinomial[:, 0].reshape(-1, 1)
    y_test = dataBinomial[:, 1].reshape(-1, 1)
    z_test = dataBinomial[:, 2].reshape(-1, 1)



    # Objects for storing results
    df = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(repeat), N, alphas, deg], names=['REP', 'N', 'ALPHA', 'DEG']),
        columns=['RMSE_PRICE', 'RMSE_DELTA'],
        dtype=np.float64
    )

    # Perform simulations, calculate prices and errors using LSMC
    for rep in tqdm(range(repeat)):
        for i, n in enumerate(N):
            dataPathwise_tmp = resample(dataPathwise, n_samples=n)

            x_train = dataPathwise_tmp[:, 0].reshape(-1, 1)
            y_train = dataPathwise_tmp[:, 1].reshape(-1, 1)
            z_train = dataPathwise_tmp[:, 2].reshape(-1, 1)
            for j, alpha in enumerate(alphas):
                for k, d in enumerate(deg):
                    # Fit differential regression models for price and delts
                    diffreg = DifferentialRegression(degree=d, alpha=alpha)
                    diffreg.fit(x_train, y_train, z_train)
                    diffpred, z_pred = diffreg.predict(x_test, predict_derivs=True)


                    df.loc[rep, n, alpha, d][['RMSE_PRICE', 'RMSE_DELTA']] = (
                        np.sqrt(np.mean((diffpred - y_test)**2)), np.sqrt(np.mean((z_pred - z_test)**2))
                    )

    # Summary statistics
    df_tmp = pd.concat([
        df.groupby(['N', 'ALPHA', 'DEG']).mean(),
        df.groupby(['N', 'ALPHA', 'DEG']).std()
        ], axis=1)
    df_tmp.columns = ['MEAN_' + c if i < 2 else 'STD_' + c for i, c in enumerate(df_tmp.columns)]


    df_summary_price = pd.DataFrame(index=pd.MultiIndex.from_product([alphas, N]),
                                    columns=['DEG_' + str(p) for p in deg])
    for d in deg:
        for a in alphas:
            for n in N:
                cell = '{:.3f} ({:.4f})'.format(
                    df_tmp.loc[n, a, d]['MEAN_RMSE_PRICE'], df_tmp.loc[n, a, d]['STD_RMSE_PRICE']
                )
                df_summary_price.loc[a, n]['DEG_' + str(d)] = cell

    df_summary_delta = pd.DataFrame(index=pd.MultiIndex.from_product([alphas, N]),
                                    columns=['DEG_' + str(p) for p in deg])
    for d in deg:
        for a in alphas:
            for n in N:
                cell = '{:.3f} ({:.4f})'.format(
                    df_tmp.loc[n, a, d]['MEAN_RMSE_DELTA'], df_tmp.loc[n, a, d]['STD_RMSE_DELTA']
                )
                df_summary_delta.loc[a, n]['DEG_' + str(d)] = cell



    # Export summary results
    # Below provides latex code for the two tables
    print("PRICE table in latex mode: \n")
    print(df_summary_price.to_latex(float_format='%.4f'))

    print("DELTA table in latex mode: \n")
    print(df_summary_delta.to_latex(float_format='%.4f'))

    #df_summary.to_csv(export_path, float_format='%.4f')

    df_summary_price.to_csv('/Users/sebastianhansen/Documents/UNI/PUK/regtablePrice.csv', float_format='%.4f')
    df_summary_delta.to_csv('/Users/sebastianhansen/Documents/UNI/PUK/regtableDelta.csv', float_format='%.4f')















    """
    # ------------------------------------- #
    # Analysis of Letourneau & Stentoft     #
    # ------------------------------------- #

    # RMSE initialization
    mse_price = np.zeros((len(deg_polynomial), num_models), dtype=np.float64)
    mse_delta = np.zeros((len(deg_polynomial), num_models), dtype=np.float64)

    # Create figures
    fig, ax = plt.subplots(nrows=len(deg_polynomial), ncols=2, sharex='col')
    ax[0, 0].set_title('Price')
    ax[0, 1].set_title('Delta')
    ax[len(deg_polynomial) - 1, 0].set_xlabel('Spot')
    ax[len(deg_polynomial) - 1, 1].set_xlabel('Spot')

    for i, degree in tqdm(enumerate(deg_polynomial)):

        # Fit regression models for price and delta
        # Ridge regression
        ridgereg = make_ridge_cv(degree=degree)
        ridgereg.fit(x_train, y_train)
        ridgecoef = ridgereg['ridgecv'].coef_[0] #predict(x_test)
        ridgepred = ridgereg.predict(x_test) # price
        z_pred_ridge = np.polyder(ridgecoef, 1) # delta
        alpha_ridge = ridgereg['ridgecv'].alpha_

        # Differential regression
        # only prices i.e. for alpha = 0
        diffreg_0 = DifferentialRegression(degree=degree, alpha=alpha_differential_regression[0])
        diffreg_0.fit(x_train, y_train, z_train)
        diffpred_0, z_pred_0 = diffreg_0.predict(x_test, predict_derivs=True)

        # half prices half deltas i.e. alpha = 0.5
        diffreg_05 = DifferentialRegression(degree=degree, alpha=alpha_differential_regression[1])
        diffreg_05.fit(x_train, y_train, z_train)
        diffpred_05, z_pred_05 = diffreg_05.predict(x_test, predict_derivs=True)

        # only deltas i.e. alpha = 1
        diffreg_1 = DifferentialRegression(degree=degree, alpha=alpha_differential_regression[2])
        diffreg_1.fit(x_train, y_train, z_train)
        diffpred_1, z_pred_1 = diffreg_1.predict(x_test, predict_derivs=True)

        # Calculate mse for prices and delta
        mse_price[i] = [np.mean((ridgepred - y_test)**2), np.mean((diffpred_0 - y_test)**2), np.mean((diffpred_05 - y_test)**2), np.mean((diffpred_1 - y_test)**2)]
        mse_delta[i] = [np.mean((z_pred_ridge - z_test) ** 2), np.mean((z_pred_0 - z_test) ** 2), np.mean((z_pred_05 - z_test) ** 2), np.mean((z_pred_1 - z_test) ** 2)]

        # Add subplot for price
        ax[i, 0].scatter(x_train, y_train, marker='x', color='cyan', s=2, alpha=0.5, label='train')
        ax[i, 0].scatter(x_test, y_test, marker='o', color='red', s=2, alpha=0.5, label="true")
        ax[i, 0].scatter(x_test, ridgepred, marker='o', color='pink', s=2, alpha=0.5,
                         label='diff. reg. α={:.2f}'.format(alpha_ridge) + ' RMSE={:.2E}'.format(mse_price[i][0]))
        ax[i, 0].scatter(x_test, diffpred_0, marker='o', color='blue', s=2, alpha=0.5,
                    label='diff. reg. α={:.2f}' .format(alpha_differential_regression[0]) + ' RMSE={:.2E}'.format(mse_price[i][1]))
        ax[i, 0].scatter(x_test, diffpred_05, marker='o', color='purple', s=2, alpha=0.5,
                    label='diff. reg. α={:.2f}' .format(alpha_differential_regression[1]) + ' RMSE={:.2E}'.format(mse_price[i][2]))
        ax[i, 0].scatter(x_test, diffpred_1, marker='o', color='orange', s=2, alpha=0.5,
                    label='diff. reg. α={:.2f}' .format(alpha_differential_regression[2]) + ' RMSE={:.2E}'.format(mse_price[i][3]))


        # add subplots for deltas
        ax[i, 1].scatter(x_train, z_train, marker='o', color='cyan', s=2, alpha=0.5, label='train')
        ax[i, 1].scatter(x_test, z_test, marker='o', color='red', s=2, alpha=0.5, label="true")
        ax[i, 1].scatter(x_test, z_pred_0, marker='o', color='pink', s=2, alpha=0.5,
                         label='diff. reg. α={:.2f}'.format(alpha_ridge) + ' RMSE={:.2E}'.format(mse_delta[i][0]))
        ax[i, 1].scatter(x_test, z_pred_0, marker='o', color='blue', s=2, alpha=0.5,
                    label='diff. reg. α={:.2f}'.format(alpha_differential_regression[0]) + ' RMSE={:.2E}'.format(mse_delta[i][1]))
        ax[i, 1].scatter(x_test, z_pred_05, marker='o', color='purple', s=2, alpha=0.5,
                    label='diff. reg. α={:.2f}'.format(alpha_differential_regression[1]) + ' RMSE={:.2E}'.format(mse_delta[i][2]))
        ax[i, 1].scatter(x_test, z_pred_1, marker='o', color='orange', s=2, alpha=0.5,
                    label='diff. reg. α={:.2f}'.format(alpha_differential_regression[2]) + ' RMSE={:.2E}'.format(mse_delta[i][3]))

        # formatting
        ax[i, 0].set_ylabel('deg={}'.format(degree))
        ax[i, 0].legend()
        ax[i, 1].legend()
        # ax[i, 0].set_xlim(0.8 * np.min(x_test), 1.2 * np.max(x_test))
        # ax[i, 0].set_ylim(-0.2, 1.2 * np.max(binom_price))
        # ax[i, 0].text(60, (-0.2 + 1.2 * np.max(binom_price)) / 2, 'MSE = {:.2E}'.format(mse_price[i]))

    plt.show()

        """


