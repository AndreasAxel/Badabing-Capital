
import numpy as np
from tqdm import tqdm
from application.utils.path_utils import get_data_path
from sklearn.utils import resample
from application.models.regressionModels import DifferentialRegression, make_ridge_cv
import pandas as pd
from application.models.neural_approximator import *





if __name__ == '__main__':
    # ------------------------------------------------- #
    # Parameter settings                                #
    # ------------------------------------------------- #
    #num_models = 4  # number of considered models i.e. ridge regression and diff regressions for alpha = {0, 0.5, 1}
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
    # Data manipulation                                 #
    # ------------------------------------------------- #

    if sizeTest < len(dataBinomial):
        dataBinomial = resample(dataBinomial, n_samples=sizeTest)

    x_test = dataBinomial[:, 0].reshape(-1, 1)
    y_test = dataBinomial[:, 1].reshape(-1, 1)
    z_test = dataBinomial[:, 2].reshape(-1, 1)

    # Degrees to vary
    deg = ['standard', 'differential'] #methods
    N = [256*4**i for i in range(5)]
    epochs = [10, 25, 50, 100, 200] #epochs

    # ------------------------------------- #
    # Analysis of Regression models         #
    # ------------------------------------- #

    # Number of repetitions
    repeat = 25

    # Objects for storing results
    df = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(repeat), N, epochs, deg], names=['REP', 'N', 'EPOCH', 'DEG']),
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
            # Learn neural approximation
            regressor = Neural_approximator(x_raw=x_train, y_raw=y_train, dydx_raw=z_train)

            for j, epoch in enumerate(epochs):
                for k, d in enumerate(deg):

                    # standard network
                    if d == "standard":
                        differentials = False
                    else:
                        differentials = True

                    regressor.prepare(n, differentials, weight_seed=None)  # Don't set differentials
                    regressor.train("{}".format(d), epochs=epoch)
                    predictions, deltas = regressor.predict_values_and_derivs(x_test)

                    """
                    # Fit differential regression models for price and delts
                    diffreg = DifferentialRegression(degree=d, alpha=alpha)
                    diffreg.fit(x_train, y_train, z_train)
                    diffpred, z_pred = diffreg.predict(x_test, predict_derivs=True)
                    """

                    df.loc[rep, n, epoch, d][['RMSE_PRICE', 'RMSE_DELTA']] = (
                        np.sqrt(np.mean((predictions - y_test)**2)), np.sqrt(np.mean((deltas - z_test)**2))
                    )

    # Summary statistics
    df_tmp = pd.concat([
        df.groupby(['N', 'EPOCH', 'DEG']).mean(),
        df.groupby(['N', 'EPOCH', 'DEG']).std()
        ], axis=1)

    df_tmp.columns = ['MEAN_' + c if i < 2 else 'STD_' + c for i, c in enumerate(df_tmp.columns)]


    df_summary_price = pd.DataFrame(index=pd.MultiIndex.from_product([epochs, N]),
                                    columns=['DEG_' + str(p) for p in deg])
    for d in deg:
        for e in epochs:
            for n in N:
                cell = '{:.3f} ({:.4f})'.format(
                    df_tmp.loc[n, e, d]['MEAN_RMSE_PRICE'], df_tmp.loc[n, e, d]['STD_RMSE_PRICE']
                )
                df_summary_price.loc[e, n]['DEG_' + str(d)] = cell

    df_summary_delta = pd.DataFrame(index=pd.MultiIndex.from_product([epochs, N]),
                                    columns=['DEG_' + str(p) for p in deg])
    for d in deg:
        for e in epochs:
            for n in N:
                cell = '{:.3f} ({:.4f})'.format(
                    df_tmp.loc[n, e, d]['MEAN_RMSE_DELTA'], df_tmp.loc[n, e, d]['STD_RMSE_DELTA']
                )
                df_summary_delta.loc[e, n]['DEG_' + str(d)] = cell



    # Export summary results
    print(df_summary_price.to_latex(float_format='%.4f'))
    print(df_summary_delta.to_latex(float_format='%.4f'))

    df_export = pd.concat([df_summary_price, df_summary_delta], axis=1)
    export_filepath = get_data_path('NN_choos_epoch.csv')
    df_export.to_csv(export_filepath, float_format='%.4f')

    #df_summary_price.to_csv(export_path, float_format='%.4f')
    #df_summary_delta.to_csv('export_path, float_format='%.4f')















