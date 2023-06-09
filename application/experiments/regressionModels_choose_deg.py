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
    N = [256*4**i for i in range(5)]
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

    #df_summary_price.to_csv('/Users/sebastianhansen/Documents/UNI/PUK/regtablePrice.csv', float_format='%.4f')
    #df_summary_delta.to_csv('/Users/sebastianhansen/Documents/UNI/PUK/regtableDelta.csv', float_format='%.4f')


