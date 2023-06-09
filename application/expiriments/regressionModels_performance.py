import matplotlib.pyplot as plt
import numpy as np
from application.utils.path_utils import get_data_path
from sklearn.utils import resample
from application.models.regressionModels import DifferentialRegression


if __name__ == '__main__':
    # Param setting
    degree = 9
    alpha_differential_regression = [0.00, 0.5, 1.00]
    seed = 1234
    sizeTrain = 99999 # Plot predictions for each training number
    sizeTest = 1000          # Number of test observations used for predictions
    spot_cutoff = False     # For important sampling

    # Load generated, pathwise data
    pathwisePath = get_data_path("LSMC_pathwise_ISD.csv")
    dataPathwise = np.genfromtxt(pathwisePath, delimiter=",", skip_header=0)
    # Assigning datastructures
    dataPathwise = resample(dataPathwise, n_samples=len(dataPathwise), random_state=seed)
    """
    # resampled entire data to be sure cropping works
    # assigning datastructures
    if sizeTrain < len(dataPathwise):
        dataPathwise = resample(dataPathwise, n_samples=sizeTrain)
    """
    x_train = dataPathwise[:, 0].reshape(-1, 1)
    y_train = dataPathwise[:, 1].reshape(-1, 1)
    z_train = dataPathwise[:, 2].reshape(-1, 1)


    # Load Binomial data for reference
    binomialPath = get_data_path("binomial_unif.csv")
    dataBinomial = np.genfromtxt(binomialPath, delimiter=",", skip_header=0)

    if spot_cutoff:
        dataBinomial = dataBinomial[335:, :]
    # assigning datastructures
    dataBinomial = resample(dataBinomial, n_samples=sizeTest, random_state=seed)

    """
    # resampled entire data to be sure cropping works
    if sizeTest < len(dataBinomial):
        dataBinomial = resample(dataBinomial, n_samples=sizeTest)
    """
    x_test = dataBinomial[:sizeTest, 0].reshape(-1, 1)
    y_test = dataBinomial[:sizeTest, 1].reshape(-1, 1)
    z_test = dataBinomial[:sizeTest, 2].reshape(-1, 1)

    mask = np.argsort(x_test.reshape(np.size(x_test), ))
    x_test = x_test[mask]
    y_test = y_test[mask]
    z_test = z_test[mask]

    # 3) Differential regression
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

    # Plot results

    def plot_performance(x_test, predictedVals, predictedDeltas, save=False, savePath=None, figName=None):
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(4 * 2 + 1.5, 4 * 2)
        ax[0, 0].set_title("Price")
        ax[0, 1].set_title("Delta")

        ax[0, 0].plot(x_test, predictedVals[0],  label="α=0.0")
        #ax[0, 0].plot(x_test, predictedVals[1], linestyle='solid', color = 'purple', markersize=2, markerfacecolor='white', label="α=0.5")
        #ax[0, 0].plot(x_test, predictedVals[2], '.', color = 'orange', markersize=2, markerfacecolor='white', label="α=1.0")
        #ax[0, 0].plot(x_test, y_test,'.', color = 'red', markersize=0.5, label='Binomial Model')
        ax[0, 0].legend(prop={'size': 8}, loc='upper left')

        ax[0, 1].plot(x_test, predictedDeltas[0], 'bo', markersize=2, markerfacecolor='white', label="α=0.0")
        ax[0, 1].plot(x_test, predictedDeltas[1], 'bo', markersize=2, markerfacecolor='white', label="α=0.5")
        ax[0, 1].plot(x_test, predictedDeltas[2], 'bo', markersize=2, markerfacecolor='white', label="α=1.0")
        ax[0, 1].plot(x_test, z_test, 'r.', markersize=0.5, label='Binomial Model')
        ax[0, 1].legend(prop={'size': 8}, loc='upper left')

        ax[1, 0].plot(x_test, predictedVals[0] - y_test, 'bo', markersize=2, markerfacecolor='white', label="Predicted")
        ax[1, 0].plot(x_test, predictedVals[1] - y_test, 'bo', markersize=2, markerfacecolor='white', label="Predicted")
        ax[1, 0].plot(x_test, predictedVals[2] - y_test, 'bo', markersize=2, markerfacecolor='white', label="Predicted")
        ax[1, 0].plot(x_test, y_test - y_test, 'r.', markersize=0.5, label='Binomial Model')

        ax[1, 1].plot(x_test, predictedDeltas[0] - z_test, 'bo', markersize=2, markerfacecolor='white', label="Predicted")
        ax[1, 1].plot(x_test, predictedDeltas[1] - z_test, 'bo', markersize=2, markerfacecolor='white', label="Predicted")
        ax[1, 1].plot(x_test, predictedDeltas[2] - z_test, 'bo', markersize=2, markerfacecolor='white', label="Predicted")
        ax[1, 1].plot(x_test, z_test.reshape(-1) - z_test.reshape(-1), 'r.', markersize=0.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        if save:
            plt.savefig(savePath + figName + '.png', dpi=400)
        plt.show()


    plot_performance(x_test=x_test,
                     predictedVals=[diffpred_0, diffpred_05, diffpred_1],
                     predictedDeltas=[z_pred_0, z_pred_05, z_pred_1],
                     save=False,
                     savePath='/Users/sebastianhansen/Documents/UNI/PUK/',
                     figName='nnPerfomance'
                     )

