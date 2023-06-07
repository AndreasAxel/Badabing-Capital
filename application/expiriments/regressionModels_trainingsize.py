import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression, RidgeCV
from application.utils.path_utils import get_data_path

"""
Functions and classes are based on Differential Regression Notebook

https://github.com/differential-machine-learning
"""

class DifferentialRegression:
    def __init__(self, degree=5, alpha=0.5):
        self.degree = degree
        self.polynomial_features = PolynomialFeatures(degree=degree, include_bias=True)
        self.alpha = alpha
        self.epsilon = 1.0e-08

    def fit(self, x, y, z):
        self.phi_ = self.polynomial_features.fit_transform(x)
        self.powers_ = self.polynomial_features.powers_

        self.dphi_ = self.phi_[:, :, np.newaxis] * self.powers_[np.newaxis, :, :] / (x[:, np.newaxis, :] + self.epsilon)

        self.lamj_ = ((y ** 2).mean(axis=0) / (z ** 2).mean(axis=0)).reshape(1, 1, -1)
        self.dphiw_ = self.dphi_ * self.lamj_

        phiTphi = np.tensordot(self.dphiw_, self.dphi_, axes=([0, 2], [0, 2]))
        phiTz = np.tensordot(self.dphiw_, z, axes=([0, 2], [0, 1])).reshape(-1, 1)

        inv = np.linalg.inv(self.phi_.T @ self.phi_ + self.alpha * phiTphi)
        self.beta_ = (inv @ (self.phi_.T @ y + self.alpha * phiTz)).reshape(-1, 1)

    def predict(self, x, predict_derivs=False):
        phi = self.polynomial_features.transform(x)
        y_pred = phi @ self.beta_

        if predict_derivs:
            dphi = phi[:, :, np.newaxis] * self.powers_[np.newaxis, :, :] / (x[:, np.newaxis, :] + self.epsilon)
            z_pred = np.tensordot(dphi, self.beta_, (1, 0)).reshape(dphi.shape[0], -1)
            return y_pred, z_pred
        else:
            return y_pred


def create_polynomial(degree = 5):
    """
    Classic linear regression

    :param degree:
    :return:
    """
    # Construct pipeline for given estimators
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, order='F'),
        LinearRegression(n_jobs=-1, fit_intercept=True)
    )


def make_ridge_cv(degree=5, min_alpha=1e-05, max_alpha=1e02, num_alphas=100):
    """
    Ridge regression with built-in cross-validation on alpha

    :param degree:
    :param min_alpha:
    :param max_alpha:
    :param num_alphas:
    :return:
    """
    alphas = np.exp(np.linspace(np.log(min_alpha), np.log(max_alpha), num_alphas))
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree),
        RidgeCV(alphas=alphas)
    )


def plot_one(ax, x_train, y_train, x_test, y_test, pred, rmse=None):
    """
    Function for creating each subplot

    :param ax:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param pred:
    :param rmse:
    :return:
    """
    ax.set_xlim(10, 75)
    ax.set_ylim(-5, 25)
    if np.shape(pred)[1] > 1:
        samples, = ax.plot(x_train, y_train, 'co', markersize=5, markerfacecolor="white", label="samples")
        alphas = [0.5, 1]
        color = ['blue', 'purple']
        for i, prediction in enumerate(pred):
            predict, = ax.plot(x_test, pred[i], color[i], linestyle='solid', label="predict for α=" + str(alphas[i]) + " , rmse=%.4f" % rmse[i])

        correct, = ax.plot(x_test, y_test, 'r-', label="correct")
        return samples, predict, correct

    samples, = ax.plot(x_train, y_train, 'co', markersize=5, markerfacecolor="white", label="samples")
    predict, = ax.plot(x_test, pred, 'b-', label="predict"+", rmse=%.4f" % rmse)
    correct, = ax.plot(x_test, y_test, 'r-', label="correct")
    return samples, predict, correct


def plot_multi(x_train, y_train, x_test, y_test, titles, preds, rmse=None):
    """
    Function for creating the entire figure composited of subplots

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param titles:
    :param preds:
    :param rmse:
    :return:
    """
    nplots = len(preds)
    nrows = (nplots - 1) // 3 + 1
    ncols = min(nplots, 3)

    fig, axs = plt.subplots(nrows, ncols, squeeze=False)
    fig.set_size_inches(ncols * 5, nrows * 5)

    lines = []
    for i, ax in enumerate(axs.flatten()):
        if i < nplots:
            samples, predict, correct = plot_one(ax, x_train, y_train, x_test, y_test, preds[i], rmse[i])
            lines.extend([samples, predict, correct])
            ax.legend()
            ax.set_title(titles[i])
    return fig, lines


if __name__ == '__main__':
    # ------------------------------------------------- #
    # Parameter settings                                #
    # ------------------------------------------------- #
    degree = 5
    alpha_differential_regression = [0.00, 0.5, 1.00]
    sizeTrain = 9999
    sizeTest = 5000
    seedNo = None
    letourneau = False # include Letourneau comparison Delta prediction
    piecewise = False  # include piecewise linear regression in Delta prediction

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

    # assigning datastructures
    if sizeTrain < len(dataPathwise):
        dataPathwise = resample(dataPathwise, n_samples=sizeTrain)

    x_train = dataPathwise[:, 0].reshape(-1, 1)
    y_train = dataPathwise[:, 1].reshape(-1, 1)
    z_train = dataPathwise[:, 2].reshape(-1, 1)

    if sizeTest < len(dataBinomial):
        dataBinomial = resample(dataBinomial, n_samples=sizeTest)

    x_test = dataBinomial[:, 0].reshape(-1, 1)
    y_test = dataBinomial[:, 1].reshape(-1, 1)
    z_test = dataBinomial[:, 2].reshape(-1, 1)

    # ------------------------------------------------- #
    # Learn and predict for different methods,          #
    # Calculate RMSE for each model, and                #
    # Plot the results                                  #
    # ------------------------------------------------- #

    def graph(title,
              predictions,
              xAxis,
              xAxisName,
              yAxisName,
              targets,
              sizes,
              computeRmse=False,
              weights=None,
              save=False):

        numRows = len(sizes)
        numCols = 3

        fig, ax = plt.subplots(numRows, numCols, squeeze=False)
        fig.set_size_inches(4 * numCols + 1.5, 4 * numRows)

        for i, size in enumerate(sizes):
            ax[i, 0].annotate("size %d" % size, xy=(0, 0.5),
                              xytext=(-ax[i, 0].yaxis.labelpad - 5, 0),
                              xycoords=ax[i, 0].yaxis.label, textcoords='offset points',
                              ha='right', va='center')

        ax[0, 0].set_title("Linear Regression")
        ax[0, 1].set_title("Ridge Regression")
        ax[0, 2].set_title("Differential Regression")

        for i, size in enumerate(sizes):
            for j, regType in enumerate(["standard", "ridge", "differential"]):

                if computeRmse:
                    errors = (predictions[(regType, size)] - targets)
                    if weights is not None:
                        errors /= weights
                    rmse = np.sqrt((errors ** 2).mean(axis=0))
                    t = "RMSE %.5f" % rmse
                else:
                    t = xAxisName

                ax[i, j].set_xlabel(t)
                ax[i, j].set_ylabel(yAxisName)
                # ax[i, j].set_xlim(35, 45)

                if regType == 'ridge':
                    ax[i, j].plot(xAxis, predictions[(regType, size)], 'co',
                                  markersize=2, markerfacecolor='white', label=r"Predicted, $\alpha=${}".format(np.round(alphas[(regType, size)] ,3)))
                else:
                    ax[i, j].plot(xAxis, predictions[(regType, size)], 'co',
                                  markersize=2, markerfacecolor='white', label="Predicted")
                ax[i, j].plot(xAxis, targets, 'r.', markersize=0.5, label='Binomial Model')

                ax[i, j].legend(prop={'size': 8}, loc='upper left')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        # plt.suptitle("% s -- %s" % (title, yAxisName), fontsize=16)
        if save:
            plt.savefig('/Users/sebastianhansen/Documents/UNI/PUK/regression.png', dpi=400)
        plt.show()



    sizes = [256, 1024]
    predvalues = {}
    preddeltas = {}
    alphas = {}

    ## assign plot data
    for size in sizes:
        xtrain = x_train[:size  ]  # resample(x_train, n_samples=size)
        ytrain = y_train[:size  ]  # resample(y_train, n_samples=size)
        ztrain = z_train[:size  ]  # resample(z_train, n_samples=size)

        # xtest = resample(x_test, n_samples=size)
        scaler = StandardScaler()
        scaler.fit(x_train)
        xtest = scaler.transform(x_test)
        xtest = x_test



        # 1) Classical linear regression
        linreg = create_polynomial(degree=degree)
        linreg.fit(xtrain, ytrain)
        linpred = linreg.predict( x_test)

        linDeriv = np.polyder(linreg['linearregression'].coef_[0][::-1], 1)
        linDevs = np.polyval(linDeriv, (x_test - np.mean(x_train)) / np.std(x_train)) / np.std(x_train)

        predictions, deltas = linpred, linDevs
        predvalues[("standard", size)] = predictions
        preddeltas[("standard", size)] = deltas

        # 2) Ridge regression
        ridgereg = make_ridge_cv(degree=degree)
        ridgereg.fit(xtrain, ytrain)

        ridgeDeriv = np.polyder(ridgereg['ridgecv'].coef_[0][::-1], 1)
        ridgepred = ridgereg.predict(x_test)

        ridgeDevs = np.polyval(ridgeDeriv, (x_test - np.mean(x_train)) / np.std(x_train)) / np.std(x_train)
        alpha_ridge = ridgereg['ridgecv'].alpha_

        predictions, deltas = ridgepred, ridgeDevs
        predvalues[("ridge", size)] = predictions
        alphas[('ridge', size)] = alpha_ridge
        preddeltas[("ridge", size)] = deltas

        # 3) Differential regression
        diffreg = DifferentialRegression(degree=degree, alpha=1)
        diffreg.fit(xtrain, ytrain, ztrain)
        diffpred, z_pred = diffreg.predict(x_test, predict_derivs=True)

        predictions, deltas = diffpred, z_pred
        predvalues[("differential", size)] = predictions
        preddeltas[("differential", size)] = deltas



    # Show payoff predictions
    graph(title="",
          predictions=predvalues,
          xAxis=x_test,
          xAxisName="",
          yAxisName="",
          targets=y_test,
          sizes=sizes,
          computeRmse=True,
          weights=None,
          save=False
          )


    # Show delta predictions
    graph(title="",
          predictions=preddeltas,
          xAxis=x_test,
          xAxisName="",
          yAxisName="",
          targets=z_test,
          sizes=sizes,
          computeRmse=True,
          weights=None,
          save=False
          )

