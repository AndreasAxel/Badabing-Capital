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
    ax.set_xlim(0, 75)
    ax.set_ylim(-10, 35)
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
    alpha_differential_regression = 1.00
    sizeTrain = 1000
    sizeTest = 1000
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

    # 1) Classical linear regression
    linreg = create_polynomial(degree=degree)
    linreg.fit(x_train, y_train)
    linpred = linreg.predict(x_test)

    # 2) Ridge regression
    ridgereg = make_ridge_cv(degree=degree)
    ridgereg.fit(x_train, y_train)
    ridgepred = ridgereg.predict(x_test)
    alpha_ridge = ridgereg['ridgecv'].alpha_

    # 3) Differential regression
    diffreg = DifferentialRegression(degree=degree, alpha=alpha_differential_regression)
    diffreg.fit(x_train, y_train, z_train)
    diffpred, z_pred = diffreg.predict(x_test, predict_derivs=True)

    # Calculate RMSE for the models
    lin_errors = linpred - y_test
    lin_rmse = np.sqrt(np.square(lin_errors).mean())

    ridge_errors = ridgepred - y_test
    ridge_rmse = np.sqrt(np.square(ridge_errors).mean())

    diff_errors = diffpred - y_test
    diff_rmse = np.sqrt(np.square(diff_errors).mean())

    # Plot results
    print("Ridge regression alpha = %.4f" % alpha_ridge)
    fig, lines = plot_multi(
        x_train, y_train, x_test, y_test,
        ["Classical Linear Regression",
         "Ridge Regression (α={:.2f})".format(alpha_ridge),
         "Differential Regression (α={:.2f})".format(alpha_differential_regression)],
        [linpred, ridgepred, diffpred],
        [lin_rmse, ridge_rmse, diff_rmse])
    plt.show()

    # ------------------------------------------------- #
    # Plot predicted delta from Differential Regression #
    # ------------------------------------------------- #

    deltaRmseDiff = np.sqrt(np.square(z_pred - z_test).mean()).round(4)
    plt.scatter(x_train, z_train, marker='x', color='cyan', s=2, alpha=0.5, label='train')
    plt.scatter(x_test, z_test, marker='o', color='red', s=2, alpha=0.5, label="true")
    plt.scatter(x_test, z_pred, marker='o', color='blue', s=2, alpha=0.5, label='diff, RMSE={}'.format(deltaRmseDiff))
    plt.show()

    # -------------------------------------------------- #
    # Comparison to Letourneau & Stentofts' naive method #
    # -------------------------------------------------- #

    if letourneau:
        from application.models.LetourneauStentoft import ISD, disperseFit, Letourneau
        fitted = disperseFit(t0=0,
                         T=1,
                         x0=40,
                         N=10000,
                         M=52,
                         r=0.06,
                         sigma=0.2,
                         K=40,
                         seed=1234,
                         deg_lsmc=9,
                         deg_stentoft=9,
                         option_type='PUT',
                         x_isd=ISD(N=10000, x0=40, alpha=25, seed=1234))
        dataLetourneau = Letourneau(spot=x_test, x0=fitted[0], priceFit=fitted[1], deltaFit=fitted[2], gammaFit=fitted[3])


        plt.scatter(x_test, dataLetourneau[1], color='orange', s=2, alpha=0.5, label='letourneau')
        plt.title("∆ predictions")
        plt.legend()
        plt.show()


    """
    # Piece-wise regression
    if piecewise:
        import pwlf
        myPWLF = pwlf.PiecewiseLinFit(x_train.reshape(sizeTrain), z_train.reshape(sizeTrain))
        # fit the data for n line segments
        res = myPWLF.fit(4)
        # calculate slopes
        slopes = myPWLF.calc_slopes()

        # predict for the determined points
        xHat = x_test.reshape(sizeTest)
        zHat = myPWLF.predict(xHat)
        
        deltaRmsePw = np.sqrt(np.square(zHat - z_test).mean()).round(4)
        plt.scatter(x_test, zHat, marker='o', color='green', s=2, alpha=0.5, label='pw, RMSE = {}'.format(deltaRmsePw))

    """

