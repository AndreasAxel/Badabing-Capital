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


def plot_one(ax, x_train, y_train, x_test, y_test, pred, rmse=None, alpha_ridge_label = None):
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
        samples, = ax.plot(x_train, y_train, 'co', markersize=5, markerfacecolor="white", label="Samples")
        alpha_label = [0.5, 1.0]
        color = ['blue', 'purple']
        for i, prediction in enumerate(pred):
            predict, = ax.plot(x_test, pred[i], color[i], linestyle='solid', label="Predict for α=" + str(alpha_label[i]) + " , rmse=%.4f" % rmse[i])

        correct, = ax.plot(x_test, y_test, 'r-', label="Binomial Model")
        return samples, predict, correct

    # to put alpha label on legend rather than on title
    if alpha_ridge_label != None:
        samples, = ax.plot(x_train, y_train, 'co', markersize=5, markerfacecolor="white", label="Samples")
        predict, = ax.plot(x_test, pred, 'b-', label="Predict for α={:.2f}".format(alpha_ridge_label) + ", rmse=%.4f" % rmse)
        correct, = ax.plot(x_test, y_test, 'r-', label="Binomial Model")
        return samples, predict, correct

    samples, = ax.plot(x_train, y_train, 'co', markersize=5, markerfacecolor="white", label="Samples")
    predict, = ax.plot(x_test, pred, 'b-', label="Predict"+", rmse=%.4f" % rmse)
    correct, = ax.plot(x_test, y_test, 'r-', label="Binomial Model")
    return samples, predict, correct


def plot_multi(x_train, y_train, x_test, y_test, titles, preds, rmse=None, alpha_ridge_label = None):
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
            if titles[i] == "Ridge Regression":
                samples, predict, correct = plot_one(ax, x_train, y_train, x_test, y_test, preds[i], rmse[i],
                                                     alpha_ridge_label)
            else:
                samples, predict, correct = plot_one(ax, x_train, y_train, x_test, y_test, preds[i], rmse[i])
            lines.extend([samples, predict, correct])
            ax.legend()
            ax.set_title(titles[i])
    return fig, lines


if __name__ == '__main__':
    # ------------------------------------------------- #
    # Parameter settings                                #
    # ------------------------------------------------- #
    degree = 9
    alpha_differential_regression = [0.00, 0.5, 1.00]
    sizeTrain = 1024
    sizeTest = 5000
    seedNo = 1234
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
        dataPathwise = resample(dataPathwise, n_samples=sizeTrain, random_state=seedNo)

    x_train = dataPathwise[:, 0].reshape(-1, 1)
    y_train = dataPathwise[:, 1].reshape(-1, 1)
    z_train = dataPathwise[:, 2].reshape(-1, 1)

    if sizeTest < len(dataBinomial):
        dataBinomial = resample(dataBinomial, n_samples=sizeTest, random_state=seedNo)

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

    # Calculate RMSE for the models

    lin_errors = linpred - y_test
    lin_rmse = np.sqrt(np.square(lin_errors).mean())

    ridge_errors = ridgepred - y_test
    ridge_rmse = np.sqrt(np.square(ridge_errors).mean())

    # differential RMSE
    diff_0_errors = diffpred_0 - y_test
    diff_0_rmse = np.sqrt(np.square(diff_0_errors).mean())

    diff_05_errors = diffpred_05 - y_test
    diff_05_rmse = np.sqrt(np.square(diff_05_errors).mean())
    diff_05_rmse = np.sqrt(np.mean((diff_05_errors) ** 2))


    diff_1_errors = diffpred_1 - y_test
    diff_1_rmse = np.sqrt(np.square(diff_1_errors).mean())

    # Plot results
    """
    print("Ridge regression alpha = %.4f" % alpha_ridge)
    fig, lines = plot_multi(
        x_train, y_train, x_test, y_test,
        ["Classical Linear Regression",
         "Ridge Regression (α={:.2f})".format(alpha_ridge),
         "Differential Regression (α={:.2f})".format(alpha_differential_regression)],
        [linpred, ridgepred, diffpred],
        [lin_rmse, ridge_rmse, diff_rmse])
    plt.show()
    """

    # Pricing functions
    fig, lines = plot_multi(
        x_train, y_train, x_test, y_test,
        titles = ["Classical Linear Regression",
         "Ridge Regression",
         "Differential Regression"],
        preds = [linpred, ridgepred, [diffpred_05, diffpred_1]],
        rmse = [lin_rmse, ridge_rmse, [diff_05_rmse, diff_1_rmse]],
        alpha_ridge_label= alpha_ridge
    )
    plt.show()


    # ------------------------------------------------- #
    # Plot predicted delta from Differential Regression #
    # ------------------------------------------------- #

    deltaRmseDiff_0 = np.sqrt(np.square(z_pred_0 - z_test).mean()).round(4)
    deltaRmseDiff_05 = np.sqrt(np.square(z_pred_05 - z_test).mean()).round(4)
    deltaRmseDiff_1 = np.sqrt(np.square(z_pred_1 - z_test).mean()).round(4)

    plt.plot(x_train, z_train, 'co', markersize=5, markerfacecolor="white", alpha=0.5, label='Samples')
    plt.plot(x_test, z_test, marker='o', color='red', markersize=2, alpha=0.5, label="Binomial Model")
    plt.plot(x_test, z_pred_0, marker='o', color='blue', markersize=2, alpha=0.5,
                label='diff. reg. α='+str(alpha_differential_regression[0])+', RMSE={}'.format(deltaRmseDiff_0))
    plt.plot(x_test, z_pred_05, marker='o', color='purple', markersize=2, alpha=0.5,
                label='diff. reg. α=' + str(alpha_differential_regression[1]) + ', RMSE={}'.format(deltaRmseDiff_05))
    plt.plot(x_test, z_pred_1, marker='o', color='orange', markersize=2, alpha=0.5,
                label='diff. reg. α=' + str(alpha_differential_regression[2]) + ', RMSE={}'.format(deltaRmseDiff_1))

    # -------------------------------------------------- #
    # Comparison to Letourneau & Stentofts' naive method #
    # -------------------------------------------------- #

    if letourneau:
        from application.models.LetourneauStentoft import ISD, disperseFit, Letourneau
        fitted = disperseFit(t0=0,
                         T=1,
                         x0=40,
                         N=sizeTrain,
                         M=52,
                         r=0.06,
                         sigma=0.2,
                         K=40,
                         seed=seedNo,
                         deg_lsmc=9,
                         deg_stentoft=5,
                         option_type='PUT',
                         x_isd=ISD(N=sizeTrain, x0=40, alpha=25, seed=seedNo))
        dataLetourneau = Letourneau(spot=x_test, x0=fitted[0], priceFit=fitted[1], deltaFit=fitted[2], gammaFit=fitted[3])


        plt.scatter(x_test, dataLetourneau[1], color='orange', s=2, alpha=0.5, label='letourneau')
    plt.title("∆ predictions")
    plt.legend(draggable=True)
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

