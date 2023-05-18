import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from application.utils.path_utils import get_data_path
from application.utils.visualize_results import plot_results
from application.models.polynomial_regression import polynomial_regression


class DifferentialRegression:

    def __init__(self, degree=5, alpha=1.0):
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

        # note we use np.linalg.pinv (as opposed to np.linalg.inv) to perform safe (SVD) inversion, resilient to near singularities
        inv = np.linalg.pinv(self.phi_.T @ self.phi_ + self.alpha * phiTphi, hermitian=True)
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


if __name__ == '__main__':
    # Load generated data
    import_filename = get_data_path("LSMC_pathwise.csv")
    data = np.genfromtxt(import_filename, delimiter=",", skip_header=0)

    # Separate option prices from input data
    #idx = np.random.default_rng().integers(low=0, high=66000, size=10000)

    X = data[:,0].reshape(-1, 1) # Spot
    y = data[:,1:].reshape(-1, 2) # Include payoff and delta in order to make proper train test split
    #y = y.reshape(np.shape(y)[0], 2)  # Reshape in order to match shape of predicted y's


    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)


    # divide data properly to also consider derivatives z
    z_train = y_train[:, 1, np.newaxis]
    y_train = y_train[:, 0, np.newaxis]
    z_test = y_test[:, 1, np.newaxis]
    y_test = y_test[:, 0, np.newaxis]


    # compute classical polynomial predictions
    classic_pred = polynomial_regression(X_train, y_train, X_test, deg=5)

    # construct differential regression
    diff_reg = DifferentialRegression()
    diff_reg.fit(X_train, y_train, z_train)
    diff_pred, z_pred = diff_reg.predict(X_test, predict_derivs=True)


    plt.figure(figsize=(7.5, 7.5))
    plot_results(ax=plt.gca(), y_test=y_test, y_pred=[classic_pred, diff_pred],
                 title="continuation value by differential regression, applied to test set",
                 colors=['c', 'm'],
                 labels=['classic regression', 'differential regression'])
    _ = plt.show()

    plt.figure(figsize=(7.5, 7.5))
    plot_results(ax=plt.gca(), y_test=z_test, y_pred=[z_pred],
                 title="delta values by differential regression, applied to test set")
    _ = plt.show()






