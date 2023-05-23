import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from application.utils.path_utils import get_data_path
from application.utils.visualize_results import plot_results
from application.utils.data_management import data_preprocessing


def create_polynomial(deg = 5):
    return make_pipeline(PolynomialFeatures(degree=deg, order='F'), Normalizer(norm="l2"), LinearRegression(n_jobs=-1)) # Construct pipeline for given estimators

def polynomial_regression(X_train, y_train, X_test, deg=None):
    poly_reg = create_polynomial(deg=deg)
    poly_reg = poly_reg.fit(X_train, y_train)
    pred = poly_reg.predict(X_test)
    return pred


if __name__ == '__main__':
    # Load generated data
    import_filename = get_data_path("letourneauStentoft_data.csv")
    data = np.genfromtxt(import_filename, delimiter=",", skip_header=0)

    # Split data into training and test sets
    X_train, y_train, X_test, y_test = data_preprocessing(data=data[:,:2], compute_z=False)

    # compute classical polynomial predictions
    classic_pred = polynomial_regression(X_train, y_train, X_test, deg=5)

    plt.figure(figsize=(7.5, 7.5))
    plot_results(ax=plt.gca(), y_test=y_test, y_pred=[classic_pred],
                 title="continuation value by classic regression, applied to test set")
    _=plt.show()







