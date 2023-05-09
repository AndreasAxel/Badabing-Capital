import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from application.utils.path_utils import get_data_path
from application.utils.visualize_results import plot_results


def create_polynomial(deg = 5):
    return make_pipeline(PolynomialFeatures(degree=deg, order='F'), Normalizer(norm="l2"), LinearRegression(n_jobs=-1)) # Construct pipeline for given estimators

def polynomial_regression(X_train, y_train, X_test, deg=None):
    pol_reg = create_polynomial(deg=deg)
    pol_reg = pol_reg.fit(X_train, y_train)
    pred = pol_reg.predict(X_test)
    return pred


if __name__ == '__main__':
    # Load generated data
    import_filename = get_data_path("training_data_PUT.csv")
    data = np.genfromtxt(import_filename, delimiter=",", skip_header=0)

    # Separate option prices from input data
    idx = np.random.default_rng().integers(low=0, high=66000, size=10000)
    X = data[idx, :-1]
    y = data[idx, -1]
    y = y.reshape((len(y), 1))  # Reshape in order to match shape of predicted y's
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = polynomial_regression(X_train, y_train, X_test, deg=5) # compute classical polynomial predictions
    plt.figure(figsize=(7.5, 7.5))
    plot_results(ax=plt.gca(), y_test=y_test, y_pred=[y_pred], title="continuation value by classic regression, applied to test set")
    _=plt.show()







