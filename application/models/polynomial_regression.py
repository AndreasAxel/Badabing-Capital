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

def polynomial_regression(X_train, y_train, X_test, deg=5):
    poly_reg = create_polynomial(deg=deg)
    poly_reg = poly_reg.fit(X_train, y_train)
    pred = poly_reg.predict(X_test)
    return pred


## plotting forked from differential regression notebook

# plot results
def plot_one(ax, x_train, y_train, x_test, y_test, pred):
    ax.set_xlim(0, 75)
    ax.set_ylim(-10, 35)
    samples, = ax.plot(x_train, y_train, 'co', markersize=5, markerfacecolor="white", label="samples")
    predict, = ax.plot(x_test, pred, 'bo', label="predict")
    #correct, = ax.plot(x_test, y_test, 'ro', label="correct")
    return samples, predict #, correct


def plot_multi(x_train, y_train, x_test, y_test, titles, preds):
    nplots = len(preds)
    nrows = (nplots - 1) // 3 + 1
    ncols = min(nplots, 3)

    fig, axs = plt.subplots(nrows, ncols, squeeze=False)
    fig.set_size_inches(ncols * 5, nrows * 5)

    lines = []
    for i, ax in enumerate(axs.flatten()):
        if i < nplots:
            samples, predict = plot_one(ax, x_train, y_train, x_test, y_test, preds[i])
            #samples, predict, correct = plot_one(ax, x_train, y_train, x_test, y_test, preds[i])
            lines.extend([samples, predict])
            #lines.extend([samples, predict, correct])
            ax.legend()
            ax.set_title(titles[i])

    return fig, lines



if __name__ == '__main__':
    # Load generated data
    import_filename = get_data_path("LSMC_pathwise_ISD.csv")
    data = np.genfromtxt(import_filename, delimiter=",", skip_header=0)

    from application.models.differential_regression import DifferentialRegression

    # Split data into training and test sets
    #X_train, y_train, X_test, y_test = data_preprocessing(data=data[:,:2], compute_z=False)

    X_train, y_train, z_train, X_test, y_test, z_test = data_preprocessing(data=data[:, :3], compute_z=True)

    # compute classical polynomial predictions
    classic_pred = polynomial_regression(X_train, y_train, X_test, deg=5)

    diffreg = DifferentialRegression()
    diffreg.fit(X_train, y_train, z_train)
    diffpred = diffreg.predict(X_test)


    fig, lines = plot_multi(X_train, y_train, X_test, y_test,
                            ["linear regression", "differential regression"],
                            [classic_pred, diffpred])
    plt.show()
    """
    plt.figure(figsize=(7.5, 7.5))
    plot_results(ax=plt.gca(), y_test=y_test, y_pred=[classic_pred],
                 title="continuation value by classic regression, applied to test set")
    _=plt.show()
    """







