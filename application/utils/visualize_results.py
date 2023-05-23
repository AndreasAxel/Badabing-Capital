import numpy as np


def plot_results(ax, y_test, y_pred, title, labels=None, colors=None):

    for i, pred in enumerate(y_pred):
        errors = pred - y_test
        rmse = np.sqrt(np.square(errors).mean())
        ax.plot(y_test, pred, ls="None", marker='o', color=colors[i] if colors else 'c', markerfacecolor="white", label="predicted"+(" by " + labels[i] if labels else '')+", rmse=%.2f" % rmse)

    ax.plot(y_test, y_test, 'r--', label='test data')
    ax.set_title(title)
    #ax.set_xlim(ax.get_ylim())
    ax.legend()

