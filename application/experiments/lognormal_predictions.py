from application.utils.path_utils import get_data_path
from sklearn.utils import resample
from application.experiments.nn_trainingsize import Neural_approximator
from application.experiments.nn_change_activation_function import graph
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Param setting
    seed = 1234
    sizeTrain = [1024, 8192] # Plot predictions for each training number
    sizeTest = 1000           # Number of test observations used for predictions
    spot_cutoff = False
    showDeltas = True

    # Load generated, pathwise data
    pathwisePath = get_data_path("LSMC_pathwise_lognormal.csv")
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

    plt.scatter(x_train, z_train, alpha=0.5)
    plt.show()


    # Load Binomial data for reference
    binomialPath = get_data_path("binomial_unif.csv")
    dataBinomial = np.genfromtxt(binomialPath, delimiter=",", skip_header=0)
    if spot_cutoff:
        dataBinomial = dataBinomial[335:, :]
    # assigning datastructures
    dataBinomial = resample(dataBinomial, n_samples=len(dataBinomial), random_state=seed)
    """
    # resampled entire data to be sure cropping works
    if sizeTest < len(dataBinomial):
        dataBinomial = resample(dataBinomial, n_samples=sizeTest)
    """
    x_test = dataBinomial[:sizeTest, 0].reshape(-1, 1)
    y_test = dataBinomial[:sizeTest, 1].reshape(-1, 1)
    z_test = dataBinomial[:sizeTest, 2].reshape(-1, 1)

    # Learn neural approximation
    regressor = Neural_approximator(x_raw=x_train, y_raw=y_train, dydx_raw=z_train)

    sizes = sizeTrain
    weightSeed = 1234
    deltidx = 0

    predvalues = {}
    preddeltas = {}

    for size in sizes:
        print("\nsize %d" % size)
        regressor.prepare(size, False, weight_seed=weightSeed)

        regressor.train("standard training")
        predictions, deltas = regressor.predict_values_and_derivs(x_test)
        predvalues[("standard", size)] = predictions
        preddeltas[("standard", size)] = deltas[:, deltidx]

        regressor.prepare(size, True, weight_seed=weightSeed)

        regressor.train("differential training")
        predictions, deltas = regressor.predict_values_and_derivs(x_test)
        predvalues[("differential", size)] = predictions
        preddeltas[("differential", size)] = deltas[:, deltidx]

    # show predicitions
    graph(title=" ",
          predictions=predvalues,
          xAxis=x_test,
          xAxisName="",
          yAxisName="",
          targets=y_test,
          sizes=sizes,
          computeRmse=True,
          weights=None
          )

    # show deltas
    if showDeltas:
        graph(title=" ",
              predictions=preddeltas,
              xAxis=x_test,
              xAxisName="",
              yAxisName="",
              targets=z_test.reshape(-1, ),  # reshaped in order to match dimensionality
              sizes=sizes,
              computeRmse=True,
              weights=None
              )







