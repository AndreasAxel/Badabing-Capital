import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from application.utils.path_utils import get_data_path, get_model_path
from sklearn.model_selection import train_test_split


def format_data(X, y):
    """
    Format data into tensor for model
    :param X: Input data (covariates)
    :param y: Output data
    :return: X, y formatted as tensors
    """

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X, y


def calc_rmse(y_pred, y_test, axis=None):
    """
    Calculate root-mean-squared-error (RMSE)
    :param y_pred
    :param y_test
    :param axis     Axis to take mean over (passed to numpy.mean)
    :return:    RMSE
    """
    return np.sqrt(np.mean(y_pred - y_test, axis=axis) ** 2)


if __name__ == "__main__":

    # Parameters
    epochs = 10000
    batch_size = 128
    workers = -1
    use_multiprocessing = True
    loss = 'mse'
    optimizer = 'adam'

    file_name_save_model = 'LSTM_01'
    file_name_load_model = None

    # --------------------------------------------------------------------- #
    #                                                                       #
    # DATA                                                                  #
    #                                                                       #
    # --------------------------------------------------------------------- #

    # Load generated data
    import_filename = get_data_path("training_data_PUT.csv")
    data = np.genfromtxt(import_filename, delimiter=",", skip_header=0)

    # Separate option prices from input data
    X = data[:, :-1]
    y = data[:, -1]
    y = y.reshape((len(y), 1))  # Reshape in order to match shape of predicted y's

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert covariates into tensors
    X_train, y_train = format_data(X_train, y_train)
    X_test, y_test = format_data(X_test, y_test)

    # Set number of inputs and number of outputs
    num_inputs = X_train.shape[2]
    num_outputs = y_train.shape[1]

    # --------------------------------------------------------------------- #
    #                                                                       #
    # MODEL                                                                 #
    #                                                                       #
    # --------------------------------------------------------------------- #

    if file_name_load_model is not None:
        # Load model from file
        path_name_load_model = get_model_path(file_name_load_model)
        model = tf.keras.models.load_model(path_name_load_model)
    else:
        # Create and train / fit new model
        # Define model
        model = tf.keras.Sequential([
            layers.LSTM(units=32, input_shape=(1, num_inputs)),  # LSTM layer
            layers.Dense(units=num_outputs)  # Regular densely-connected NN layers
        ])

    # The model only trains if epochs is greater than 0
    if epochs > 0:
        # Compile model
        model.compile(optimizer=optimizer, loss=loss)

        # Fit / train model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, workers=workers,
                  use_multiprocessing=use_multiprocessing)

    # Option to save model after training
    if file_name_save_model is not None:
        path_name_save_model = get_model_path(file_name_save_model)
        model.save(path_name_save_model)

    # Test the model "in-sample"
    y_pred = model.predict(X_train)
    rmse = calc_rmse(y_pred=y_pred, y_test=y_train)
    print('RMSE of in-sample:', rmse)

    # Test the model "out-of-sample"
    y_pred = model.predict(X_test)
    rmse = calc_rmse(y_pred=y_pred, y_test=y_test)
    print('RMSE of out-of-sample:', rmse)

    # --------------------------------------------------------------------- #
    #                                                                       #
    # Plot results for "base-scenario"                                      #
    #                                                                       #
    # --------------------------------------------------------------------- #

    # Select data where r = 0.03, sigma = 0.2, T = 1
    data_base_scenario = data[((data[:, 1] == 0.03) + (data[:, 2] == 0.2) + (data[:, 3] == 1.0)) == 1]
    X_base_scenario = data_base_scenario[:, :-1]
    y_base_scenario = data_base_scenario[:, -1]
    X_base_scenario, y_base_scenario = format_data(X_base_scenario, y_base_scenario)
    moneyness = X_base_scenario[:, 0, 0]
    price_lsmc = y_base_scenario
    price_lstm = model.predict(X_base_scenario)[:, 0]
    error = ((price_lstm + 1) / (price_lsmc + 1))
    plt.scatter(moneyness, error, color='black', alpha=0.2)
    plt.ylim([0.5, 1.05])
    plt.title('Price NN / Price LSMC')
    plt.xlabel('Moneyness')
    plt.show()
