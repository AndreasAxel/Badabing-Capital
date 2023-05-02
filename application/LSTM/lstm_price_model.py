import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from application.utils.path_utils import get_data_path
from sklearn.model_selection import train_test_split

# Long-Short Term Memory network used for estimating deltas through back-propagation
def LSTM_model(num_inputs, num_outputs, *args, **kwargs):
    model = tf.keras.Sequential([
        layers.LSTM(units=32, input_shape=(1, num_inputs)),  # LSTM layer
        layers.Dense(units=num_outputs)  # Regular densely-connected NN layers
    ])
    return model

# Prepare training data into tensor for LSTM model
def prepare_data(X, y):
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X, y

# Train LSTM model for American option pricing
def LSTM_train_model(X_train, y_train, epochs = 1, batch_size = 32, workers=1, use_multiprocessing=False):
    model = LSTM_model(X_train.shape[2], y_train.shape[1])
    model.compile(optimizer='adam', loss='mse') # Compile the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, workers=workers,
              use_multiprocessing=use_multiprocessing)
    return model


# Test the LSTM model to price American options
def LSTM_test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean(y_pred-y_test)**2)  # Penalty function
    print(f"RMSE: {rmse:.4f}")


def predict_price(model, input_data):
    price = model.predict(input_data)
    return price



if __name__ == "__main__":
    epochs = 500
    batch_size = 128
    workers = -1
    use_multiprocessing = True

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
    X_train, y_train = prepare_data(X_train, y_train)
    X_test, y_test = prepare_data(X_test, y_test)

    # Train model
    model = LSTM_train_model(X_train, y_train, epochs=epochs, batch_size=batch_size, workers=workers,
                             use_multiprocessing=use_multiprocessing)

    # Test the model
    print("Starting model test:")
    LSTM_test_model(model, X_test, y_test)

    # Make option price predictions
    print("Starting model prediction")
    price_pred = predict_price(model, X_test)
    print(price_pred)

    import matplotlib.pyplot as plt
    # Select data where r = 0.03, sigma = 0.2, T = 1
    data_base_scenario = data[((data[:, 1] == 0.03) + (data[:, 2] == 0.2) + (data[:, 3] == 1.0)) == 1]
    X_base_scenario = data_base_scenario[:, :-1]
    y_base_scenario = data_base_scenario[:, -1]
    X_base_scenario, y_base_scenario = prepare_data(X_base_scenario, y_base_scenario)
    moneyness = X_base_scenario[:, 0, 0]
    price_lsmc = y_base_scenario
    price_lstm = predict_price(model, X_base_scenario)[:, 0]
    error = ((price_lstm + 1) / (price_lsmc + 1))
    plt.scatter(moneyness, error, color='black', alpha=0.2)
    plt.ylim([0.9, 1.1])
    plt.title('Price NN / Price LSMC')
    plt.xlabel('Moneyness')
    plt.show()

