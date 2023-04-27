import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import csv
from pathlib import Path

# Long-Short Term Memory network used for estimating deltas through back-propagation
def LSTM_model(num_inputs, num_outputs, *args, **kwargs):
    model = tf.keras.Sequential([
        layers.LSTM(units=32, input_shape=(1, num_inputs)), #LSTM layer
        layers.Dense(units=num_outputs) # regular densely-connected NN layers
    ])
    return model

# Prepare training data into tensor for LSTM model
def prepare_data(X, y):
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X, y

# Train LSTM model for American option pricing
def LSTM_train_model(X_train, y_train, epochs = 1, batch_size = 100):
    model = LSTM_model(X_train.shape[2], y_train.shape[1])
    model.compile(optimizer='adam', loss='mse') # Compile the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model


# Test the LSTM model to price American options
def LSTM_test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    #y_pred = y_pred.reshape((len(y_pred), ))
    print("y_pred shape: ", y_pred.shape)
    print("y_test_shape:", y_test.shape)
    #print("y_pred type: ", np.typename(y_pred))
    print("y_pred: ", y_pred)
    print("y_test:", y_test)
    rmse = np.sqrt(np.mean(y_pred-y_test)**2)  # penalty function
    print(f"RMSE: {rmse:.4f}")


def predict_price(model, input_data):
    #input_data = np.array(input_data)
    #input_data = np.reshape(input_data, (1, 1, input_data.shape[0]))
    price = model.predict(input_data)
    return price



if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # Load generated data
    import_filename = Path(__file__).parent.parent.parent / "data/training_data_PUT.csv"
    data = np.genfromtxt(import_filename, delimiter = ",", skip_header = 0)

    # Seperate option prices from input data
    X = data[:, :-1]
    y = data[:, -1]
    y = y.reshape((len(y), 1)) # Reshape in order to match shape of predicted y's

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert covariates into tensors
    X_train, y_train = prepare_data(X_train, y_train)
    X_test, y_test = prepare_data(X_test, y_test)

    # Train model
    model = LSTM_train_model(X_train, y_train)

    # Test the model
    print("Starting model test:")
    LSTM_test_model(model, X_test, y_test)

    # Make option price predictions
    print("Starting model prediction")
    price_pred = predict_price(model, X_test)
    print(price_pred)

