import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Long-Short Term Memory network used for estimating deltas through back-propagation
def LSTM_model(num_inputs, num_outputs, *args, **kwargs):
    model = tf.keras.sequential([
        layers.LSTM(units=32, input_shape=(1, num_inputs)), #LSTM layer
        layers.Dense(units=num_outputs, activation = "sigmoid") # regular densely-connected NN layers
    ])
    return model

# Prepare training data into tensor for LSTM model
def prepare_data(X, y):
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X, y

# Train LSTM model for American option pricing
def LSTM_train_model(X_train, y_train, epocs = 2, batch_size = 100):
    model = LSTM_model(X_train.shape[2], y_train.shape[1])
    model.compile(optimizer='adam', loss='mse') # Compile the model
    model.fit(X_train, y_train, epocs, batch_size)
    return model


# Test the LSTM model to price American options
def LSTM_test_model(model, X_test, y_test):
    y_pred = model.predict(X_test):
    rmse = np.sqrt(np.mean(y_pred-y_test)**2) # penalty function
    print(f"RMSE: {rmse:.4f}")


def predict_delta(model, input_data):
    input_data = np.array(input_data)
    input_data = np.reshape(input_data, (1, 1, input_data.shape[0]))
    price = model.predict(input_data)
    return price




