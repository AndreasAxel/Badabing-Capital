import numpy as np
import tensorflow as tf


def fit_poly(x, y, deg, *args, **kwargs):
    return np.polyfit(x, y, deg)


def pred_poly(x, fit, *args, **kwargs):
    return np.polyval(fit, x)


def fit_laguerre_poly(x, y, deg, *args, **kwargs):
    return np.polynomial.laguerre.lagfit(x, y, deg)


def pred_laguerre_poly(x, fit, *args, **kwargs):
    return np.polynomial.laguerre.lagval(x, fit)


def NN_fit(X, y, num_epochs=10, batch_size=None, *args, **kwargs):
    num_features = len(X)
    X = np.reshape(X, (num_features, 1))
    X = tf.convert_to_tensor(X)

    # Define the neural network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, activation='relu', input_shape=(num_features, 1)),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the neural network
    model.fit(X, y, epochs=num_epochs, batch_size=batch_size, workers=4, use_multiprocessing=True, verbose=0)
    return model


def NN_pred(X, fit, *args, **kwargs):
    num_features = len(X)
    X = np.reshape(X, (num_features, 1))
    return fit.predict(X).flatten()
