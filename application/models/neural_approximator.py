



# Taken from Huge & Savine's Differential ML github

### Initial requirements for code to work - START
import tensorflow as tf2

# disable eager execution etc
tf = tf2.compat.v1
#tf.disable_eager_execution()

# representation of real numbers in TF, change here for 32/64 bits
real_type = tf.float32

### Initial requirements for code to work - END

def vanilla_net(
        input_dim,  # dimension of inputs, e.g. 10
        hidden_units,  # units in hidden layers, assumed constant, e.g. 20
        hidden_layers,  # number of hidden layers, e.g. 4
        seed):  # seed for initialization or None for random

    # set seed

    tf.set_random_seed(seed)

    # input layer
    xs = tf.placeholder(shape=[None, input_dim], dtype=real_type)

    # connection weights and biases of hidden layers
    ws = [None]
    bs = [None]
    # layer 0 (input) has no parameters

    # layer 0 = input layer
    zs = [xs]  # eq.3, l=0

    # first hidden layer (index 1)
    # weight matrix
    ws.append(tf.get_variable("w1", [input_dim, hidden_units],
                              initializer=tf.variance_scaling_initializer(), dtype=real_type))
    # bias vector
    bs.append(tf.get_variable("b1", [hidden_units],
                              initializer=tf.zeros_initializer(), dtype=real_type))
    # graph
    zs.append(zs[0] @ ws[1] + bs[1])  # eq. 3, l=1

    # second hidden layer (index 2) to last (index hidden_layers)
    for l in range(1, hidden_layers):
        ws.append(tf.get_variable("w%d" % (l + 1), [hidden_units, hidden_units],
                                  initializer=tf.variance_scaling_initializer(), dtype=real_type))
        bs.append(tf.get_variable("b%d" % (l + 1), [hidden_units],
                                  initializer=tf.zeros_initializer(), dtype=real_type))
        zs.append(tf.nn.softplus(zs[l]) @ ws[l + 1] + bs[l + 1])  # eq. 3, l=2..L-1

    # output layer (index hidden_layers+1)
    ws.append(tf.get_variable("w" + str(hidden_layers + 1), [hidden_units, 1],
                              initializer=tf.variance_scaling_initializer(), dtype=real_type))
    bs.append(tf.get_variable("b" + str(hidden_layers + 1), [1],
                              initializer=tf.zeros_initializer(), dtype=real_type))
    # eq. 3, l=L
    zs.append(tf.nn.softplus(zs[hidden_layers]) @ ws[hidden_layers + 1] + bs[hidden_layers + 1])

    # result = output layer
    ys = zs[hidden_layers + 1]

    # return input layer, (parameters = weight matrices and bias vectors),
    # [all layers] and output layer
    return xs, (ws, bs), zs, ys


# compute d_output/d_inputs by (explicit) backprop in vanilla net
def backprop(
        weights_and_biases,  # 2nd output from vanilla_net()
        zs):  # 3rd output from vanilla_net()

    ws, bs = weights_and_biases
    L = len(zs) - 1

    # backpropagation, eq. 4, l=L..1
    zbar = tf.ones_like(zs[L])  # zbar_L = 1
    for l in range(L - 1, 0, -1):
        zbar = (zbar @ tf.transpose(ws[l + 1])) * tf.nn.sigmoid(zs[l])  # eq. 4
    # for l=0
    zbar = zbar @ tf.transpose(ws[1])  # eq. 4

    xbar = zbar  # xbar = zbar_0

    # dz[L] / dx
    return xbar




def vanilla_training_graph(input_dim, hidden_units, hidden_layers, seed):

    # net
    inputs, weights_and_biases, layers, predictions = vanilla_net(input_dim, hidden_units, hidden_layers, seed)

    # backprop even though we are not USING differentials for training
    # we still need them to predict derivatives dy_dx
    derivs_predictions = backprop(weights_and_biases, layers)

    # placeholder for labels
    labels = tf.placeholder(shape=[None, 1], dtype=real_type)

    # loss
    loss = tf.losses.mean_squared_error(labels, predictions)

    # optimizer
    learning_rate = tf.placeholder(real_type)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

    # return all necessary
    return inputs, labels, predictions, derivs_predictions, learning_rate, loss, optimizer.minimize(loss)


def diff_training_graph(
        # same as vanilla
        input_dim,
        hidden_units,
        hidden_layers,
        seed,
        # balance relative weight of values and differentials
        # loss = alpha * MSE(values) + beta * MSE(greeks, lambda_j)
        # see online appendix
        alpha,
        beta,
        lambda_j):
    # net, now a twin
    inputs, predictions, derivs_predictions = twin_net(input_dim, hidden_units, hidden_layers, seed)

    # placeholder for labels, now also derivs labels
    labels = tf.placeholder(shape=[None, 1], dtype=real_type)
    derivs_labels = tf.placeholder(shape=[None, derivs_predictions.shape[1]], dtype=real_type)

    # loss, now combined values + derivatives
    loss = alpha * tf.losses.mean_squared_error(labels, predictions) \
           + beta * tf.losses.mean_squared_error(derivs_labels * lambda_j, derivs_predictions * lambda_j)

    # optimizer, as vanilla
    learning_rate = tf.placeholder(real_type)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # return all necessary tensors, including derivatives
    # predictions and labels
    return inputs, labels, derivs_labels, predictions, derivs_predictions, \
        learning_rate, loss, optimizer.minimize(loss)





class Neural_approximator():
    def __init__(self, x_raw, y_raw,
                 dydx_raw=None):  # derivatives labels,

        self.x_raw = x_raw
        self.y_raw = y_raw
        self.dydx_raw = dydx_raw

        # tensorflow logic
        self.graph = None
        self.session = None

    def build_graph(self,
                    differential,  # differential or not
                    lam,  # balance cost between values and derivs
                    hidden_units,
                    hidden_layers,
                    weight_seed):

        # first, deal with tensorflow logic
        if self.session is not None:
            self.session.close()

        self.graph = tf.Graph()

        with self.graph.as_default():

            # build the graph, either vanilla or differential
            self.differential = differential

            if not differential:
                # vanilla

                self.inputs, \
                    self.labels, \
                    self.predictions, \
                    self.derivs_predictions, \
                    self.learning_rate, \
                    self.loss, \
                    self.minimizer \
                    = vanilla_training_graph(self.n, hidden_units, hidden_layers, weight_seed)

            else:
                # differential

                if self.dy_dx is None:
                    raise Exception("No differential labels for differential training graph")

                self.alpha = 1.0 / (1.0 + lam * self.n)
                self.beta = 1.0 - self.alpha

                self.inputs, \
                    self.labels, \
                    self.derivs_labels, \
                    self.predictions, \
                    self.derivs_predictions, \
                    self.learning_rate, \
                    self.loss, \
                    self.minimizer = diff_training_graph(self.n, hidden_units, \
                                                         hidden_layers, weight_seed, \
                                                         self.alpha, self.beta, self.lambda_j)

            # global initializer
            self.initializer = tf.global_variables_initializer()

        # done
        self.graph.finalize()
        self.session = tf.Session(graph=self.graph)

        # prepare for training with m examples, standard or differential
        def prepare(self,
                    m,
                    differential,
                    lam=1,  # balance cost between values and derivs
                    # standard architecture
                    hidden_units=20,
                    hidden_layers=4,
                    weight_seed=None):

            # prepare dataset
            self.x_mean, self.x_std, self.x, self.y_mean, self.y_std, self.y, self.dy_dx, self.lambda_j = \
                normalize_data(self.x_raw, self.y_raw, self.dydx_raw, m)

            # build graph
            self.m, self.n = self.x.shape
            self.build_graph(differential, lam, hidden_units, hidden_layers, weight_seed)

        def train(self,
                  description="training",
                  # training params
                  reinit=True,
                  epochs=100,
                  # one-cycle learning rate schedule
                  learning_rate_schedule=[
                      (0.0, 1.0e-8),
                      (0.2, 0.1),
                      (0.6, 0.01),
                      (0.9, 1.0e-6),
                      (1.0, 1.0e-8)],
                  batches_per_epoch=16,
                  min_batch_size=256,
                  # callback and when to call it
                  # we don't use callbacks, but this is very useful, e.g. for debugging
                  callback=None,  # arbitrary callable
                  callback_epochs=[]):  # call after what epochs, e.g. [5, 20]

            train(description,
                  self,
                  reinit,
                  epochs,
                  learning_rate_schedule,
                  batches_per_epoch,
                  min_batch_size,
                  callback,
                  callback_epochs)

        def predict_values(self, x):
                # scale
                x_scaled = (x - self.x_mean) / self.x_std
                # predict scaled
                y_scaled = self.session.run(self.predictions, feed_dict={self.inputs: x_scaled})
                # unscale
                y = self.y_mean + self.y_std * y_scaled
                return y

        def predict_values_and_derivs(self, x):
                # scale
                x_scaled = (x - self.x_mean) / self.x_std
                # predict scaled
                y_scaled, dyscaled_dxscaled = self.session.run(
                    [self.predictions, self.derivs_predictions],
                    feed_dict={self.inputs: x_scaled})
                # unscale
                y = self.y_mean + self.y_std * y_scaled
                dydx = self.y_std / self.x_std * dyscaled_dxscaled
                return y, dydx