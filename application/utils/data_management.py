from sklearn.model_selection import train_test_split

def data_preprocessing(data, X_feature_dim=1, y_feature_dim=1, compute_z=False, deriv_dim = 1):
    # data structure needs to be of the form
    # X inputs | y outputs/prices | z differentials/sensitivities
    # the function cannot handle multiple differentials so far, but a further fix to do so will be added

    # deriv_dim is the number of derivs needs to be computed, i.e. if delta and gamma is inputs then deriv_dim = 2

    # ensure labels has proper dimensionality
    X = data[:, (X_feature_dim - 1)].reshape(-1, X_feature_dim)
    y = data[:, X_feature_dim:].reshape(-1, y_feature_dim + deriv_dim) if compute_z else data[:, 1:].reshape(-1, y_feature_dim)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    if compute_z:
        z_train = y_train[:, y_feature_dim:].reshape(-1, deriv_dim)
        y_train = y_train[:, 0].reshape(-1, y_feature_dim)

        z_test = y_test[:, y_feature_dim:].reshape(-1, deriv_dim)
        y_test = y_test[:, 0].reshape(-1, y_feature_dim)

        return X_train, y_train, z_train, X_test, y_test, z_test

    return X_train, y_train, X_test, y_test
