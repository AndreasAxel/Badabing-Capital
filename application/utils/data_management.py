from sklearn.model_selection import train_test_split

def data_preprocessing(data, X_feature_dim=1, y_feature_dim=1, compute_z=False):
    # data structure needs to be of the form
    # X inputs | y outputs | z differentials
    # the function cannot handle multiple differentials so far, but a further fix to do so will be added

    # ensure labels has proper dimensionality
    X = data[:, (X_feature_dim - 1)].reshape(-1, X_feature_dim)
    y = data[:, X_feature_dim:].reshape(-1, y_feature_dim + 1) if compute_z else data[:, 1:].reshape(-1, y_feature_dim)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    if compute_z:
        z_train = y_train[:, y_feature_dim:].reshape(-1, y_feature_dim)
        y_train = y_train[:, 0].reshape(-1, 1)

        z_test = y_test[:, y_feature_dim:].reshape(-1, y_feature_dim)
        y_test = y_test[:, 0].reshape(-1, 1)

        return X_train, y_train, z_train, X_test, y_test, z_test

    return X_train, y_train, X_test, y_test
