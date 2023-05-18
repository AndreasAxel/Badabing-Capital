from sklearn.model_selection import train_test_split

def data_preprocessing(data, X_feature_dim=1, y_feature_dim=1, compute_z=False):
    # ensure labels has proper dimensionality
    X = data[:, (X_feature_dim - 1)].reshape(-1, X_feature_dim)
    y = data[:, X_feature_dim:].reshape(-1, y_feature_dim + 1) if compute_z else data[:, 1:].reshape(-1, y_feature_dim)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    if compute_z:
        z_train = y_train[:, 0].reshape(-1, 1)
        y_train = y_train[:, y_feature_dim:].reshape(-1, y_feature_dim)
        z_test = y_test[:, 0].reshape(-1, 1)
        y_test = y_test[:, y_feature_dim:].reshape(-1, y_feature_dim)

        return X_train, z_train, y_train, X_test, z_test, y_test

    return X_train, y_train, X_test, y_test
