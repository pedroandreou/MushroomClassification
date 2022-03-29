from sklearn.linear_model import LogisticRegression


def train_model(model, train_features, train_target, test_features):
    # Fitting Logistic Regression
    model.fit(train_features, train_target)

    test_target_prediction = model.predict(test_features)  # predict class labels for samples in features of test set

    return test_target_prediction
