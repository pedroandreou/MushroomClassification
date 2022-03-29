from sklearn.preprocessing import StandardScaler


def standardisation(train_features, test_features):
    sc = StandardScaler()
    train_features = sc.fit_transform(train_features)
    test_features = sc.transform(test_features)

    return train_features, test_features
