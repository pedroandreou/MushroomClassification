from sklearn.model_selection import train_test_split


def separate_features_label(df):
    features = df.iloc[:, 1:]  # all rows, all the features and no labels
    target = df.iloc[:, 0]  # all rows, label only

    # see if the information we want is in place
    # print(features.head())
    # print(target.head())

    # print(features.shape)
    # print(target.shape)

    return features, target


def data_splitting_train_test_sets(features, target):
    # Training set and Test set
    train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2,
                                                                                random_state=0)

    print("Shape of features in the train dataset: ", train_features.shape)
    print("Shape of target in the train dataset: ", train_target.shape)
    print("Shape of features in the test dataset: ", test_features.shape)
    print("Shape of target in the test dataset: ", test_target.shape)

    return train_features, test_features, train_target, test_target
