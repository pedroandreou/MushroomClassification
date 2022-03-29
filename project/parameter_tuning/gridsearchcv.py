from sklearn.model_selection import GridSearchCV
import numpy as np


def tuning(model, train_features, train_target, test_features, test_target):
    # get the dictionary of parameters from config file
    file="file.config"
    contents=open(file).read()
    parameters=eval(contents)


    clf = GridSearchCV(model,  # model
                       param_grid=parameters,  # hyperparameters
                       scoring='accuracy',  # metric for scoring
                       cv=10)  # number of folds

    # train the model with the training_model set
    clf.fit(train_features, train_target)

    # print the tuned-hyperparameters and the accuracy of the tuned model
    print("Tuned Hyperparameters :", clf.best_params_)
    print("Accuracy using GridSearchCV:", clf.best_score_)
