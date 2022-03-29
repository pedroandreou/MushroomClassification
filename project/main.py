from preprocessing_data import check_dataset_balance, missing_data, understand_data, visualise_data, \
    encode_data, split_data, feature_scaling
from training_model import model
from parameter_tuning import gridsearchcv, cross_validation
from evaluating_model import metrics

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


data_file = "C:/Users/user/Desktop/artificial-intelligence-cw/project/collecting_data/mushrooms.csv"

df = pd.read_csv(data_file)

# preprocessing_data

# understand data
understand_data.understand_data(df)
# check for missing values within the dataset
missing_data.check_for_missing_data(df)
# visualise the data of each column of the dataset
visualise_data.data_visualisation(df)
# check if the dataset is balanced by visualising the label col and see the balance numerically
check_dataset_balance.check_balance(df)
# encode variables
encode_data.encode_categorical_vars(df)
# separate features and label
features, target = split_data.separate_features_label(df)
# split dataset into train and test sets
train_features, test_features, train_target, test_target = split_data.data_splitting_train_test_sets(features, target)
# scale the train and test features
train_features, test_features = feature_scaling.standardisation(train_features, test_features)

# choose model
# logistic regression

lr = LogisticRegression()

# train model - logistic regression

# get the prediction of the target of the test set
test_target_prediction = model.train_model(lr, train_features, train_target, test_features)

# evaluate model -logistic regression

# accuracy score for logistic regression model
metrics.accuracy(test_target, test_target_prediction)
# confusion matrix for logistic regression model
metrics.conf_matrix(test_target, test_target_prediction)

# cross validation

# train logistic regression model again - this time with cross validation using KFold
cross_validation.kfold(lr, features, target)

# GridSearchCV

gridsearchcv.tuning(lr, train_features, train_target, test_features, test_target)


# Pipeline

pipe = Pipeline([('scaler', StandardScaler()), ('model', lr)])
pipe.fit(train_features, train_target)
    
# print Pipeline score
print("Pipeline's accuracy score: ", pipe.score(test_features, test_target))