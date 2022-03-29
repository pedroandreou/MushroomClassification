import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# cross validation using KFold
def kfold(model, features, target):
    cv = KFold(n_splits=10, random_state=10, shuffle=True)

    scores = cross_val_score(model, features, target, scoring="average_precision", cv=10)

    print("Cross validation scores:", scores, "\n")
    print("Average accuracy of Cross validation scores: {}".format(scores.mean()))

    # plot boxplot of cross-validation scores
    sns.boxplot(x=scores)
