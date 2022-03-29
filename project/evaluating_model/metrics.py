import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def accuracy(test_target, test_target_prediction):
    # accuracy
    print("Accuracy score:", accuracy_score(test_target, test_target_prediction, normalize=True) * 100)


def conf_matrix(test_target, test_target_prediction):
    # confusion matrix
    # Visualising Confusion Matrix
    cm = confusion_matrix(test_target, test_target_prediction)
    sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', cbar=False, yticklabels=['Edible', 'Poisonous'],
                xticklabels=['Predicted Edible', 'Predicted Poisonous'])
    plt.show()
