import seaborn as sns
import matplotlib.pyplot as plt


def check_balance(df):
    # countplot of every variable with hue = class
    for i, col in enumerate(df.columns):
        plt.figure(i)  # create a figure for each feature of the dataset

        # plot the figure for the specific feature and its corresponding values hue parameter takes as argument the
        # class feature for color encoding each unique value of each feature based on the class
        sns_plot = sns.countplot(x=col, hue='class', data=df)

    # print the values of for each unique value of the class to see if it is balanced numerically
    print(df['class'].value_counts())
