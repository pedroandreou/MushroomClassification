import seaborn as sns
import matplotlib.pyplot as plt


def data_visualisation(df):
    # countplot of every variable
    for i, col in enumerate(df.columns):
        plt.figure(i)  # create a figure for each feature of the dataset

        # plot the figure for the specific feature and its corresponding values
        sns.countplot(x=col, data=df)
