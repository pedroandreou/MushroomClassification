import pandas as pd


# lab3 task2 ex1
def unique_values(df):
    features = df.columns

    print("\033[1m" + "Unique Values of each feature:" + "\033[0m" + "\n")

    for i in features:
        print(i, ":", len(df[i].unique()))


# lab3 task2 ex2
def probability_distribution(df):
    features = df.columns  # get the columns of the provided df

    # loop through the columns
    for i in features:
        sum = 0
        vals = df[i].unique()  # get the unique values of that column/feature

        # loop through the unique values of that specific column
        # and find the sum of all the column's values
        for j in vals:
            amount = pd.value_counts(df[i]).loc[j]  # get the amount of times that feature's unique value appears
            # print("The ", j, " unique value of the feature ", i, " has appeared ", count ," times")
            sum = sum + amount

        # for printing all the possible outcomes of the specific feature
        print("\n\n\n\n{} = {{".format(i), end="")

        # exclude the last value for not printing ';'
        for j in vals[:-1]:
            print("{}; ".format(j), end="")

        # print the last value
        print("{}}}".format(vals[-1]), end="")

        # -------------------------------------

        # for printing the probabilities
        print("\n\nP({}) = {{".format(i), end="")
        for j in vals[:-1]:
            amount = pd.value_counts(df[i]).loc[j]
            probability = amount / sum
            print("{}; ".format(probability), end="")

        # exclude the last value for not printing ';'
        amount = pd.value_counts(df[i]).loc[vals[-1]]
        probability = amount / sum
        print("{}}}".format(probability), end="")


# lab3 task2 ex3
def expected_value(df):
    features = df.columns  # get the columns of the provided df

    # loop through the columns
    for i in features:
        sum = 0
        max = 0
        vals = df[i].unique()  # get the unique values

        # loop through the unique values of that specific column
        # and find the sum of all the column's values
        for j in vals:
            amount = pd.value_counts(df[i]).loc[j]  # get the amount of values of that specific feature's unique value
            # print("The ", j, " unique value of the feature ", i, " has appeared ", count ," times")
            sum = sum + amount

        for j in vals:
            amount = pd.value_counts(df[i]).loc[j]  # get the amount of times that feature's unique value appears
            probability = amount / sum

            if probability > max:
                max = probability
                feature_name = i
                valuename_of_feature = j

        print("E[{0}] = {1}".format(feature_name, valuename_of_feature))


def understand_data(df):
    print(df.head())  # quick look

    # target is only column
    print(df['class'].unique())  # see the unique values of target

    print(df.shape)  # see instances and total columns - one target = features

    print(df.dtypes)  # check the datatype of each column

    print(df.describe(include=[object]))  # print statistics summary of the categorical data

    unique_values(df)  # print unique values of each column

    probability_distribution(df)  # print the prob distribution of each column

    expected_value(df)  # print the expected value for each column
