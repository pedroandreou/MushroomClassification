def missing_values(df):
    features = df.columns  # get the columns of the provided df

    # loop through the columns
    for i in features:
        print(df[i].value_counts(dropna=False), "\n")  # display the NaN values


def check_for_missing_data(df):
    print(df.isnull().values.any())  # print if the entire dataset includes any missing data

    print("Missing values for each column:\n", df.isnull().sum())  # print the total count of missing values of each
    # column

    missing_values(df)
