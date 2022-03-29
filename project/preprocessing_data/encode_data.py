from sklearn.preprocessing import LabelEncoder


def encode_categorical_vars(df):
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])

    # check that all the encoded values are different between them
    # df['population'].unique()

    # check the size of the encoded values of the feature class¶
    # print(df.groupby('class').size())  # 0 and 1
