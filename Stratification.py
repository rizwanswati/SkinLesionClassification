"""
    Stratification of data into test train validation sets.
    using test_train_split method with stratify parameter
    Dated : @8/20/2023
    Author : @Mahnoor Khan
"""

from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd


def stratify(filePath):
    data = pd.read_csv(filePath)
    X = data.drop("malignant", axis=1)
    y = data["malignant"]
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

    # Split the data into training and test sets
    for train_index, test_index in split.split(X, y):
        X_train = X.loc[train_index]
        X_test = X.loc[test_index]
        y_train = y.loc[train_index]
        y_test = y.loc[test_index]



    # Split the data into training and test sets
    for train_index, test_index in split.split(X_train, y_train):
        X_train = X.loc[train_index]
        X_validation = X.loc[test_index]
        y_train = y.loc[train_index]
        y_validation = y.loc[test_index]


def main():
    file_path = 'E:/DDI/ddi_metadata.csv'
    stratify(file_path)


if __name__ == '__main__':
    main()