"""
    Stratification of data into test train validation sets.
    using test_train_split method with stratify parameter
    Dated : @8/20/2023
    Author : @Mahnoor Khan

    image_path = "E:/DDI/DDI_Clean_Output/DDI_original/"
    testing = "D:/MS Thesis/SkinLesionDetectionCode/Testing"
    training = "D:/MS Thesis/SkinLesionDetectionCode/Training"
    validation = "D:/MS Thesis/SkinLesionDetectionCode/Validation"
    file_path = 'E:/DDI/ddi_metadata.csv'
"""

from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from PIL import Image, ImageFilter


def saveimages(test_images, original_image_path, test_dir):
    names = test_images.DDI_file
    for file_name in names:
        full_image_path = original_image_path + "/" + file_name
        image_file = Image.open(full_image_path);
        image_file.save(test_dir + "/" + file_name)


def stratify(filePath):
    image_path = "D:/SkinLesionClassification/DDI_Clean_Output/MedianFilteredImages"
    testing = "D:/SkinLesionClassification/Testing"
    training = "D:/SkinLesionClassification/Training"
    validation = "D:/SkinLesionClassification/Validation"

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

    saveimages(X_test, image_path, testing)
    # Split the data into training and validation
    for train_index, test_index in split.split(X_train, y_train):
        X_train = X.loc[train_index]
        X_validation = X.loc[test_index]
        y_train = y.loc[train_index]
        y_validation = y.loc[test_index]

    saveimages(X_validation, image_path, validation)
    saveimages(X_train, image_path, training)

def main():
    file_path = 'D:/SkinLesionClassification/ddi_metadata.csv'
    stratify(file_path)


if __name__ == '__main__':
    main()
