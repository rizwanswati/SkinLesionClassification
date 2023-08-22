"""
    Stratification of data into test train validation sets.
    using test_train_split method with stratify parameter
    Dated : @8/20/2023
    Author : @Mahnoor Khan
"""

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pandas as pd
from PIL import Image, ImageFilter


def saveimages(test_images, original_image_path, test_dir):
    names = test_images.DDI_file
    for file_name in names:
        full_image_path = original_image_path + "/" + file_name
        image_file = Image.open(full_image_path);
        image_file.save(test_dir + "/" + file_name)


def stratify(filePath):
    image_path = "E:/DDI/DDI_Clean_Output/DDI_original/"
    testing = "D:/MS Thesis/SkinLesionDetectionCode/Testing"
    training = "D:/MS Thesis/SkinLesionDetectionCode/Training"
    validation = "D:/MS Thesis/SkinLesionDetectionCode/Validation"

    data = pd.read_csv(filePath)
    X = data.drop("malignant", axis=1)
    y = data["malignant"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
    X_train_actual, X_validation, y_train_actual, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

    saveimages(X_test, image_path, testing)
    saveimages(X_validation, image_path, validation)
    saveimages(X_train_actual, image_path, training)

def main():
    file_path = 'E:/DDI/ddi_metadata.csv'
    stratify(file_path)


if __name__ == '__main__':
    main()
