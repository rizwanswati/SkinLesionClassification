"""
    This file is used to split the training images into two folders, Benign, Malignant,
    that are further divided into three folders i.e. Black, white, brown.
"""
import glob
import shutil
import csv


def save_to_folder(image_name, csv_file, output_benign, output_malignant, training_folder):
    csv_file = csv.reader(open(csv_file, "r"), delimiter=",")
    for row in csv_file:
        if row[2] == image_name and row[3] == '56' and row[4] == 'TRUE':
            source_file_path = training_folder + "/" + image_name
            shutil.copy(source_file_path, output_malignant + "Black")
        if row[2] == image_name and row[3] == '56' and row[4] == 'FALSE':
            source_file_path = training_folder + "/" + image_name
            shutil.copy(source_file_path, output_benign + "Black")
        if row[2] == image_name and row[3] == '34' and row[4] == 'TRUE':
            source_file_path = training_folder + "/" + image_name
            shutil.copy(source_file_path, output_malignant + "Brown")
        if row[2] == image_name and row[3] == '34' and row[4] == 'FALSE':
            source_file_path = training_folder + "/" + image_name
            shutil.copy(source_file_path, output_benign + "Brown")
        if row[2] == image_name and row[3] == '12' and row[4] == 'TRUE':
            source_file_path = training_folder + "/" + image_name
            shutil.copy(source_file_path, output_malignant + "White")
        if row[2] == image_name and row[3] == '12' and row[4] == 'FALSE':
            source_file_path = training_folder + "/" + image_name
            shutil.copy(source_file_path, output_benign + "White")


def save_test_valid_images_to_folder(image_name, csv_file, output_benign, output_malignant, folder):
    csv_file = csv.reader(open(csv_file, "r"), delimiter=",")
    for row in csv_file:
        if row[2] == image_name and row[4] == 'TRUE':
            source_file_path = folder + "/" + image_name
            shutil.move(source_file_path, output_malignant)
        if row[2] == image_name and row[4] == 'FALSE':
            source_file_path = folder + "/" + image_name
            shutil.move(source_file_path, output_benign)


def split_images(training_image_folder, output_benign, out_malignant, class_file_path, training_folder):
    for filePath in glob.glob(training_image_folder):
        filename = filePath.partition("\\")[2]
        save_to_folder(filename, class_file_path, output_benign, out_malignant, training_folder)


def split_validation_test_images(input_image_folder, output_benign, output_malignant, class_file_path, output_folder):
    for filePath in glob.glob(input_image_folder):
        filename = filePath.partition("\\")[2]
        save_test_valid_images_to_folder(filename, class_file_path, output_benign, output_malignant, output_folder)


def main():
    output_path_benign = "D:/SkinLesionClassification/Augmentation/Benign/"
    output_path_malignant = "D:/SkinLesionClassification/Augmentation/Malignant/"
    valid_benign = "D:/SkinLesionClassification/Validation/benign"
    valid_malignant = "D:/SkinLesionClassification/Validation/malignant"
    test_benign = "D:/SkinLesionClassification/Testing/benign"
    test_malignant = "D:/SkinLesionClassification/Testing/malignant"
    training_folder_path = "D:/SkinLesionClassification/Training/*.png"
    training_folder = "D:/SkinLesionClassification/Training/"
    valid_folder_path = "D:/SkinLesionClassification/Validation/*.png"
    valid_folder = "D:/SkinLesionClassification/Validation/"
    test_folder_path = "D:/SkinLesionClassification/Testing/*.png"
    test_folder = "D:/SkinLesionClassification/Testing/"
    class_file_path = "D:/SkinLesionClassification/ddi_metadata.csv"

    split_images(training_folder_path, output_path_benign, output_path_malignant, class_file_path, training_folder)
    split_validation_test_images(valid_folder_path, valid_benign, valid_malignant, class_file_path, valid_folder)
    split_validation_test_images(test_folder_path, test_benign, test_malignant, class_file_path, test_folder)


if __name__ == "__main__":
    main()
