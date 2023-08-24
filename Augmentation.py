"""
    Augmentation of images
    Generate multiple images from one image via changing different parameters
    using keras preprocessing image
    Dated : @8/16/2023
    Author : @Mahnoor Khan
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import load_img, img_to_array
import glob
import shutil
import csv


def augment(folderPath, outputPath, iteration):
    data_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.01, horizontal_flip=True,
                                  vertical_flip=True)
    # now taking image one by one and converting it to tensor
    for filePath in glob.glob(folderPath):
        srcImg = load_img(filePath, grayscale=False)
        imgArray = img_to_array(srcImg)
        filename = filePath.partition("\\")[2]
        tensorImage = imgArray.reshape((1,) + imgArray.shape)

        for i, _ in enumerate(data_gen.flow(x=tensorImage,
                                            batch_size=1,
                                            save_to_dir=outputPath,
                                            save_prefix=filename + "_",
                                            save_format=".png")):
            if i >= iteration:
                break


def save_to_folder(image_name, csv_file, output_benign, output_malignant, subfolder_path):
    csv_file = csv.reader(open(csv_file, "r"), delimiter=",")
    for row in csv_file:
        if row[2] == image_name and row[3] == '56' and row[4] == 'TRUE':
            source_file_path = subfolder_path + "/" + image_name
            shutil.copy(source_file_path, output_malignant)
        if row[2] == image_name and row[3] == '56' and row[4] == 'FALSE':
            source_file_path = subfolder_path + "/" + image_name
            shutil.copy(source_file_path, output_benign)
        if row[2] == image_name and row[3] == '34' and row[4] == 'TRUE':
            source_file_path = subfolder_path + "/" + image_name
            shutil.copy(source_file_path, output_malignant)
        if row[2] == image_name and row[3] == '34' and row[4] == 'FALSE':
            source_file_path = subfolder_path + "/" + image_name
            shutil.copy(source_file_path, output_benign)
        if row[2] == image_name and row[3] == '12' and row[4] == 'TRUE':
            source_file_path = subfolder_path + "/" + image_name
            shutil.copy(source_file_path, output_malignant)
        if row[2] == image_name and row[3] == '12' and row[4] == 'FALSE':
            source_file_path = subfolder_path + "/" + image_name
            shutil.copy(source_file_path, output_benign)


def copy_original_to_augmentation_benign(benign, csv_path, output_benign, output_malignant):
    for benign_subfolder in benign:
        for filePath in glob.glob(benign_subfolder+"/*.png"):
            filename = filePath.partition("\\")[2]
            save_to_folder(filename, csv_path, output_benign, output_malignant, benign_subfolder)


def copy_original_to_augmentation_malignant(malignant, csv_path, output_benign, output_malignant):
    for malignant_subfolder in malignant:
        for filePath in glob.glob(malignant_subfolder+"/*.png"):
            filename = filePath.partition("\\")[2]
            save_to_folder(filename, csv_path, output_benign, output_malignant, malignant_subfolder)


def main():
    folder_path_benign = [
        "D:/SkinLesionClassification/Augmentation/Benign/Black/*.png",
        "D:/SkinLesionClassification/Augmentation/Benign/Brown/*.png",
        "D:/SkinLesionClassification/Augmentation/Benign/White/*.png"
    ]

    folder_path_malignant = [
        "D:/SkinLesionClassification/Augmentation/Malignant/Black/*.png",
        "D:/SkinLesionClassification/Augmentation/Malignant/Brown/*.png",
        "D:/SkinLesionClassification/Augmentation/Malignant/White/*.png"
    ]

    folder_benign = [
        "D:/SkinLesionClassification/Augmentation/Benign/Black",
        "D:/SkinLesionClassification/Augmentation/Benign/Brown",
        "D:/SkinLesionClassification/Augmentation/Benign/White"
    ]

    folder_malignant = [
        "D:/SkinLesionClassification/Augmentation/Malignant/Black",
        "D:/SkinLesionClassification/Augmentation/Malignant/Brown",
        "D:/SkinLesionClassification/Augmentation/Malignant/White"
    ]

    output_path_benign = "D:/SkinLesionClassification/TrainingDS/benign/"
    output_path_malignant = "D:/SkinLesionClassification/TrainingDS/malignant/"
    csv_path = "D:/SkinLesionClassification/ddi_metadata.csv"

    for folderPath in folder_path_benign:
        augment(folderPath, output_path_benign, 1)

    for folderPath in folder_path_malignant:
        augment(folderPath, output_path_malignant, 5)

    copy_original_to_augmentation_malignant(folder_malignant, csv_path, output_path_benign, output_path_malignant)
    copy_original_to_augmentation_benign(folder_benign, csv_path, output_path_benign, output_path_malignant)


if __name__ == '__main__':
    main()
