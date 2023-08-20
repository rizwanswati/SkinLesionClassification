"""
This file is used to split the training images into two folders, Benign, Malignant, that are further divided into three folders
three folders i.e. Black, white, brown.

"""
import cv2
import glob

def split_images(training_image_folder, output_benign, out_malignant):
    for filePath in glob.glob(training_image_folder):
        filename = filePath.partition("\\")[2]


def main():
    output_path_benign = "D:/MS Thesis/SkinLesionDetectionCode/Benign/"
    output_path_malignant = "D:/MS Thesis/SkinLesionDetectionCode/Malignant/"
    training_folder_path = "D:/MS Thesis/SkinLesionDetectionCode/Training"

    split_images(training_folder_path, output_path_benign, output_path_malignant)