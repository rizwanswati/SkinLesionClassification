import os
import cv2
import splitfolders
import numpy as np
import pandas as pd
from skimage.exposure import exposure
from sklearn import svm
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import joblib


# Define Constants


# Function to preprocess and extract HOG features from an image
def preprocess_and_extract_features(image_path, IMAGE_SIZE):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read the image in color
    image = cv2.resize(image, IMAGE_SIZE)  # Resize to desired size
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features


# Function to generate segmentation mask (simple threshold-based approach)
def generate_segmentation_mask(image_path_segmentation, IMAGE_SIZE):
    image = cv2.imread(image_path_segmentation, cv2.IMREAD_COLOR)
    image = cv2.resize(image, IMAGE_SIZE)

    # converting to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # applying Otsu thresholding
    # as an extra flag in binary thresholding
    ret, thresh1 = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU)

    return thresh1


def split_into_test_train():
    image_path = "D:/SkinLesionClassification/SegmentationMasks"
    print((os.listdir(image_path)))
    splitfolders.ratio(image_path, seed=1337, output="DDI-Split", ratio=(0.8, 0.2))


def main():

    # split_into_test_train()

    # Load and preprocess the dataset, including segmentation mask generation
    IMAGE_SIZE = (224, 224)
    dataset = "D:/SkinLesionClassification/DDI_original"
    dataset_path = dataset
    data = []
    labels = []
    masks = []  # Store segmentation masks

    class_names = ["benign", "malignant"]

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            features = preprocess_and_extract_features(image_path, IMAGE_SIZE)
            segmentation_mask = generate_segmentation_mask(image_path, IMAGE_SIZE)
            data.append(features)
            labels.append(class_idx)
            masks.append(segmentation_mask)
            cv2.imwrite("D:/SkinLesionClassification/SegmentationMasks/" + image_name, segmentation_mask,
                       [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    """ for masks_batch in masks:
        for i in range(12):
            ax = plt.subplot(3, 4, i + 1)
            plt.imshow(masks_batch[i].numpy().astype("uint8"))
            plt.axis("off")"""


if __name__ == '__main__':
    main()
