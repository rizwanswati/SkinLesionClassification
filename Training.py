"""
    This file is being made to train a deep learning model on DDI-dataset.
    The dataset has been pre-processed, split into train-test-valid sets using stratification
    and then augmented. This augmented data is being used to train the model.
    @:var EPOCHS_SIZE =
    @:var BATCH_SIZE =
    @:var NO_OF_LAYERS =

    Creation date: August 22, 2023
    Author: @Mahnoor Khan
"""

import tensorflow as tf
from keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score


def initialize_data(data_path, IMAGE_SIZE, BATCH_SIZE):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(data_path, seed=123, shuffle=True,
                                                                  image_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                  batch_size=BATCH_SIZE
                                                                  )
    return dataset


def main():
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    CHANNELS = 3
    EPOCHS = 100

    dataset_path = "D:/SkinLesionClassification/TrainingDS"
    dataset = initialize_data(dataset_path, IMAGE_SIZE, BATCH_SIZE)
    print(dataset.class_names)


if __name__ == "__main__":
    main()
