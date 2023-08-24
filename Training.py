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


def initialize_training_dataset(data_path, IMAGE_SIZE, BATCH_SIZE):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(data_path, seed=123, shuffle=True,
                                                                  image_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                  batch_size=BATCH_SIZE
                                                                  )
    return dataset


def initialize_dataset(data_path, IMAGE_SIZE, BATCH_SIZE):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(data_path, labels=None, seed=123, shuffle=True,
                                                                  image_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                  batch_size=BATCH_SIZE
                                                                  )
    return dataset


def print_image_label_batch(dataset):
    for image_batch, label_batch in dataset:
        print(image_batch.shape)
        print(label_batch.numpy())


def visiualize_images(dataset, class_names):
    for image_batch, labels_batch in dataset.take(1):
        for i in range(12):
            ax = plt.subplot(3, 4, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[labels_batch[i]])
            plt.axis("off")


def essential_info(trainingDS, testingDS, validationDS):
    print(trainingDS.class_names)
    print(testingDS.class_names)
    print(validationDS.class_names)

    print(len(trainingDS))
    print(len(testingDS))
    print(len(validationDS))


def cache_shuffle_prefetch(trainDS, testDS, validationDS):
    train_ds = trainDS.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = testDS.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = validationDS.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds, val_ds,


def main():
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    CHANNELS = 3
    EPOCHS = 100

    dataset_path = "D:/SkinLesionClassification/TrainingDS"
    testing_dataset_path = "D:/SkinLesionClassification/Testing/"
    validation_dataset_path = "D:/SkinLesionClassification/Validation/"

    training_dataset = initialize_training_dataset(dataset_path, IMAGE_SIZE, BATCH_SIZE)
    testing_dataset = initialize_dataset(testing_dataset_path, IMAGE_SIZE, BATCH_SIZE)
    validation_dataset = initialize_dataset(validation_dataset_path, IMAGE_SIZE, BATCH_SIZE)

    print_image_label_batch(training_dataset)
    visiualize_images(training_dataset, training_dataset.class_names)
    essential_info(training_dataset, testing_dataset, validation_dataset)

    trainDS, testDS, validationDS = cache_shuffle_prefetch(training_dataset, testing_dataset, validation_dataset)


if __name__ == "__main__":
    main()
