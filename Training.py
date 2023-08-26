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
# from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from keras.optimizers import Adam



def print_data(data):
    print(data)
    print(data.params)
    print(data.history.keys())
    print(data.history['loss'][:10])  # loss of first 10 epochs
    print(data.history['accuracy'])
    print(data.history['val_accuracy'])
    print(data.history['loss'])
    print(data.history['val_loss'])


def initialize_training_dataset(data_path, IMAGE_SIZE, BATCH_SIZE):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(data_path, labels="inferred", seed=123, shuffle=True,
                                                                  image_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                  batch_size=BATCH_SIZE
                                                                  )
    return dataset


def initialize_validation_dataset(data_path, IMAGE_SIZE, BATCH_SIZE):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(data_path, labels="inferred", seed=123, shuffle=True,
                                                                  image_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                  batch_size=BATCH_SIZE
                                                                  )
    return dataset


def initialize_testing_dataset(data_path, IMAGE_SIZE, BATCH_SIZE):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(data_path, labels="inferred", seed=123, shuffle=True,
                                                                  image_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                  batch_size=BATCH_SIZE
                                                                  )
    return dataset


def print_image_label_batch(dataset):
    for image_batch, label_batch in dataset:
        print(image_batch.shape)
        print(label_batch.numpy())


def visualize_images(dataset, class_names):
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


def build_model(trainDS, input_shape, no_classes, batch_size, validationDS, epochs, IMAGE_SIZE):
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.Rescaling(1. / 244),
    ])

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])
    train_dataset = trainDS.map(
        lambda x, y: (data_augmentation(x, training=True), y)
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape[1:]),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(no_classes, activation='softmax'),
    ])
    model.build(input_shape=input_shape)
    model.summary()
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    history = model.fit(
        train_dataset,
        batch_size=batch_size,
        validation_data=validationDS,
        verbose="auto",
        epochs=epochs,
        use_multiprocessing= True

    )
    return history


def shuffle_training_data(trainDS):
    return trainDS.shuffle(10000, seed=100)


def main():
    BATCH_SIZE = 32
    IMAGE_SIZE = 244
    CHANNELS = 3
    EPOCHS = 49

    dataset_path = "D:/SkinLesionClassification/TrainingDS"
    testing_dataset_path = "D:/SkinLesionClassification/Testing"
    validation_dataset_path = "D:/SkinLesionClassification/Validation/"

    training_dataset = initialize_training_dataset(dataset_path, IMAGE_SIZE, BATCH_SIZE)
    testing_dataset = initialize_testing_dataset(testing_dataset_path, IMAGE_SIZE, BATCH_SIZE)
    validation_dataset = initialize_validation_dataset(validation_dataset_path, IMAGE_SIZE, BATCH_SIZE)

    print_image_label_batch(training_dataset)
    visualize_images(training_dataset, training_dataset.class_names)
    essential_info(training_dataset, testing_dataset, validation_dataset)

    trainDS, testDS, validationDS = cache_shuffle_prefetch(training_dataset, testing_dataset, validation_dataset)
    trainDS = shuffle_training_data(trainDS)

    input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    classes = 2
    history = build_model(trainDS, input_shape, classes, BATCH_SIZE, validationDS, EPOCHS, IMAGE_SIZE)
    print_data(history)


if __name__ == "__main__":
    main()
