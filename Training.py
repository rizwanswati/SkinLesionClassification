"""
    This file is being made to train a deep learning model on DDI-dataset.
    The dataset has been pre-processed, split into train-test-valid sets using stratification
    and then augmented. This augmented data is being used to train the model.
    @:var EPOCHS_SIZE =
    @:var BATCH_SIZE =
    @:var NO_OF_LAYERS =
    Optimal model values at EPOCH: 34
    Highest model accuracies at EPOCH: 60/65
    Creation date: August 22, 2023
    Author: @Mahnoor Khan
"""
import h5py
import keras.models
import sklearn.metrics
import tensorflow as tf
from keras import models, layers
# from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import glob

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def print_data(data):
    print(data)
    print(data.params)
    print(data.history.keys())
    print(data.history['loss'][:10])  # loss of first 10 epochs
    print(data.history['accuracy'])
    print(data.history['val_accuracy'])
    print(data.history['loss'])
    print(data.history['val_loss'])


def precision_recall_and_f1score(model, test_ds):
    test_images = []
    test_labels = []
    predicted_labels = []
    class_names = test_ds.class_names

    for images_batch, labels_batch in test_ds:
        test_images.extend(images_batch.numpy())
        test_labels.extend(labels_batch.numpy())
        batch_prediction = model.predict(images_batch)
        predicted_labels.extend(np.argmax(batch_prediction, axis=1))

    test_labels = np.array(test_labels)
    predicted_labels = np.array(predicted_labels)

    # cm = confusion_matrix(test_labels, predicted_labels)
    """plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=55)
    plt.yticks(tick_marks, class_names)"""

    precision = precision_score(test_labels, predicted_labels, average='weighted')
    print("Precision:", precision)
    recall = recall_score(test_labels, predicted_labels, average='weighted')
    print("Recall:", recall)
    f1 = f1_score(test_labels, predicted_labels, average='weighted')
    print("F1-Score:", f1)
    # print("Confusion matrix:", cm)

    # Plotting the metrics
    plt.figure(figsize=(10, 6))
    x = np.arange(1)

    plt.bar(x - 0.2, precision, width=0.2, label='Precision')
    plt.bar(x, recall, width=0.2, label='Recall')
    plt.bar(x + 0.2, f1, width=0.2, label='F1 Score')

    plt.xticks(x, rotation=45)
    plt.xlabel('Skin Lesion Classification')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score')
    plt.legend()
    plt.show()


def roc_auc_plot(model, testDS):
    test_images = []
    test_labels = []
    predicted_probabilities = []

    for images_batch, labels_batch in testDS:
        test_images.extend(images_batch.numpy())
        test_labels.extend(labels_batch.numpy())
        batch_prediction = model.predict(images_batch)
        predicted_probabilities.extend(batch_prediction)

    test_labels = np.array(test_labels)
    predicted_probabilities = np.array(predicted_probabilities)

    # Calculate the ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(test_labels, predicted_probabilities[:])
    auc_score = roc_auc_score(test_labels, predicted_probabilities[:])

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    print('AUC Score:', auc_score)


def PlotData(history, EPOCHS):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), acc, label='Training Accuracy')
    plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xticks([5, 10, 20, 30, 40, 50])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCHS), loss, label='Training Loss')
    plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xticks([5, 10, 20, 30, 40, 50])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title('Training and Validation Loss')
    plt.show()


def prediction_on_sample_image(model, testDS):
    for images_batch, labels_batch in testDS.take(1):
        first_image = images_batch[0].numpy().astype('uint8')
        first_label = labels_batch[0].numpy()

        print("first image to predict")
        plt.imshow(first_image)
        firstLable = testDS.class_names[first_label]
        print("actual label:", firstLable)

        batch_prediction = model.predict(images_batch)
        print("predicted label:", testDS.class_names[np.argmax(batch_prediction[0])])
        plt.axis("off")
        plt.show()


def predict(model, image, classes):
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    prediction = model.predict(image_array)
    predicted_class = classes[np.argmax(prediction[0])]
    confidence = round(100 * (np.max(prediction[0])), 2)
    return predicted_class, confidence


def predict_on_multiple_images(model, test_ds):
    plt.figure(figsize=(15, 15))
    for images, labels in test_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            classes = test_ds.class_names
            predicted_class, confidence = predict(model, images[1].numpy(), classes)
            actual_class = test_ds.class_names[labels[i]]
            plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}. \n Confidence: {confidence}%")
            plt.axis("off")
    plt.show()


def evaluate_show_score_plot(model, testDS, history):
    scores = model.evaluate(testDS)
    # Plotting the Accuracy Score
    plt.figure(figsize=(6, 4))
    plt.bar(["Accuracy"], [scores[1]], color='blue')
    plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
    plt.xlabel("Metrics")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Score")
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(8, 4))
    metrics = ["Training Accuracy", "Validation Accuracy"]
    accuracy_scores = [acc[-1], val_acc[-1]]

    plt.bar(metrics, accuracy_scores, color=['blue', 'orange'])
    plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
    plt.xlabel("Metrics")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Scores")
    plt.show()


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
        # layers.RandomFlip("horizontal_and_vertical"),
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
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax'),
    ])
    model.build(input_shape=input_shape)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'], )
    history = model.fit(
        train_dataset,
        batch_size=batch_size,
        validation_data=validationDS,
        # callbacks=[EarlyStopping(monitor='val_loss', verbose=1, patience=50), ModelCheckpoint(
        #  filepath='D:/SkinLesionClassification/skin_lesion.h5',
        # monitor='val_loss',
        # save_best_only=True)],
        verbose="auto",
        epochs=epochs,
        use_multiprocessing=True

    )
    return history, model


def save_model(model, path):
    model.save(path + "/skin_lesion_detection.h5")
    return path + "/skin_lesion_detection.h5"


def shuffle_training_data(trainDS):
    return trainDS.shuffle(10000, seed=100)


def load_model(path):
    model = tf.keras.models.load_model(path)
    vector = model.get_layer("max_pooling2d_5").output
    feature_extractor = tf.keras.Model(model.input, vector)
    return feature_extractor, vector


def feature_lable_extractor(feature_extractor, data_set_svm, classes):
    X_list = []
    Y_list = []
    for label in classes:
        if label == 'malignant':
            f_lable = 1
            print("malignant 1")
        else:
            f_lable = 0
            print("benign 0")
        folder_path = data_set_svm + "/" + label + "/*.png"
        for filePath in glob.glob(folder_path):
            filename = filePath.partition("\\")[2]
            processed_img = preprocess_image(filePath)
            extracted_feature_vector = feature_extractor.predict(processed_img)
            X_list.append(extracted_feature_vector.ravel())
            Y_list.append(f_lable)

    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)
    return X, Y


def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(244, 244)
    )
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    return image_array


def prepare_build_svm(X, Y, classes):
    for s in range(100):
        X, Y = shuffle(X, Y)

    print("Shape of feature matrix X")
    print(X.shape)
    print("\nShape of label matrix Y")
    print(Y.shape)

    class_types, counts = np.unique(Y, return_counts=True)

    print("\nClass labels")
    print(class_types)
    print("\nClass counts")
    print(counts)

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2,
                                                        stratify=Y,
                                                        random_state=0)

    print("Shape of train_X")
    print(train_X.shape)
    print("\nShape of test_X")
    print(test_X.shape)

    svm_lin = svm.SVC(C=1.0, kernel="linear")
    svm_lin.fit(train_X, train_Y)
    y_pred = svm_lin.predict(test_X)
    print(classification_report(test_Y, y_pred,
                                ))
    print(confusion_matrix(test_Y, y_pred))
    fpr, tpr, thresholds = roc_curve(test_Y, y_pred)
    auc_score = roc_auc_score(test_Y, y_pred)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    print('AUC Score:', auc_score)


def main():
    BATCH_SIZE = 32
    IMAGE_SIZE = 244
    CHANNELS = 3
    EPOCHS = 50
    dataset_path = "D:/SkinLesionClassification/TrainingDS"
    testing_dataset_path = "D:/SkinLesionClassification/Testing"
    validation_dataset_path = "D:/SkinLesionClassification/Validation/"
    model_saving_path = "D:/SkinLesionClassification"

    training_dataset = initialize_training_dataset(dataset_path, IMAGE_SIZE, BATCH_SIZE)
    testing_dataset = initialize_testing_dataset(testing_dataset_path, IMAGE_SIZE, BATCH_SIZE)
    validation_dataset = initialize_validation_dataset(validation_dataset_path, IMAGE_SIZE, BATCH_SIZE)

    """print_image_label_batch(training_dataset)
    visualize_images(training_dataset, training_dataset.class_names)
    essential_info(training_dataset, testing_dataset, validation_dataset)

    trainDS, testDS, validationDS = cache_shuffle_prefetch(training_dataset, testing_dataset, validation_dataset)
    trainDS = shuffle_training_data(trainDS)"""

    input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    classes = 2
    """history, model = build_model(trainDS, input_shape, classes, BATCH_SIZE, validationDS, EPOCHS, IMAGE_SIZE)
    print_data(history)
    PlotData(history, EPOCHS)
    prediction_on_sample_image(model=model, testDS=testing_dataset)
    predict_on_multiple_images(model, testing_dataset)
    evaluate_show_score_plot(model, testing_dataset, history)
    precision_recall_and_f1score(model, testing_dataset)
    roc_auc_plot(model, testing_dataset)
    model_path = save_model(model, model_saving_path)"""

    # for SVM learning, change
    model_path = "D:/SkinLesionClassification/skin_lesion_detection.h5"
    feature_extractor, vector = load_model(model_path)
    X, Y = feature_lable_extractor(feature_extractor, dataset_path, ['benign', 'malignant'])
    prepare_build_svm(X, Y, ['benign', 'malignant'])


if __name__ == "__main__":
    main()
