import tensorflow as tf
from keras import models, layers
# from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix


def precision_recall_and_f1score(model, test_ds):
    test_images = []
    test_labels = []
    predicted_labels = []
    predicted_probabilities = []

    for images_batch, labels_batch in test_ds:
        test_images.extend(images_batch.numpy())
        test_labels.extend(labels_batch.numpy())
        batch_prediction = model.predict(images_batch)
        predicted_labels.extend(np.argmax(batch_prediction, axis=1))
        predicted_probabilities.extend(batch_prediction)

    test_labels = np.array(test_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probabilities = np.array(predicted_probabilities)

    precision = precision_score(test_labels, predicted_labels, average='weighted')
    print("Precision:", precision)
    recall = recall_score(test_labels, predicted_labels, average='weighted')
    print("Recall:", recall)
    f1 = f1_score(test_labels, predicted_labels, average='weighted')
    print("F1-Score:", f1)
    # specificity = recall = recall_score(test_labels, predicted_labels, average='binary', pos_label=0)
    # print(specificity)
    fpr, tpr, thresholds = roc_curve(test_labels, predicted_probabilities[:, 1])
    auc_score = roc_auc_score(test_labels, predicted_probabilities[:, 1])

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


def initialize_testing_dataset(data_path, IMAGE_SIZE, BATCH_SIZE):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(data_path, labels="inferred", seed=123, shuffle=True,
                                                                  image_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                  batch_size=BATCH_SIZE
                                                                  )
    return dataset


def main():
    testing_dataset_path = "D:/SkinLesionClassification/ISIC"
    model = models.load_model("D:/SkinLesionClassification/skin_lesion_detection.h5")

    test_ds = initialize_testing_dataset(testing_dataset_path, 224, 32)
    precision_recall_and_f1score(model, test_ds)


if __name__ == "__main__":
    main()
