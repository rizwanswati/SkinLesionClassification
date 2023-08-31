import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(244, 244)
    )
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    return image_array


def predict_image(image_path, model, class_names):
    image_array = preprocess_image(image_path)
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence


def evaluate_show_score_plot(model, testDS):
    scores = model.evaluate(testDS)
    # Plotting the Accuracy Score
    plt.figure(figsize=(6, 4))
    plt.bar(["Accuracy"], [scores[1]], color='blue')
    plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
    plt.xlabel("Metrics")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Score")
    plt.show()


def main():
    model_path = "D:/SkinLesionClassification/lesion_detection.h5"
    model = tf.keras.models.load_model(model_path)
    class_names = ["benign", "malignant"]
    image_path = "D:/SkinLesionClassification/Testing/benign/000223.png"
    predicted_class, confidence = predict_image(image_path, model, class_names)
    print("Predicted class:", predicted_class)
    print("Confidence:", confidence)


if __name__ == "__main__":
    main()
