import tensorflow as tf
import numpy as np


def main():
    model_path = "D:/SkinLesionClassification/modelVVG.h5"
    model = tf.keras.models.load_model(model_path)
    class_names = ["benign", "malignant"]
    print(model.summary)


if __name__ == "__main__":
    main()
