import tensorflow as tf
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
from keras import models, layers





def main():
    model_path = "D:/SkinLesionClassification/resnet50.h5"
    model_json = "D:/SkinLesionClassification/resnet.json"

    with open(model_json) as json_file:
        model = model_from_json(json_file.read())
        model.load_weights(model_path)

    model.trainable = False
    model.include_top = False


    model.summary()

    # model = tf.keras.models.load_model(model_path)
    # model.layers[0].trainable = False
    # class_names = ["benign", "malignant"]


if __name__ == "__main__":
    main()
