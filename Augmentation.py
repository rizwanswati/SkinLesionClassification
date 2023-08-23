"""
    Augmentation of images
    Generate multiple images from one image via changing different parameters
    using keras preprocessing image
    Dated : @8/16/2023
    Author : @Mahnoor Khan
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import load_img, img_to_array
import glob


def augment(folderPath, outputPath, iteration):
    data_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.01, horizontal_flip=True,
                                  vertical_flip=True)
    # now taking image one by one and converting it to tensor
    for filePath in glob.glob(folderPath):
        srcImg = load_img(filePath, grayscale=False)
        imgArray = img_to_array(srcImg)
        filename = filePath.partition("\\")[2]
        tensorImage = imgArray.reshape((1,) + imgArray.shape)

        for i, _ in enumerate(data_gen.flow(x=tensorImage,
                                            batch_size=1,
                                            save_to_dir=outputPath,
                                            save_prefix=filename + "_",
                                            save_format=".png")):
            if i >= iteration:
                break


def main():
    folder_path_benign = [
        "D:/SkinLesionClassification/Augmentation/Benign/Black/*.png",
        "D:/SkinLesionClassification/Augmentation/Benign/Brown/*.png",
        "D:/SkinLesionClassification/Augmentation/Benign/White/*.png"
    ]

    folder_path_malignant = [
        "D:/SkinLesionClassification/Augmentation/Malignant/Black/*.png",
        "D:/SkinLesionClassification/Augmentation/Malignant/Brown/*.png",
        "D:/SkinLesionClassification/Augmentation/Malignant/White/*.png"
    ]

    output_path_benign = "D:/SkinLesionClassification/TrainingDS/benign/"
    output_path_malignant = "D:/SkinLesionClassification/TrainingDS/malignant/"

    for folderPath in folder_path_benign:
        augment(folderPath, output_path_benign, 1)

    for folderPath in folder_path_malignant:
        augment(folderPath, output_path_malignant, 5)


if __name__ == '__main__':
    main()
