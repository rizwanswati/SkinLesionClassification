"""
    Augmentation of images
    Generate multiple images from one image via changing different parameters
    using keras preprocessing image
    Dated : @8/16/2023
    Author : @Mahnoor Khan
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import load_img, img_to_array, save_img
import glob

pathMalignantBlack = "D:/SkinLesionClassification/Augmentation/Malignant/Black/*.png"
pathMalignantAugmented = "D:/SkinLesionClassification/Augmentation/Malignant/"
total_malignant_black = 5
data_gen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,vertical_flip=True)
# now taking image one by one and converting it to tensor
for filePath in glob.glob(pathMalignantBlack):
    srcImg = load_img(filePath, grayscale=False)
    imgArray = img_to_array(srcImg)
    filename = filePath.partition("\\")[2]
    tensorImage = imgArray.reshape((1,)+imgArray.shape)

    for i, _ in enumerate(data_gen.flow(x=tensorImage,
                                        batch_size=1,
                                        save_to_dir=pathMalignantAugmented+"Black/Augmented",
                                        save_prefix="generated",
                                        save_format=".png")):
        if i >= total_malignant_black:
            break
