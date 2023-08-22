# -*- coding: utf-8 -*-
"""
Following are the DHR tasks followed in this example code:
    
    -- Applying Morphological Black-Hat transformation
    -- Creating the mask for InPainting task
    -- Applying inpainting algorithm on the image
    -- Applying Median Filter
    -- Applying Blue/Black dot removal methods

"""

import cv2
import glob
from PIL import Image, ImageFilter
import numpy as np


def preprocess_image_dataset(ddi_clean, ddi_clean_output):
    for filePath in glob.glob(ddi_clean):
        src = cv2.imread(filePath)
        filename = filePath.partition("\\")[2]

        # Convert the original image to grayscale
        grayScaleImage = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        # don't need to write output of every step but for proofreading lets do it anyway.
        cv2.imwrite(ddi_clean_output + "GrayScales/" + filename, grayScaleImage,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # Kernel for the morphological filtering
        kernel = cv2.getStructuringElement(1, (17, 17))

        # # Perform the blackHat filtering on the grayscale image to find the
        # # hair countours
        blackhatImage = cv2.morphologyEx(grayScaleImage, cv2.MORPH_BLACKHAT, kernel)
        cv2.imwrite(ddi_clean_output + "BlackHatFiltered/" + filename, blackhatImage,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # intensify the hair countours in preparation for the inpainting
        # algorithm
        ret, thresh2Image = cv2.threshold(blackhatImage, 10, 255, cv2.THRESH_BINARY)
        cv2.imwrite(ddi_clean_output + "ThresholdIntensified/" + filename, thresh2Image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # # inpaint the original image depending on the mask
        inpaintedImage = cv2.inpaint(src, thresh2Image, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(ddi_clean_output + "Inpainted/" + filename, inpaintedImage,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # Blue color removal
        blueImage = cv2.imread(ddi_clean_output + "Inpainted/" + filename)
        hsv = cv2.cvtColor(blueImage, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([150, 255, 255])

        kernel = np.ones((3, 3), np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_blue2 = cv2.dilate(mask_blue, kernel, iterations=5)
        dst = cv2.inpaint(blueImage, mask_blue2, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(ddi_clean_output + "BlueMarksRemoved/" + filename, dst,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # purple color removal
        purple = cv2.imread(ddi_clean_output + "BlueMarksRemoved/" + filename)

        hsv = cv2.cvtColor(purple, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([129, 50, 70])
        upper_blue = np.array([158, 255, 255])
        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_blue2 = cv2.dilate(mask_blue, kernel, iterations=6)
        dst = cv2.inpaint(purple, mask_blue2, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(ddi_clean_output + "PurpleRemoved/" + filename, dst,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # black color removal
        black = cv2.imread(ddi_clean_output + "PurpleRemoved/" + filename)

        hsv = cv2.cvtColor(black, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([0, 0, 0])
        upper_blue = np.array([180, 255, 30])
        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_blue2 = cv2.dilate(mask_blue, kernel, iterations=6)
        dst = cv2.inpaint(black, mask_blue2, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(ddi_clean_output + "BlackMarkRemoved/" + filename, dst,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        imgForMedianFilter = Image.open(ddi_clean_output + "BlackMarkRemoved/" + filename)
        medianFilteredImage = imgForMedianFilter.filter(ImageFilter.MedianFilter(size=1))
        medianFilteredImage.save(ddi_clean_output + "MedianFilteredImages/" + filename)


def main():
    ddi_clean = "D:/SkinLesionClassification/DDI_Clean/*.png"
    ddi_clean_output = "D:/SkinLesionClassification/DDI_Clean_Output/"
    preprocess_image_dataset(ddi_clean, ddi_clean_output)


if __name__ == '__main__':
    main()
