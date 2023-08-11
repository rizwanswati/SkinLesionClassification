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

ddi_clean = "D:/SkinLesionClassification/DDI_Clean/*.png"
ddi_clean_output = "D:/SkinLesionClassification/DDI_Clean_Output/"
imgs = []

for filePath in glob.glob(ddi_clean):
    src = cv2.imread(filePath)
    filename = filePath.partition("\\")[2]

    # Convert the original image to grayscale
    grayScaleImage = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    # don't need to write output of every step but for proofreading lets do it anyway.
    cv2.imwrite(ddi_clean_output+"GrayScales/"+filename, grayScaleImage,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1,(17,17))

    # # Perform the blackHat filtering on the grayscale image to find the
    # # hair countours
    blackhatImage = cv2.morphologyEx(grayScaleImage, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite(ddi_clean_output+"BlackHatFiltered/"+filename, blackhatImage,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret,thresh2Image = cv2.threshold(blackhatImage,10,255,cv2.THRESH_BINARY)
    cv2.imwrite(ddi_clean_output+"ThresholdIntensified/"+filename, thresh2Image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # # inpaint the original image depending on the mask
    inpaintedImage = cv2.inpaint(src,thresh2Image,1,cv2.INPAINT_TELEA)
    cv2.imwrite(ddi_clean_output+"Inpainted/"+filename, inpaintedImage,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # # Now applying Median Filter to Smoothen up the image
    # # creating an image object
    imgForMedianFilter = Image.open(ddi_clean_output+"Inpainted/"+filename)
    medianFilteredImage = imgForMedianFilter.filter(ImageFilter.MedianFilter(size=3))
    medianFilteredImage.save(ddi_clean_output+"MedianFilteredImages/"+filename)

    # Removing Blue marks from the photo using replacement pixels technique and inpainting
    # Method 1: Simplistic overpaint blue with white
    imageForBlueRemoval = cv2.imread(ddi_clean_output+"MedianFilteredImages/"+filename)
    # Define lower and upper limits of our blue
    # BlueMin = np.array([90, 200, 200], np.uint8)
    # BlueMax = np.array([100, 255, 255], np.uint8)
    #
    # # Go to HSV colourspace and get mask of blue pixels
    # HSV = cv2.cvtColor(imageForBlueRemoval, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(HSV, BlueMin, BlueMax)
    #
    # SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # mask = cv2.dilate(mask, kernel, iterations=1)
    #
    # # Make all pixels in mask white
    # imageForBlueRemoval[mask > 0] = [255, 255, 255]

    imageForBlueRemoval[np.all(imageForBlueRemoval == (255, 0, 0), axis=-1)] = np.uint8([255, 255, 255])
    mask = np.all(imageForBlueRemoval == (255, 0, 0), axis=-1)
    blueDotsRemovedImage = cv2.inpaint(imageForBlueRemoval, np.uint8(mask) * 255, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(ddi_clean_output+"BlueMarksRemoved/"+filename, blueDotsRemovedImage,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])