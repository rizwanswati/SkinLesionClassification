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

ddi_clean = "D:/DigitalHairRemoval/DDI_Clean/*.png"
ddi_clean_output = "D:/DigitalHairRemoval/DDI_Clean_Output/"
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
    imgForMedianFilter = Image.open(filePath)
    medianFilteredImage = imgForMedianFilter.filter(ImageFilter.MedianFilter(size=3))
    medianFilteredImage.save(ddi_clean_output+"MedianFilteredImages/"+filename)
    # imgForMedianFilter.save(ddi_clean_output+"MedianFilteredImages/"+filename)
    # cv2.imwrite(ddi_clean_output+"MedianFilteredImages/"+filename, medianFilteredImage,
    #             [int(cv2.IMWRITE_JPEG_QUALITY), 90])
